import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from accelerate.checkpointing import load_custom_state, save_custom_state
from deepspeed import DeepSpeedEngine
from deepspeed import comm as dist
from torch.nn import functional as F

from treetune.common import JsonDict
from treetune.logging_utils import get_logger
from treetune.trainers.base_trainer import Trainer
from treetune.trainers.ppo_trainer import PPOHParams, PPOTrainer
from treetune.trainers.utils import masked_mean

logger = get_logger(__name__)


@dataclass
class VMPOHParams(PPOHParams):
    # VMPO E-step
    topk_fraction: float = 0.5
    epsilon_eta: float = 0.1
    temperature_init: float = 1.0
    temperature_lr: float = 1e-4

    # VMPO trust region dual
    epsilon_alpha_kl: float = 0.01
    alpha_init: float = 1.0
    alpha_lr: float = 1e-4

    # Numerical stability for duals and advantage weighting
    dual_min: float = 1e-8
    dual_max: float = 1e6
    clip_advantages_min: Optional[float] = None
    clip_advantages_max: Optional[float] = None

    def __post_init__(self):
        super().__post_init__()
        assert 0 < self.topk_fraction <= 1.0, "topk_fraction must be in (0, 1]."
        assert self.epsilon_eta > 0, "epsilon_eta must be positive."
        assert self.epsilon_alpha_kl > 0, "epsilon_alpha_kl must be positive."
        assert self.temperature_init > 0, "temperature_init must be positive."
        assert self.alpha_init > 0, "alpha_init must be positive."
        assert self.temperature_lr > 0, "temperature_lr must be positive."
        assert self.alpha_lr > 0, "alpha_lr must be positive."
        assert self.dual_min > 0, "dual_min must be positive."
        assert self.dual_max > self.dual_min, "dual_max must be > dual_min."


class _VMPODualState:
    def __init__(self, trainer: "VMPOTrainer"):
        self._trainer = trainer

    def state_dict(self) -> Dict[str, float]:
        return {
            "log_temperature": float(
                self._trainer.log_temperature.detach().cpu().item()
            ),
            "log_alpha_kl": float(self._trainer.log_alpha_kl.detach().cpu().item()),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._trainer.log_temperature.data = torch.tensor(
            state_dict["log_temperature"],
            device=self._trainer.log_temperature.device,
            dtype=self._trainer.log_temperature.dtype,
        )
        self._trainer.log_alpha_kl.data = torch.tensor(
            state_dict["log_alpha_kl"],
            device=self._trainer.log_alpha_kl.device,
            dtype=self._trainer.log_alpha_kl.dtype,
        )


@Trainer.register("vmpo")
class VMPOTrainer(PPOTrainer):
    def __init__(self, params: JsonDict, **kwargs):
        params = dict(params)

        ppo_param_names = {f.name for f in fields(PPOHParams)}
        vmpo_param_names = {f.name for f in fields(VMPOHParams)}

        extra_param_names = sorted(set(params.keys()) - vmpo_param_names)
        if len(extra_param_names) > 0:
            logger.warning(
                "Ignoring unknown VMPO params: %s",
                ", ".join(extra_param_names),
            )

        ppo_params = {k: v for k, v in params.items() if k in ppo_param_names}
        super().__init__(params=ppo_params, **kwargs)

        merged_params = asdict(self.ppo_hparams)
        merged_params.update({k: v for k, v in params.items() if k in vmpo_param_names})
        self.vmpo_hparams = VMPOHParams(**merged_params)

        dual_device = self.distributed_state.device
        self.log_temperature = torch.nn.Parameter(
            torch.tensor(
                np.log(
                    max(
                        self.vmpo_hparams.temperature_init,
                        self.vmpo_hparams.dual_min,
                    )
                ),
                dtype=torch.float32,
                device=dual_device,
            )
        )
        self.log_alpha_kl = torch.nn.Parameter(
            torch.tensor(
                np.log(max(self.vmpo_hparams.alpha_init, self.vmpo_hparams.dual_min)),
                dtype=torch.float32,
                device=dual_device,
            )
        )

        self.temperature_optimizer = torch.optim.Adam(
            [self.log_temperature],
            lr=self.vmpo_hparams.temperature_lr,
            eps=1e-5,
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha_kl],
            lr=self.vmpo_hparams.alpha_lr,
            eps=1e-5,
        )
        self._vmpo_dual_state = _VMPODualState(self)

    def _positive_dual(self, dual_logvar: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            F.softplus(dual_logvar) + self.vmpo_hparams.dual_min,
            min=self.vmpo_hparams.dual_min,
            max=self.vmpo_hparams.dual_max,
        )

    @staticmethod
    def _sync_grads(params):
        if not dist.is_initialized():
            return

        world_size = dist.get_world_size()
        for p in params:
            if p.grad is None:
                continue
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= world_size

    def _compute_actor_loss(
        self,
        actor: DeepSpeedEngine,
        model_inputs: Dict[str, torch.Tensor],
        shifted_labels_mask: torch.LongTensor,
        old_logprobs: torch.FloatTensor,
        ref_logprobs: Optional[torch.FloatTensor],
        advantages: torch.FloatTensor,
    ) -> Tuple[
        torch.FloatTensor, bool, Dict[str, torch.Tensor], Optional[torch.FloatTensor]
    ]:
        action_mask = shifted_labels_mask

        outputs = self._forward_pass_actor(
            actor,
            model_inputs,
            return_all_logps=True,
            return_entropy=self.report_entropy,
        )
        logprobs = outputs["all_logps"]

        assert logprobs.shape == old_logprobs.shape
        assert action_mask.shape == logprobs.shape

        log_ratio = (logprobs - old_logprobs) * action_mask
        ratio = torch.exp(log_ratio)
        avg_ratio = masked_mean(ratio, action_mask)
        approx_kl = 0.5 * masked_mean((logprobs - old_logprobs) ** 2, action_mask)
        policy_kl = masked_mean(old_logprobs - logprobs, action_mask)

        if avg_ratio.item() > self.ppo_hparams.ratio_threshold:
            logger.warning(
                f"High policy ratio detected: {avg_ratio.item():.2f}. Skipping this batch."
            )
            zero = torch.zeros((), device=logprobs.device)
            zero_loss = logprobs.sum() * 0.0
            metrics = {
                "actor/approx_kl": approx_kl.detach(),
                "actor/policy_kl": policy_kl.detach(),
                "actor/clip_frac": zero,
                "actor/ratio": avg_ratio.detach(),
                "actor/vmpo_weighted_nll": zero,
                "actor/vmpo_policy_kl_selected": zero,
                "actor/vmpo_policy_kl_all": zero,
                "actor/vmpo_dual_eta_loss": zero,
                "actor/vmpo_dual_alpha_loss": zero,
                "vmpo/temperature": zero,
                "vmpo/alpha_kl": zero,
                "vmpo/epsilon_eta": torch.tensor(
                    self.vmpo_hparams.epsilon_eta,
                    device=logprobs.device,
                    dtype=logprobs.dtype,
                ),
                "vmpo/epsilon_alpha_kl": torch.tensor(
                    self.vmpo_hparams.epsilon_alpha_kl,
                    device=logprobs.device,
                    dtype=logprobs.dtype,
                ),
                "vmpo/selected_fraction": zero,
                "vmpo/selected_adv_threshold": zero,
                "vmpo/effective_sample_size": zero,
                "vmpo/adv_std_over_temperature": zero,
            }
            if "entropy" in outputs:
                metrics["actor/logit_entropy"] = outputs["entropy"].detach()
            return zero_loss, True, metrics, None

        flat_action_mask = action_mask.bool().reshape(-1)
        flat_advantages = advantages.reshape(-1)[flat_action_mask]
        flat_logprobs = logprobs.reshape(-1)[flat_action_mask]
        flat_old_logprobs = old_logprobs.reshape(-1)[flat_action_mask]

        if flat_advantages.numel() == 0:
            zero_loss = logprobs.sum() * 0.0
            empty_metric = torch.zeros((), device=logprobs.device)
            metrics = {
                "actor/approx_kl": empty_metric,
                "actor/policy_kl": empty_metric,
                "actor/clip_frac": empty_metric,
                "actor/ratio": empty_metric,
                "actor/vmpo_weighted_nll": empty_metric,
                "actor/vmpo_dual_eta_loss": empty_metric,
                "actor/vmpo_dual_alpha_loss": empty_metric,
                "vmpo/temperature": empty_metric,
                "vmpo/alpha_kl": empty_metric,
                "vmpo/selected_fraction": empty_metric,
                "vmpo/selected_adv_threshold": empty_metric,
                "vmpo/effective_sample_size": empty_metric,
            }
            if "entropy" in outputs:
                metrics["actor/logit_entropy"] = outputs["entropy"].detach()
            return zero_loss, True, metrics, None

        if (
            self.vmpo_hparams.clip_advantages_min is not None
            or self.vmpo_hparams.clip_advantages_max is not None
        ):
            flat_advantages = torch.clamp(
                flat_advantages,
                min=(
                    self.vmpo_hparams.clip_advantages_min
                    if self.vmpo_hparams.clip_advantages_min is not None
                    else -torch.inf
                ),
                max=(
                    self.vmpo_hparams.clip_advantages_max
                    if self.vmpo_hparams.clip_advantages_max is not None
                    else torch.inf
                ),
            )

        num_valid = flat_advantages.numel()
        topk = max(1, int(self.vmpo_hparams.topk_fraction * num_valid))
        selected_advantages, selected_indices = torch.topk(
            flat_advantages,
            k=topk,
            largest=True,
            sorted=False,
        )
        selected_logprobs = flat_logprobs[selected_indices]
        selected_old_logprobs = flat_old_logprobs[selected_indices]
        selected_threshold = selected_advantages.min().detach()

        eta = self._positive_dual(self.log_temperature)
        log_count = math.log(float(topk))
        eta_dual_loss = eta * self.vmpo_hparams.epsilon_eta + eta * (
            torch.logsumexp(selected_advantages.detach() / eta, dim=0) - log_count
        )

        self.temperature_optimizer.zero_grad(set_to_none=True)
        eta_dual_loss.backward()
        self._sync_grads([self.log_temperature])
        self.temperature_optimizer.step()

        with torch.no_grad():
            eta_detached = self._positive_dual(self.log_temperature.detach())
            weights = torch.softmax(selected_advantages.detach() / eta_detached, dim=0)
            effective_sample_size = 1.0 / (weights.pow(2).sum() + 1e-12)
            selected_fraction = float(topk) / float(num_valid)
            adv_std_over_temperature = (
                flat_advantages.std(unbiased=False) / (eta_detached + 1e-12)
            )

        weighted_nll = -(weights.detach() * selected_logprobs).sum()

        policy_kl_all = (flat_old_logprobs.detach() - flat_logprobs).mean()
        policy_kl_selected = (
            selected_old_logprobs.detach() - selected_logprobs
        ).mean()

        alpha_kl = self._positive_dual(self.log_alpha_kl)
        alpha_dual_loss = alpha_kl * (
            self.vmpo_hparams.epsilon_alpha_kl - policy_kl_selected.detach()
        )

        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_dual_loss.backward()
        self._sync_grads([self.log_alpha_kl])
        self.alpha_optimizer.step()

        with torch.no_grad():
            alpha_kl_detached = self._positive_dual(self.log_alpha_kl.detach())

        pg_loss = weighted_nll + alpha_kl_detached * policy_kl_selected

        if self.ppo_hparams.kl_penalty_loss_type is not None:
            assert ref_logprobs is not None
            ref_kl = self._compute_kl_penalty(
                logprobs,
                ref_logprobs,
                estimation_type=self.ppo_hparams.kl_penalty_loss_type,
            )
            ref_kl = torch.clamp(
                ref_kl * action_mask,
                min=self.ppo_hparams.kl_penalty_loss_clip_min,
                max=self.ppo_hparams.kl_penalty_loss_clip_max,
            )

            ref_kl_loss = self.kl_ctl.value * ref_kl.sum(dim=1).mean()
            pg_loss = pg_loss + ref_kl_loss
            ref_kl = ref_kl.detach()
        else:
            ref_kl = None
            ref_kl_loss = None

        metrics = {
            "actor/approx_kl": approx_kl.detach(),
            "actor/policy_kl": policy_kl.detach(),
            "actor/clip_frac": torch.zeros((), device=logprobs.device),
            "actor/ratio": avg_ratio.detach(),
            "actor/vmpo_weighted_nll": weighted_nll.detach(),
            "actor/vmpo_policy_kl_selected": policy_kl_selected.detach(),
            "actor/vmpo_policy_kl_all": policy_kl_all.detach(),
            "actor/vmpo_dual_eta_loss": eta_dual_loss.detach(),
            "actor/vmpo_dual_alpha_loss": alpha_dual_loss.detach(),
            "vmpo/temperature": eta_detached.detach(),
            "vmpo/alpha_kl": alpha_kl_detached.detach(),
            "vmpo/epsilon_eta": torch.tensor(
                self.vmpo_hparams.epsilon_eta,
                device=logprobs.device,
                dtype=logprobs.dtype,
            ),
            "vmpo/epsilon_alpha_kl": torch.tensor(
                self.vmpo_hparams.epsilon_alpha_kl,
                device=logprobs.device,
                dtype=logprobs.dtype,
            ),
            "vmpo/selected_fraction": torch.tensor(
                selected_fraction,
                device=logprobs.device,
                dtype=logprobs.dtype,
            ),
            "vmpo/selected_adv_threshold": selected_threshold,
            "vmpo/effective_sample_size": effective_sample_size.detach(),
            "vmpo/adv_std_over_temperature": adv_std_over_temperature.detach(),
        }

        if "entropy" in outputs:
            metrics["actor/logit_entropy"] = outputs["entropy"].detach()
        if ref_kl_loss is not None:
            metrics["actor/ref_kl_loss"] = ref_kl_loss.detach()

        return pg_loss, False, metrics, ref_kl

    def _save_trainer_state(self, checkpoint_path: Path) -> None:
        super()._save_trainer_state(checkpoint_path)
        if self._is_main_process():
            save_custom_state(self._vmpo_dual_state, checkpoint_path, index=20)
            save_custom_state(self.temperature_optimizer, checkpoint_path, index=21)
            save_custom_state(self.alpha_optimizer, checkpoint_path, index=22)

    def _load_training_state(self, checkpoint_path: Path) -> None:
        super()._load_training_state(checkpoint_path)

        # Backward compatibility: old PPO checkpoints don't include VMPO dual states.
        try:
            load_custom_state(self._vmpo_dual_state, checkpoint_path, index=20)
            load_custom_state(self.temperature_optimizer, checkpoint_path, index=21)
            load_custom_state(self.alpha_optimizer, checkpoint_path, index=22)
        except Exception as exc:
            logger.warning(
                "Could not load VMPO dual states from checkpoint %s: %s",
                checkpoint_path,
                exc,
            )
