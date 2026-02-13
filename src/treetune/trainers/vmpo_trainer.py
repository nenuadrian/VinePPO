import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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

    # VMPO trust region dual (decoupled analogue for LLMs)
    # alpha_mu constrains average log-prob shift
    epsilon_alpha_mu: Optional[float] = None
    alpha_mu_init: Optional[float] = None
    alpha_mu_lr: Optional[float] = None

    # alpha_sigma constrains dispersion of log-prob shift
    epsilon_alpha_sigma: Optional[float] = None
    alpha_sigma_init: Optional[float] = None
    alpha_sigma_lr: Optional[float] = None

    # Backward-compat aliases (single-alpha VMPO)
    epsilon_alpha_kl: float = 0.01
    alpha_init: float = 1.0
    alpha_lr: float = 1e-4

    # Numerical stability for duals and advantage weighting
    dual_min: float = 1e-8
    dual_max: float = 1e6
    clip_advantages_min: Optional[float] = None
    clip_advantages_max: Optional[float] = None

    # LLM critic normalization (PopArt)
    use_popart: bool = True
    popart_beta: float = 3e-4
    popart_eps: float = 1e-4
    popart_min_sigma: float = 1e-4

    def __post_init__(self):
        super().__post_init__()
        assert 0 < self.topk_fraction <= 1.0, "topk_fraction must be in (0, 1]."
        assert self.epsilon_eta > 0, "epsilon_eta must be positive."
        if self.epsilon_alpha_mu is None:
            self.epsilon_alpha_mu = self.epsilon_alpha_kl
        if self.epsilon_alpha_sigma is None:
            self.epsilon_alpha_sigma = self.epsilon_alpha_kl
        if self.alpha_mu_init is None:
            self.alpha_mu_init = self.alpha_init
        if self.alpha_sigma_init is None:
            self.alpha_sigma_init = self.alpha_init
        if self.alpha_mu_lr is None:
            self.alpha_mu_lr = self.alpha_lr
        if self.alpha_sigma_lr is None:
            self.alpha_sigma_lr = self.alpha_lr

        assert self.epsilon_alpha_mu > 0, "epsilon_alpha_mu must be positive."
        assert self.epsilon_alpha_sigma > 0, "epsilon_alpha_sigma must be positive."
        assert self.temperature_init > 0, "temperature_init must be positive."
        assert self.alpha_mu_init > 0, "alpha_mu_init must be positive."
        assert self.alpha_sigma_init > 0, "alpha_sigma_init must be positive."
        assert self.temperature_lr > 0, "temperature_lr must be positive."
        assert self.alpha_mu_lr > 0, "alpha_mu_lr must be positive."
        assert self.alpha_sigma_lr > 0, "alpha_sigma_lr must be positive."
        assert self.dual_min > 0, "dual_min must be positive."
        assert self.dual_max > self.dual_min, "dual_max must be > dual_min."
        assert 0 < self.popart_beta <= 1.0, "popart_beta must be in (0, 1]."
        assert self.popart_eps > 0, "popart_eps must be positive."
        assert self.popart_min_sigma > 0, "popart_min_sigma must be positive."


class _VMPODualState:
    def __init__(self, trainer: "VMPOTrainer"):
        self._trainer = trainer

    def state_dict(self) -> Dict[str, float]:
        return {
            "log_temperature": float(
                self._trainer.log_temperature.detach().cpu().item()
            ),
            "log_alpha_mu": float(self._trainer.log_alpha_mu.detach().cpu().item()),
            "log_alpha_sigma": float(
                self._trainer.log_alpha_sigma.detach().cpu().item()
            ),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._trainer.log_temperature.data = torch.tensor(
            state_dict["log_temperature"],
            device=self._trainer.log_temperature.device,
            dtype=self._trainer.log_temperature.dtype,
        )
        # Backward compatibility with the previous single-alpha checkpoint.
        if "log_alpha_mu" in state_dict and "log_alpha_sigma" in state_dict:
            log_alpha_mu = state_dict["log_alpha_mu"]
            log_alpha_sigma = state_dict["log_alpha_sigma"]
        elif "log_alpha_kl" in state_dict:
            log_alpha_mu = state_dict["log_alpha_kl"]
            log_alpha_sigma = state_dict["log_alpha_kl"]
        else:
            raise KeyError(
                "Could not find alpha dual values in checkpoint state dict."
            )

        self._trainer.log_alpha_mu.data = torch.tensor(
            log_alpha_mu,
            device=self._trainer.log_alpha_mu.device,
            dtype=self._trainer.log_alpha_mu.dtype,
        )
        self._trainer.log_alpha_sigma.data = torch.tensor(
            log_alpha_sigma,
            device=self._trainer.log_alpha_sigma.device,
            dtype=self._trainer.log_alpha_sigma.dtype,
        )


class _PopArtState:
    def __init__(self, trainer: "VMPOTrainer"):
        self._trainer = trainer

    def state_dict(self) -> Dict[str, float]:
        return {
            "mu": float(self._trainer.popart_mu.detach().cpu().item()),
            "nu": float(self._trainer.popart_nu.detach().cpu().item()),
            "sigma": float(self._trainer.popart_sigma.detach().cpu().item()),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._trainer.popart_mu.copy_(torch.tensor(
            state_dict["mu"],
            device=self._trainer.popart_mu.device,
            dtype=self._trainer.popart_mu.dtype,
        ))
        self._trainer.popart_nu.copy_(torch.tensor(
            state_dict["nu"],
            device=self._trainer.popart_nu.device,
            dtype=self._trainer.popart_nu.dtype,
        ))
        sigma = max(
            float(state_dict["sigma"]),
            self._trainer.vmpo_hparams.popart_min_sigma,
        )
        self._trainer.popart_sigma.copy_(torch.tensor(
            sigma,
            device=self._trainer.popart_sigma.device,
            dtype=self._trainer.popart_sigma.dtype,
        ))


@Trainer.register("vmpo")
class VMPOTrainer(PPOTrainer):
    algorithm_name: str = "VMPO"

    @staticmethod
    def _inverse_softplus(x: float) -> float:
        # Numerically stable inverse of softplus for x > 0.
        if x <= 0:
            raise ValueError("inverse_softplus is only defined for positive inputs.")
        if x > 20.0:
            return x
        return math.log(math.expm1(x))

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
        self._dual_grad_accum_steps = max(
            1,
            int(getattr(self.args, "gradient_accumulation_steps", 1) or 1),
        )
        self._dual_grads_accumulating = False

        def _init_dual_raw(target_value: float) -> float:
            clamped_target = min(
                max(float(target_value), self.vmpo_hparams.dual_min),
                self.vmpo_hparams.dual_max,
            )
            pre_softplus = max(
                clamped_target - self.vmpo_hparams.dual_min,
                1e-12,
            )
            return self._inverse_softplus(pre_softplus)

        self.log_temperature = torch.nn.Parameter(
            torch.tensor(
                _init_dual_raw(self.vmpo_hparams.temperature_init),
                dtype=torch.float32,
                device=dual_device,
            )
        )
        self.log_alpha_mu = torch.nn.Parameter(
            torch.tensor(
                _init_dual_raw(self.vmpo_hparams.alpha_mu_init),
                dtype=torch.float32,
                device=dual_device,
            )
        )
        self.log_alpha_sigma = torch.nn.Parameter(
            torch.tensor(
                _init_dual_raw(self.vmpo_hparams.alpha_sigma_init),
                dtype=torch.float32,
                device=dual_device,
            )
        )

        self.temperature_optimizer = torch.optim.Adam(
            [self.log_temperature],
            lr=self.vmpo_hparams.temperature_lr,
            eps=1e-5,
        )
        self.alpha_mu_optimizer = torch.optim.Adam(
            [self.log_alpha_mu],
            lr=self.vmpo_hparams.alpha_mu_lr,
            eps=1e-5,
        )
        self.alpha_sigma_optimizer = torch.optim.Adam(
            [self.log_alpha_sigma],
            lr=self.vmpo_hparams.alpha_sigma_lr,
            eps=1e-5,
        )
        self._vmpo_dual_state = _VMPODualState(self)

        self.popart_mu = torch.tensor(0.0, dtype=torch.float32, device=dual_device)
        self.popart_nu = torch.tensor(1.0, dtype=torch.float32, device=dual_device)
        self.popart_sigma = torch.tensor(
            1.0, dtype=torch.float32, device=dual_device
        )
        self._popart_state = _PopArtState(self)
        self._has_logged_missing_value_head = False

    def _positive_dual(self, dual_logvar: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            F.softplus(dual_logvar) + self.vmpo_hparams.dual_min,
            min=self.vmpo_hparams.dual_min,
            max=self.vmpo_hparams.dual_max,
        )

    def _begin_dual_accumulation_if_needed(self) -> None:
        if self._dual_grads_accumulating:
            return
        self.temperature_optimizer.zero_grad(set_to_none=True)
        self.alpha_mu_optimizer.zero_grad(set_to_none=True)
        self.alpha_sigma_optimizer.zero_grad(set_to_none=True)
        self._dual_grads_accumulating = True

    def _finalize_dual_updates_if_boundary(self, *, is_grad_acc_boundary: bool) -> None:
        if not is_grad_acc_boundary:
            return
        if self._dual_grads_accumulating:
            self._sync_grads(
                [self.log_temperature, self.log_alpha_mu, self.log_alpha_sigma]
            )
            self.temperature_optimizer.step()
            self.alpha_mu_optimizer.step()
            self.alpha_sigma_optimizer.step()
        self._dual_grads_accumulating = False

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

    def _find_value_head(self, critic: DeepSpeedEngine) -> Optional[torch.nn.Linear]:
        model = critic.module if isinstance(critic, DeepSpeedEngine) else critic

        candidate_names = ["value_head", "reward_head", "v_head", "score"]
        for attr_name in candidate_names:
            module = getattr(model, attr_name, None)
            if isinstance(module, torch.nn.Linear) and module.out_features == 1:
                return module

        for name, module in model.named_modules():
            if name.endswith("value_head") and isinstance(module, torch.nn.Linear):
                if module.out_features == 1:
                    return module

        return None

    def _popart_normalize(self, values: torch.Tensor) -> torch.Tensor:
        sigma = torch.clamp(
            self.popart_sigma,
            min=self.vmpo_hparams.popart_min_sigma,
        )
        return (values - self.popart_mu) / sigma

    def _popart_denormalize(self, normalized_values: torch.Tensor) -> torch.Tensor:
        sigma = torch.clamp(
            self.popart_sigma,
            min=self.vmpo_hparams.popart_min_sigma,
        )
        return normalized_values * sigma + self.popart_mu

    def _compute_global_mean_sq(self, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if values.numel() == 0:
            mean = torch.tensor(0.0, dtype=torch.float32, device=self.popart_mu.device)
            sq_mean = torch.tensor(1.0, dtype=torch.float32, device=self.popart_mu.device)
            return mean, sq_mean

        values = values.to(torch.float32)
        sum_val = values.sum()
        sq_sum_val = (values * values).sum()
        count = torch.tensor(float(values.numel()), device=values.device, dtype=torch.float32)

        if dist.is_initialized():
            stats = torch.stack([sum_val, sq_sum_val, count])
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            global_count = torch.clamp(stats[2], min=1.0)
            mean = stats[0] / global_count
            sq_mean = stats[1] / global_count
        else:
            mean = sum_val / count
            sq_mean = sq_sum_val / count

        return mean.detach(), sq_mean.detach()

    def _update_popart_stats_and_rescale_head(
        self,
        critic: DeepSpeedEngine,
        returns: torch.Tensor,
        action_mask: torch.Tensor,
    ):
        if not self.vmpo_hparams.use_popart:
            return

        valid_returns = returns[action_mask.bool()]
        if valid_returns.numel() == 0:
            return

        old_mu = self.popart_mu.detach().clone()
        old_sigma = torch.clamp(
            self.popart_sigma.detach().clone(),
            min=self.vmpo_hparams.popart_min_sigma,
        )

        batch_mean, batch_sq_mean = self._compute_global_mean_sq(valid_returns)
        beta = self.vmpo_hparams.popart_beta

        new_mu = (1.0 - beta) * old_mu + beta * batch_mean
        new_nu = (1.0 - beta) * self.popart_nu.detach() + beta * batch_sq_mean
        new_var = torch.clamp(
            new_nu - new_mu.square(),
            min=self.vmpo_hparams.popart_eps,
        )
        new_sigma = torch.clamp(
            torch.sqrt(new_var),
            min=self.vmpo_hparams.popart_min_sigma,
        )

        value_head = self._find_value_head(critic)
        if value_head is None:
            if not self._has_logged_missing_value_head:
                logger.warning(
                    "Could not find critic value head for PopArt rescaling. "
                    "Falling back to value-target normalization only."
                )
                self._has_logged_missing_value_head = True
        else:
            with torch.no_grad():
                scale = old_sigma / new_sigma
                value_head.weight.mul_(scale)
                value_head.bias.mul_(scale)
                value_head.bias.add_((old_mu - new_mu) / new_sigma)

        self.popart_mu.copy_(new_mu.detach())
        self.popart_nu.copy_(new_nu.detach())
        self.popart_sigma.copy_(new_sigma.detach())

    def _forward_pass_critic(
        self,
        model_engine: DeepSpeedEngine,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        output = super()._forward_pass_critic(model_engine, inputs)

        if not self.vmpo_hparams.use_popart:
            return output

        normalized_values = output["values"]
        values = self._popart_denormalize(normalized_values)
        output["normalized_values"] = normalized_values
        output["values"] = values
        return output

    def _compute_critics_loss(
        self,
        critic: DeepSpeedEngine,
        model_inputs: Dict[str, torch.Tensor],
        shifted_labels_mask: torch.LongTensor,
        old_valid_values: torch.FloatTensor,
        returns: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.Tensor]]:
        action_mask = shifted_labels_mask

        if self.vmpo_hparams.use_popart:
            with torch.no_grad():
                self._update_popart_stats_and_rescale_head(critic, returns, action_mask)

        if "labels" in model_inputs:
            del model_inputs["labels"]
        outputs = self._forward_pass_critic(critic, model_inputs)

        valid_values = outputs["values"][:, :-1]

        assert valid_values.shape == old_valid_values.shape
        assert action_mask.shape == valid_values.shape

        if self.vmpo_hparams.use_popart and "normalized_values" in outputs:
            valid_values_hat = outputs["normalized_values"][:, :-1]
            returns_hat = self._popart_normalize(returns)
            old_values_hat = self._popart_normalize(old_valid_values)
            value_clip_range_hat = (
                self.ppo_hparams.cliprange_value
                / torch.clamp(
                    self.popart_sigma,
                    min=self.vmpo_hparams.popart_min_sigma,
                )
            )
            values_hat_clipped = torch.clamp(
                valid_values_hat,
                old_values_hat - value_clip_range_hat,
                old_values_hat + value_clip_range_hat,
            )
            vf_losses1 = (valid_values_hat - returns_hat) ** 2
            vf_losses2 = (values_hat_clipped - returns_hat) ** 2
            mse_metric = masked_mean((valid_values - returns) ** 2, action_mask)
            mse_norm_metric = masked_mean(vf_losses1, action_mask)
        else:
            values_clipped = torch.clamp(
                valid_values,
                old_valid_values - self.ppo_hparams.cliprange_value,
                old_valid_values + self.ppo_hparams.cliprange_value,
            )
            vf_losses1 = (valid_values - returns) ** 2
            vf_losses2 = (values_clipped - returns) ** 2
            mse_metric = masked_mean(vf_losses1, action_mask)
            mse_norm_metric = None

        vf_losses = torch.max(vf_losses1, vf_losses2)
        vf_loss = 0.5 * masked_mean(vf_losses, action_mask)

        vf_clip_frac = masked_mean(
            torch.gt(vf_losses2, vf_losses1).float(),
            action_mask,
        )

        metrics = {
            "critic/value": masked_mean(valid_values, action_mask).detach(),
            "critic/mse": mse_metric.detach(),
            "critic/clip_frac": vf_clip_frac.detach(),
        }

        if mse_norm_metric is not None:
            metrics["critic/mse_normalized"] = mse_norm_metric.detach()
            metrics["critic/popart_mu"] = self.popart_mu.detach().clone()
            metrics["critic/popart_sigma"] = self.popart_sigma.detach().clone()

        return vf_loss, metrics

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
        is_grad_acc_boundary = actor.is_gradient_accumulation_boundary()

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
                "actor/vmpo_policy_kl_mu_selected": zero,
                "actor/vmpo_policy_kl_mu_all": zero,
                "actor/vmpo_policy_kl_sigma_selected": zero,
                "actor/vmpo_policy_kl_sigma_all": zero,
                "actor/vmpo_dual_eta_loss": zero,
                "actor/vmpo_dual_alpha_mu_loss": zero,
                "actor/vmpo_dual_alpha_sigma_loss": zero,
                "vmpo/temperature": zero,
                "vmpo/alpha_mu": zero,
                "vmpo/alpha_sigma": zero,
                "vmpo/epsilon_eta": torch.tensor(
                    self.vmpo_hparams.epsilon_eta,
                    device=logprobs.device,
                    dtype=logprobs.dtype,
                ),
                "vmpo/epsilon_alpha_mu": torch.tensor(
                    self.vmpo_hparams.epsilon_alpha_mu,
                    device=logprobs.device,
                    dtype=logprobs.dtype,
                ),
                "vmpo/epsilon_alpha_sigma": torch.tensor(
                    self.vmpo_hparams.epsilon_alpha_sigma,
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
            self._finalize_dual_updates_if_boundary(
                is_grad_acc_boundary=is_grad_acc_boundary
            )
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
                "actor/vmpo_dual_alpha_mu_loss": empty_metric,
                "actor/vmpo_dual_alpha_sigma_loss": empty_metric,
                "vmpo/temperature": empty_metric,
                "vmpo/alpha_mu": empty_metric,
                "vmpo/alpha_sigma": empty_metric,
                "vmpo/selected_fraction": empty_metric,
                "vmpo/selected_adv_threshold": empty_metric,
                "vmpo/effective_sample_size": empty_metric,
            }
            if "entropy" in outputs:
                metrics["actor/logit_entropy"] = outputs["entropy"].detach()
            self._finalize_dual_updates_if_boundary(
                is_grad_acc_boundary=is_grad_acc_boundary
            )
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

        self._begin_dual_accumulation_if_needed()
        dual_loss_scale = 1.0 / float(self._dual_grad_accum_steps)
        (eta_dual_loss * dual_loss_scale).backward()

        with torch.no_grad():
            eta_detached = self._positive_dual(self.log_temperature.detach())
            weights = torch.softmax(selected_advantages.detach() / eta_detached, dim=0)
            effective_sample_size = 1.0 / (weights.pow(2).sum() + 1e-12)
            selected_fraction = float(topk) / float(num_valid)
            adv_std_over_temperature = (
                flat_advantages.std(unbiased=False) / (eta_detached + 1e-12)
            )

        weighted_nll = -(weights.detach() * selected_logprobs).sum()

        delta_logprob_all = flat_logprobs - flat_old_logprobs.detach()
        delta_logprob_selected = selected_logprobs - selected_old_logprobs.detach()

        # LLM analogue of decoupled KL terms:
        # - mu-like term: first moment of log-prob shift (on-policy KL proxy)
        # - sigma-like term: dispersion of log-prob shift
        policy_kl_mu_all = (-delta_logprob_all).mean()
        policy_kl_mu_selected = (
            -delta_logprob_selected
        ).mean()
        centered_delta_selected = (
            delta_logprob_selected - delta_logprob_selected.mean()
        )
        centered_delta_all = delta_logprob_all - delta_logprob_all.mean()
        policy_kl_sigma_selected = 0.5 * centered_delta_selected.pow(2).mean()
        policy_kl_sigma_all = 0.5 * centered_delta_all.pow(2).mean()

        alpha_mu = self._positive_dual(self.log_alpha_mu)
        alpha_sigma = self._positive_dual(self.log_alpha_sigma)
        alpha_mu_dual_loss = alpha_mu * (
            self.vmpo_hparams.epsilon_alpha_mu - policy_kl_mu_selected.detach()
        )
        alpha_sigma_dual_loss = alpha_sigma * (
            self.vmpo_hparams.epsilon_alpha_sigma
            - policy_kl_sigma_selected.detach()
        )

        (alpha_mu_dual_loss * dual_loss_scale).backward()
        (alpha_sigma_dual_loss * dual_loss_scale).backward()
        self._finalize_dual_updates_if_boundary(
            is_grad_acc_boundary=is_grad_acc_boundary
        )

        with torch.no_grad():
            alpha_mu_detached = self._positive_dual(self.log_alpha_mu.detach())
            alpha_sigma_detached = self._positive_dual(self.log_alpha_sigma.detach())

        pg_loss = (
            weighted_nll
            + alpha_mu_detached * policy_kl_mu_selected
            + alpha_sigma_detached * policy_kl_sigma_selected
        ).mean()

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
            "actor/vmpo_policy_kl_mu_selected": policy_kl_mu_selected.detach(),
            "actor/vmpo_policy_kl_mu_all": policy_kl_mu_all.detach(),
            "actor/vmpo_policy_kl_sigma_selected": policy_kl_sigma_selected.detach(),
            "actor/vmpo_policy_kl_sigma_all": policy_kl_sigma_all.detach(),
            "actor/vmpo_dual_eta_loss": eta_dual_loss.detach(),
            "actor/vmpo_dual_alpha_mu_loss": alpha_mu_dual_loss.detach(),
            "actor/vmpo_dual_alpha_sigma_loss": alpha_sigma_dual_loss.detach(),
            "vmpo/temperature": eta_detached.detach(),
            "vmpo/alpha_mu": alpha_mu_detached.detach(),
            "vmpo/alpha_sigma": alpha_sigma_detached.detach(),
            "vmpo/epsilon_eta": torch.tensor(
                self.vmpo_hparams.epsilon_eta,
                device=logprobs.device,
                dtype=logprobs.dtype,
            ),
            "vmpo/epsilon_alpha_mu": torch.tensor(
                self.vmpo_hparams.epsilon_alpha_mu,
                device=logprobs.device,
                dtype=logprobs.dtype,
            ),
            "vmpo/epsilon_alpha_sigma": torch.tensor(
                self.vmpo_hparams.epsilon_alpha_sigma,
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
            save_custom_state(self.alpha_mu_optimizer, checkpoint_path, index=22)
            save_custom_state(self.alpha_sigma_optimizer, checkpoint_path, index=23)
            save_custom_state(self._popart_state, checkpoint_path, index=30)

    def _load_training_state(self, checkpoint_path: Path) -> None:
        super()._load_training_state(checkpoint_path)

        # Backward compatibility: older checkpoints might not include all VMPO states.
        try:
            load_custom_state(self._vmpo_dual_state, checkpoint_path, index=20)
            load_custom_state(self.temperature_optimizer, checkpoint_path, index=21)
            load_custom_state(self.alpha_mu_optimizer, checkpoint_path, index=22)
            try:
                load_custom_state(self.alpha_sigma_optimizer, checkpoint_path, index=23)
            except Exception:
                # Old single-alpha optimizer state.
                load_custom_state(self.alpha_sigma_optimizer, checkpoint_path, index=22)

            try:
                load_custom_state(self._popart_state, checkpoint_path, index=30)
            except Exception:
                # Keep default PopArt stats if unavailable.
                pass
        except Exception as exc:
            logger.warning(
                "Could not load VMPO dual states from checkpoint %s: %s",
                checkpoint_path,
                exc,
            )
