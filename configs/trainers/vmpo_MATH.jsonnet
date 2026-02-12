local ds_stage_2_w_cpu_optimizer = (import '../deepspeed/zero_2.jsonnet') + {
    zero_optimization+: {
        offload_optimizer+: {
            device: 'cpu',
            pin_memory: true,
        },
    },
};

{
    trainer+: {
        type: 'vmpo',

        actor_model+: {
            type: 'pretrained_causal_lm',
            disable_dropout: true,
            pretrained_args+: {
                use_flash_attention_2: true,
            },
        },
        actor_deepspeed_config: ds_stage_2_w_cpu_optimizer,

        critic_model+: {
            type: 'pretrained_causal_lm_with_value_head',
            pretrained_backbone_model+: {
                type: 'pretrained_causal_lm',
                disable_dropout: true,
                pretrained_args+: {
                    use_flash_attention_2: true,
                },
            },
        },
        critic_deepspeed_config: ds_stage_2_w_cpu_optimizer,

        reference_model+: {
            type: 'pretrained_causal_lm',
            pretrained_args+: {
                use_flash_attention_2: true,
            },
        },
        reference_deepspeed_config: {
            bf16: { enabled: true },
            wall_clock_breakdown: false,
            prescale_gradients: false,
            gradient_accumulation_steps: 'auto',
            train_batch_size: 'auto',
            train_micro_batch_size_per_gpu: 'auto',
        },

        params+: {
            use_score_norm: false,
            use_score_scaling: false,

            adap_kl_ctrl: false,
            init_kl_coef: 0.05,

            gamma: 1.0,
            lam: 0.96,

            whiten_rewards: false,
            whiten_advantages: true,

            # VMPO-specific settings
            topk_fraction: 0.5,
            epsilon_eta: 0.1,
            temperature_init: 1.0,
            temperature_lr: 1e-4,
            epsilon_alpha_mu: 0.01,
            epsilon_alpha_sigma: 0.01,
            alpha_mu_init: 1.0,
            alpha_sigma_init: 1.0,
            alpha_mu_lr: 1e-4,
            alpha_sigma_lr: 1e-4,
            # Backward-compatible aliases
            epsilon_alpha_kl: 0.01,
            alpha_init: 1.0,
            alpha_lr: 1e-4,

            # PopArt/value-target normalization for LLM critic
            use_popart: true,
            popart_beta: 3e-4,
            popart_eps: 1e-4,
            popart_min_sigma: 1e-4,
        },

        general_training_args: {
            target_train_batch_size: 64,

            per_device_train_batch_size: 8,

            learning_rate: 1e-6,
            weight_decay: 0.00,
            warmup_ratio: 0.03,

            max_grad_norm: 1.0,

            dataloader_num_workers: 1,
            dataloader_pin_memory: false,

            gradient_checkpointing: true,
            bf16: true,

            logging_steps: 1,

            save_steps: 32,
            checkpoint_keep_steps: 32,

            seed: std.parseInt(std.extVar('APP_SEED')),
        },

        num_epochs_per_iteration: 2,
        cache_deepspeed_engines: true,
        move_reference_model_to_cpu: true,
        save_hf_critic_checkpoint: true,
    },
}
