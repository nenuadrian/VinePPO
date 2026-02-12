(import 'polIter_rho1bSft2_ppo_MATH.jsonnet')
+ {
    trainer+: {
        type: 'vmpo',
        params+: {
            # VMPO-specific settings
            topk_fraction: 0.5,
            epsilon_eta: 0.1,
            temperature_init: 1.0,
            temperature_lr: 1e-4,
            epsilon_alpha_kl: 0.01,
            alpha_init: 1.0,
            alpha_lr: 1e-4,
        },
    },
}
+ (import 'trainers/lam1.jsonnet')
+ (import 'trainers/refKl0.0001.jsonnet')
+ (import 'trainers/klLoss.jsonnet')
