local hf_model_name = 'Qwen/Qwen2.5-0.5B-Instruct';
local task = (import 'tasks/gsm8k_orig_format.jsonnet');
local total_num_iterations = 650;


(import 'polIter_rho1bSft2_ppo_MATH.jsonnet')
+ {
    episode_generator+: {
        // Override the task
        task: task,
        reward_function+: { math_task: $.episode_generator.task },
        append_bos_to_query: false,

        initial_model_name_or_path: hf_model_name,

        inference_strategy+: {
            guidance_llm: (import 'guidance_llms/qwen2.5-0.5b-instruct.jsonnet') + { api_base: 'none' },
            // GSM8K reward extraction expects `#### <answer>`.
            question_template: '[MATH_TASK] Problem:\n{query}\n\nSolution: Think step by step and end with `#### <final answer>`.',
            node_expander+: {
                program_kwargs+: {
                    // Prevent spillover into the next prompt block.
                    stop: '"Problem:"',
                    max_tokens: 512,
                },
            },
        },
    },
    num_iterations: total_num_iterations,
    tokenizer+: {
        hf_model_name: hf_model_name,
    },
}
+ (import 'sft_qwen2p5_0p5b_instruct_for_gsm8k_eval.jsonnet')
+ (import 'trainers/lam1.jsonnet')
+ (import 'trainers/refKl0.0001.jsonnet')
+ (import 'trainers/klLoss.jsonnet')
