import torch
from vllm import LLM, SamplingParams
import json
import random
import wandb
import sys
import argparse
import yaml
import os
import datetime

from src.math_baseline import evaluate_vllm
from src.vllm_helper import *
from src.sft_helper import tokenize_prompt_and_output, get_response_log_probs
from src.drgrpo_grader import r1_zero_reward_fn
from src.grpo import *


with open('src/prompts/r1_zero.prompt', 'r') as f:
    R1_ZERO_PROMPT = f.read()

def get_starter_params(policy, learning_rate, debug=False):
    params = {
        'n_grpo_steps': 200,
        'learning_rate': learning_rate,
        'advantage_eps': 1e-6,
        'rollout_batch_size': 256,
        'group_size': 8,
        'sampling_temperature': 1.0,
        'sampling_min_tokens': 4,
        'sampling_max_tokens': 512,
        'epochs_per_rollout_batch': 1,
        'train_batch_size': 256,
        'gradient_accumulation_steps': 256,
        'gpu_memory_utilization': 0.8,  # Can use more memory since vLLM has dedicated GPU
        # 'loss_type': 'reinforce_with_baseline',
        'loss_type': 'grpo_clip',
        'use_std_normalization': True,
        'eval_sample_size': 1024,
        'eval_log_frequency': 5,
    }

    params['optimizer'] = torch.optim.AdamW(
        policy.parameters(),
        lr=params['learning_rate'],
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    if debug:
        params['n_grpo_steps'] = 5
        params['rollout_batch_size'] = 1
        params['train_batch_size'] = 1
        params['gradient_accumulation_steps'] = 1
        params['group_size'] = 1
        params['eval_sample_size'] = 16
        params['eval_log_frequency'] = 1
    
    return params

def init_sampling_params(params):
    sampling_params = SamplingParams(
        temperature=params['sampling_temperature'],
        top_p=1.0,
        min_tokens=params['sampling_min_tokens'],
        max_tokens=params['sampling_max_tokens'],
        logprobs=0,
    )
    sampling_params.stop = ["</answer>"]
    sampling_params.include_stop_str_in_output = True

    return sampling_params

def get_jsonl_data(fpath):
    with open(fpath, 'r') as f:
        prompt_data = [json.loads(json_line) for json_line in f]
    
    dataset = []

    for p in prompt_data:
        prompt_string = R1_ZERO_PROMPT.format(
            question=p['question']
        )
        answer_string = p['answer']

        dataset.append({
            'prompt': prompt_string,
            'answer': p['answer'],
        })

    return dataset

def get_training_data():
    return get_jsonl_data('./grade-school-math/grade_school_math/data/train.jsonl')

def get_eval_data():
    return get_jsonl_data('./grade-school-math/grade_school_math/data/test.jsonl')

def sample_dataset(dataset, num_samples):
    sampled_data = random.sample(dataset, num_samples)

    ret = {
        'prompts': [],
        'answers': [],
    }

    for d in sampled_data:
        ret['prompts'].append(d['prompt'])
        ret['answers'].append(d['answer'])

    return ret

def duplicate_data(arr, group_size):
    '''
    Ex: duplicate_data([1, 2, 3], 2) => [1, 1, 2, 2, 3, 3]
    '''

    return [x for x in arr for _ in range(group_size)]


def train_policy(policy, tokenizer, vllm, sampling_params, training_data, training_params,
                experiment_name, eval_data, output_dir):
    assert training_params['train_batch_size'] % training_params['gradient_accumulation_steps'] == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = training_params['train_batch_size'] // training_params['gradient_accumulation_steps']

    assert training_params['rollout_batch_size'] % training_params['group_size'] == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = training_params['rollout_batch_size'] // training_params['group_size']

    assert training_params['train_batch_size'] >= training_params['group_size'], (
        "train_batch_size must be greater than or equal to group_size"
    )
    n_microbatches_per_rollout_batch = training_params['rollout_batch_size'] // micro_train_batch_size

    device = policy.device

    wandb_log_dir = os.path.join(output_dir, 'wandb')
    os.makedirs(wandb_log_dir, exist_ok=True)
    wandb_run = wandb.init(
        project="src",
        config=training_params,
        name=experiment_name,
        dir=wandb_log_dir,
    )

    # Directory to store all models
    model_dir = os.path.join(output_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    # Setup wandb metrics
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    train_step = 0
    eval_step = 0

    for grpo_step_idx in range(training_params['n_grpo_steps']):
        # (was here) load_policy_into_vllm_instance(policy, vllm)
        # >>> CHANGED: defer sync until just before sampling

        if grpo_step_idx % training_params['eval_log_frequency'] == (training_params['eval_log_frequency'] - 1):
            load_policy_into_vllm_instance(policy, vllm)  # <<< CHANGED: sync before eval sampling

            # Sample 100 fixed responses for detailed progress tracking
            eval_fixed_size = 100
            sampled_eval_data = sample_dataset(eval_data, eval_fixed_size)
            prompts_batch = sampled_eval_data['prompts']
            answers_batch = sampled_eval_data['answers']

            vllm_rollouts = vllm.generate(prompts_batch, sampling_params)

            rollout_input_text = []
            rollout_response_text = []

            for rollout in vllm_rollouts:
                for r in rollout.outputs:
                    rollout_input_text.append(rollout.prompt)
                    rollout_response_text.append(r.text)
            
            _, _, reward_metadata = compute_group_normalized_rewards(
                r1_zero_reward_fn,
                rollout_response_text,
                answers_batch,
                1,
                training_params['advantage_eps'],
                training_params['use_std_normalization'],
            )

            # Save all 100 responses to file for progress tracking
            eval_responses_file = os.path.join(output_dir, f'eval_responses_step_{eval_step}.txt')
            with open(eval_responses_file, 'w', encoding='utf-8') as f:
                f.write(f"Evaluation Step: {eval_step}\n")
                f.write(f"GRPO Step: {grpo_step_idx}\n")
                f.write(f"Accuracy: {reward_metadata['mean']:.4f}\n")
                f.write(f"Format Reward: {reward_metadata['format_mean']:.4f}\n")
                f.write(f"Answer Reward: {reward_metadata['answer_mean']:.4f}\n")
                f.write("=" * 80 + "\n\n")
                
                for i in range(eval_fixed_size):
                    # Compute individual reward for this response
                    individual_reward = r1_zero_reward_fn(rollout_response_text[i], answers_batch[i])
                    
                    f.write(f"Sample {i+1}/100:\n")
                    f.write(f"Reward: {individual_reward['reward']:.1f} (format={individual_reward['format_reward']:.1f}, answer={individual_reward['answer_reward']:.1f})\n")
                    f.write("Prompt:\n")
                    f.write(rollout_input_text[i] + "\n")
                    f.write("Correct Answer:\n")
                    f.write(answers_batch[i] + "\n")
                    f.write("LLM Response:\n")
                    f.write(rollout_response_text[i] + "\n")
                    f.write("-" * 80 + "\n\n")
            
            print(f"Saved {eval_fixed_size} evaluation responses to: {eval_responses_file}")

            # Print a randomly sampled eval response (for immediate feedback)
            eval_rand_idx = random.randrange(eval_fixed_size)
            print('Eval step:', eval_step)
            print('Prompt:')
            print(rollout_input_text[eval_rand_idx])
            print('Correct Answer:')
            print(answers_batch[eval_rand_idx])
            print('LLM Response:')
            print(rollout_response_text[eval_rand_idx])

            # Calculate mean length of generated responses
            response_lengths = [len(response.split()) for response in rollout_response_text]
            mean_response_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0

            wandb_run.log({
                'eval_step': eval_step,
                'eval/accuracy': reward_metadata['mean'],
                'eval/format_reward_mean': reward_metadata['format_mean'],
                'eval/answer_reward_mean': reward_metadata['answer_mean'],
                'eval/mean_response_length': mean_response_length,
            })

            # Save model
            curr_model_dir = os.path.join(model_dir, 'eval_step_{}'.format(eval_step))
            # policy.save_pretrained(save_directory=curr_model_dir)
            # tokenizer.save_pretrained(save_directory=curr_model_dir)

            eval_step += 1

        # One policy gradient step per train_batch_size of data
        for rollout_batch_idx in range(0, training_params['train_batch_size'], training_params['rollout_batch_size']):
            load_policy_into_vllm_instance(policy, vllm)  # <<< CHANGED: sync at start of EACH rollout batch

            # Sample a batch of data, then select microbatches later
            sampled_training_data = sample_dataset(training_data, n_prompts_per_rollout_batch)
            prompts_batch = sampled_training_data['prompts']
            answers_batch = sampled_training_data['answers']

            prompts_batch = duplicate_data(prompts_batch, training_params['group_size'])
            answers_batch = duplicate_data(answers_batch, training_params['group_size'])

            vllm_rollouts = vllm.generate(prompts_batch, sampling_params)

            rollout_input_text = []
            rollout_response_text = []

            for rollout in vllm_rollouts:
                for r in rollout.outputs:
                    rollout_input_text.append(rollout.prompt)
                    rollout_response_text.append(r.text)
            
            advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
                r1_zero_reward_fn,
                rollout_response_text,
                answers_batch,
                training_params['group_size'],
                training_params['advantage_eps'],
                training_params['use_std_normalization'],
            )

            wandb_run.log({
                'train_step': train_step,
                'train/reward_mean': reward_metadata['mean']
            })

            rollout_data_tokenized = tokenize_prompt_and_output(
                rollout_input_text,
                rollout_response_text,
                tokenizer
            )

            # --- OOM FIX: cache old_log_probs in micro-chunks (and optionally on CPU) ---
            input_ids_full = rollout_data_tokenized['input_ids'].to(device)
            labels_full = rollout_data_tokenized['labels'].to(device)
            response_mask_full = rollout_data_tokenized['response_mask'].to(device)

            store_old_on_cpu = True                                      # <<< CHANGED: toggle to save GPU RAM
            if store_old_on_cpu:                                         # <<< CHANGED
                old_log_probs_full = torch.empty(labels_full.shape, dtype=torch.float32, device='cpu')
            else:
                old_log_probs_full = torch.empty(labels_full.shape, dtype=torch.float32, device=device)

            with torch.no_grad():                                        # <<< CHANGED (no grad memory)
                for microbatch_idx in range(n_microbatches_per_rollout_batch):  # <<< CHANGED (chunked)
                    microbatch_slice = slice(
                        microbatch_idx * micro_train_batch_size,
                        (microbatch_idx + 1) * micro_train_batch_size
                    )
                    ids_mb = input_ids_full[microbatch_slice]
                    labs_mb = labels_full[microbatch_slice]
                    lp_mb = get_response_log_probs(policy, ids_mb, labs_mb)['log_probs']
                    if store_old_on_cpu:
                        old_log_probs_full[microbatch_slice] = lp_mb.detach().to('cpu')  # <<< CHANGED
                    else:
                        old_log_probs_full[microbatch_slice] = lp_mb.detach()

            for _ in range(training_params['epochs_per_rollout_batch']):
                training_params['optimizer'].zero_grad()

                # Accumulators for logging metrics across microbatches
                rollout_batch_loss = 0
                accumulated_token_entropy = []
                accumulated_raw_rewards = []
                accumulated_advantages = []
                accumulated_clip_fractions = []

                for microbatch_idx in range(n_microbatches_per_rollout_batch):
                    microbatch_slice = slice(
                        microbatch_idx * micro_train_batch_size,
                        (microbatch_idx + 1) * micro_train_batch_size
                    )

                    microbatch_input_ids = input_ids_full[microbatch_slice]
                    microbatch_labels = labels_full[microbatch_slice]
                    microbatch_response_mask = response_mask_full[microbatch_slice]

                    advantages_microbatch = advantages[microbatch_slice].to(device)
                    raw_rewards_microbatch = raw_rewards[microbatch_slice].to(device)

                    policy_log_probs_dict = get_response_log_probs(
                        policy,
                        microbatch_input_ids,
                        microbatch_labels,
                        return_token_entropy=True
                    )
                    policy_log_probs = policy_log_probs_dict['log_probs']
                    policy_token_entropy = policy_log_probs_dict['token_entropy']

                    # Use cached old_log_probs; move from CPU to GPU if needed
                    old_log_probs = old_log_probs_full[microbatch_slice]
                    if store_old_on_cpu:
                        old_log_probs = old_log_probs.to(device)          # <<< CHANGED
                    # Match dtype (policy_log_probs is typically float32)
                    old_log_probs = old_log_probs.to(policy_log_probs.dtype)  # <<< CHANGED

                    advantages_microbatch = advantages_microbatch.unsqueeze(-1)

                    loss, loss_metadata = grpo_microbatch_train_step(
                        policy_log_probs,
                        microbatch_response_mask,
                        training_params['gradient_accumulation_steps'],
                        training_params['loss_type'],
                        raw_rewards_microbatch,
                        advantages_microbatch,
                        old_log_probs,
                        1.0,
                    )

                    rollout_batch_loss += loss.item()

                    # Accumulate metrics for logging
                    valid_token_entropy = policy_token_entropy[microbatch_response_mask.bool()]
                    if len(valid_token_entropy) > 0:
                        accumulated_token_entropy.append(valid_token_entropy.detach().cpu())
                    
                    accumulated_raw_rewards.append(raw_rewards_microbatch.detach().cpu())
                    accumulated_advantages.append(advantages_microbatch.squeeze(-1).detach().cpu())
                    
                    # Track clip fraction if using grpo_clip
                    if training_params['loss_type'] == 'grpo_clip' and 'token_clipped' in loss_metadata:
                        clipped_mask = loss_metadata['token_clipped'][microbatch_response_mask.bool()]
                        if len(clipped_mask) > 0:
                            accumulated_clip_fractions.append(clipped_mask.float().detach().cpu())
        
                # Calculate gradient norm before optimizer step
                total_grad_norm = 0.0
                for param in policy.parameters():
                    if param.grad is not None:
                        param_grad_norm = param.grad.data.norm(2)
                        total_grad_norm += param_grad_norm.item() ** 2
                total_grad_norm = total_grad_norm ** (1. / 2)

                training_params['optimizer'].step()

                # Prepare logging metrics
                rollout_batch_loss /= n_microbatches_per_rollout_batch
                
                # Aggregate accumulated metrics
                all_token_entropy = torch.cat(accumulated_token_entropy) if accumulated_token_entropy else torch.tensor([])
                all_raw_rewards = torch.cat(accumulated_raw_rewards)
                all_advantages = torch.cat(accumulated_advantages)
                
                # Calculate mean length of generated responses
                response_lengths = [len(response.split()) for response in rollout_response_text]
                mean_response_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0
                
                train_metrics = {
                    'train_step': train_step,
                    'train/loss': rollout_batch_loss,
                    'train/grad_norm': total_grad_norm,
                    'train/reward_mean': all_raw_rewards.mean().item(),
                    'train/reward_std': all_raw_rewards.std().item(),
                    'train/reward_min': all_raw_rewards.min().item(),
                    'train/reward_max': all_raw_rewards.max().item(),
                    'train/format_reward_mean': reward_metadata['format_mean'].item(),
                    'train/format_reward_std': reward_metadata['format_std'].item(),
                    'train/answer_reward_mean': reward_metadata['answer_mean'].item(),
                    'train/answer_reward_std': reward_metadata['answer_std'].item(),
                    'train/advantage_mean': all_advantages.mean().item(),
                    'train/advantage_std': all_advantages.std().item(),
                    'train/mean_response_length': mean_response_length,
                }
                
                if len(all_token_entropy) > 0:
                    train_metrics.update({
                        'train/token_entropy_mean': all_token_entropy.mean().item(),
                        'train/token_entropy_std': all_token_entropy.std().item(),
                    })
                
                if accumulated_clip_fractions:
                    all_clip_fractions = torch.cat(accumulated_clip_fractions)
                    train_metrics['train/clip_fraction'] = all_clip_fractions.mean().item()

                wandb_run.log(train_metrics)
                train_step += 1
    
    wandb_run.finish()

    model_final_dir = os.path.join(model_dir, 'final')
    policy.save_pretrained(save_directory=model_final_dir)
    tokenizer.save_pretrained(save_directory=model_final_dir)
    
    print('Training complete')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch training with config from YAML file')
    parser.add_argument('config_path', type=str, help='Path to YAML config file')
    args = parser.parse_args()

    try:
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file '{args.config_path}' not found", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}", file=sys.stderr)
        sys.exit(1)

    DEBUG = config.get('debug', 0)

    learning_rate = float(config.get('lr', 1e-5))

    # GPU allocation: vLLM on cuda:0 for inference, policy on cuda:1 for training
    policy, tokenizer = init_policy(device='cuda:1', debug=DEBUG)
    params = get_starter_params(policy, learning_rate, debug=DEBUG)
    vllm = init_vllm(
        './Qwen2.5-Math-1.5B',
        'cuda:0',  # vLLM inference on GPU 0
        42,
        params['gpu_memory_utilization'],
        debug=DEBUG
    )
    sampling_params = init_sampling_params(params)
    training_data = get_training_data()
    eval_data = get_eval_data()

    if 'n_grpo_steps' in config:
        params['n_grpo_steps'] = config['n_grpo_steps']

    if 'eval_sample_size' in config:
        params['eval_sample_size'] = config['eval_sample_size']

    if 'use_std_normalization' in config:
        params['use_std_normalization'] = config['use_std_normalization']

    if 'eval_log_frequency' in config:
        params['eval_log_frequency'] = config['eval_log_frequency']

    if DEBUG:
        experiment_name = 'debug_5_grpo_steps'
    else:
        experiment_name = os.path.splitext(os.path.basename(args.config_path))[0]

    timestamp = datetime.datetime.now(datetime.timezone.utc).timestamp()
    timestamp = int(timestamp)
    output_dir = os.path.join('./outputs', '{}_{}'.format(experiment_name, timestamp))
    os.makedirs(output_dir, exist_ok=True)
    
    policy_trained = train_policy(policy, tokenizer, vllm, sampling_params,
                                training_data, params, experiment_name, eval_data,
                                output_dir)