from vllm import LLM, SamplingParams
import json
from collections.abc import Callable
from typing import List

from src.drgrpo_grader import r1_zero_reward_fn


with open('src/prompts/r1_zero.prompt', 'r') as f:
    R1_ZERO_PROMPT = f.read()

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    answers,
    eval_sampling_params: SamplingParams
    ) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    results = []

    for output, answer in zip(outputs, answers):
        prompt = output.prompt
        generated_text = output.outputs[0].text

        reward = r1_zero_reward_fn(generated_text, answer)

        results.append({
            'prompt': prompt,
            'response': generated_text,
            'correct_answer': answer,
            'reward': reward,
        })
    
    return results

def eval_qwen_gsm8k():
    # 1. Load GSM8K
    with open('grade-school-math/grade_school_math/data/test.jsonl', 'r') as f:
        prompt_data = [json.loads(line) for line in f]

    prompts = []
    answers = []

    # 2. Use the GSM8K keys
    for item in prompt_data:
        # R1_ZERO_PROMPT should still have a `{question}` placeholder
        prompts.append(R1_ZERO_PROMPT.format(question=item['question']))
        answers.append(item['answer'])

    # 3. Sampling params
    eval_sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024
    )
    eval_sampling_params.stop = ["</answer>"]
    eval_sampling_params.include_stop_str_in_output = True

    llm = LLM(model='./Qwen2.5-Math-1.5B')

    # reuse your generic evaluator
    results = evaluate_vllm(llm, r1_zero_reward_fn, prompts, answers, eval_sampling_params)

    # 4. Write out to a new file so you don't overwrite your MATH results
    with open('./outputs/qwen_gsm8k_perf.json', 'w') as f:
        json.dump(results, f, indent=4)

    return results

if __name__ == '__main__':
    eval_qwen_gsm8k()
