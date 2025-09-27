# A minimal example for GRPO from Scratch

This repository accompanies a Medium blog post: [Hands-On LLM Alignment: Coding GRPO from Scratch, Step by Step](https://medium.com/@baicenxiao/hands-on-llm-alignment-coding-grpo-from-scratch-step-by-step-30c6aa4a2146), explaining how to implement Group Relative Policy Optimization (GRPO) from scratch. The implementation provides a clear, educational example of modern alignment techniques for language models.



If you find any issues with the code or have suggestions for improvements, please feel free to raise a GitHub issue or open a pull request.

## Setup

### 1. Install Dependencies

Install all packages except `flash-attn`, then all packages (`flash-attn` is weird)
```
git clone https://github.com/baicenxiao/hands-on-grpo-from-scratch.git
cd hands-on-grpo-from-scratch
uv sync --no-install-package flash-attn
uv sync
```

### 2. Download GSM8K Dataset

Download GSM8K JSONL files from the official repository:
```bash
# from project root
git clone https://github.com/openai/grade-school-math.git
```

### 3. Download Model Weights

Download Qwen2.5-Math-1.5B weights to a local directory:
```bash
# Ensure you're authenticated first:
# huggingface-cli login

huggingface-cli download Qwen/Qwen2.5-Math-1.5B \
  --local-dir ./Qwen2.5-Math-1.5B \
  --local-dir-use-symlinks False
```

### 4. Run baseline before GRPO
```bash
uv run src/math_baseline.py
```

### 5. Run GRPO finetuning
```bash
# You need to log into your wandb account for logging
uv run src/train_grpo.py sample_config.yaml
```

## References

The environment setup and dependency management follows the approach used in [Stanford CS336 Assignment 5: Alignment](https://github.com/stanford-cs336/assignment5-alignment), providing a robust foundation for machine learning experimentation.

### Papers
- **GRPO Paper**: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) - The original GRPO paper from DeepSeek-AI.

### Libraries
- **Hugging Face TRL**: [Transformer Reinforcement Learning](https://github.com/huggingface/trl) - A library for training transformer language models with reinforcement learning, including RLHF and alignment techniques. We do not directly use TRL, but it is an important reference.