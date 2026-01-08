# GRPO Toy Implementation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nataliatarasova/grpo-toy/blob/main/GRPO_toy.ipynb)

Hands-on GRPO (Group Relative Policy Optimization) implementation from scratch in PyTorch. Learn GRPO by training a 135M parameter model on simple arithmetic task using T4 GPU in Colab.

## What's GRPO?
*GRPO was used by DeepSeek-Math to achieve state-of-the-art mathematical reasoning performance.*

GRPO replaces PPO's value function with group statistics. Instead of learning a separate critic, it:
1. Generates multiple responses per prompt (group)
2. Uses group mean/std as baseline for advantage estimation
3. Applies PPO clipping with KL regularization

$$J_{GRPO}(\theta) = \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\{ \min\left[\frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q, o_{i,<t})} \hat{A}_{i,t}, \text{clip}\left(\frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q, o_{i,<t})}, 1-\varepsilon, 1+\varepsilon\right) \hat{A}_{i,t}\right] - \beta D_{KL}(\pi_\theta || \pi_{ref}) \right\}$$


## What's in the Notebook

**Paper-to-Code Bridge**:
- Mathematical GRPO formula → Direct PyTorch implementation
- Simplified on purpose to show clear mapping between theory and code
- Hyperparameter playground: experiment with clip_eps, beta, group_size, temperature
- See how each component (PPO clipping, KL regularization, group advantages) affects training

## Usage

1. Click the Colab badge above
2. Runtime → Change runtime type → T4 GPU
3. Run all cells (takes ~10 minutes)

No external RL libraries - just `transformers` and `torch`
