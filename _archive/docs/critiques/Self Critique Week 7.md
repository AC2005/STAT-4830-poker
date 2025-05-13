# Self-Critique – Week 7

## OBSERVE
- Good structure of report but a bit lengthly, could break down in bullet points or bold/highlights important information
- Few concrete outputs or examples of how models behaving in practice
- Could have more math in GRPO and PPO sections

---

## ORIENT

### Strengths
- Clear technical explanation of Unsloth, LoRA, and GRPO, with appropriate mathematical and empirical context.
- Logical structure connecting tooling to the broader reinforcement learning pipeline.
- Integration of PyPokerEngine demonstrates a practical plan for model testing and self-play.

### Areas for Improvement
- Missing practical results or training outputs to validate integration efforts and implementation success.
- Reward design and tuning strategies in GRPO are still vague, lacking clarity on specific reward signals used in poker scenarios.
- The connection between LoRA's compression benefits and its actual impact on model behavior is not directly demonstrated.

### Critical Risks / Assumptions
We are assuming that GRPO-based optimization will yield meaningful learning signals in a sparse, stochastic environment like poker without further reward engineering. 
There is also an implicit assumption that Unsloth with LoRA will scale sufficiently for multi-agent training on Colab or limited GPU resources.

---

## DECIDE

### Concrete Next Actions
- Run and include initial self-play results from PyPokerEngine to validate model integration and behavior.
- Formalize reward functions used in GRPO training for poker-specific actions, with examples of partial vs. full credit.

---

## ACT

### Resource Needs
We need access to mid-tier or high-tier GPUs (e.g., A100, V100) for extended fine-tuning and self-play training. Additionally, clearer documentation/examples of GRPO applied to multi-action games would help in shaping reward strategies. Will explore DeepSeek’s GitHub or reach out on forums for clarification on relative reward design in non-text domains.
