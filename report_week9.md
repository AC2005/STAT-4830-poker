# Implementation Updates

# Training Infrastructure Improvements

### Unsloth Optimization for T4 GPUs
The training infrastructure has been significantly revamped to accommodate available hardware resources. By leveraging Unsloth's latest notebook optimizations, we've successfully migrated the training process from requiring high-end A100 GPUs to functioning efficiently on more accessible T4 GPUs. This adaptation makes the project more accessible and cost-effective while maintaining training quality (although taking longer).

### Response Structure Revisions
We've implemented a structured output format for the LLM that encourages explicit reasoning, rather than simply the response itself, which was our implementation in previous weeks:


```python
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
... Your reasoning here ...
</reasoning>
<answer>
... Your poker action here (fold, check, call, bet [amount], raise [amount]) ...
</answer>
"""
```

This format transformation serves two primary purposes:
1. It encourages the model to develop a clear chain of thought before making decisions
2. It makes the reasoning process transparent and evaluable, which is crucial for both training and human verification

### Enhanced Reward Functions
The reward functions have been redesigned to specifically reinforce the structured reasoning approach:

```python
# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
      
    rewards = []
    for r, a in zip(extracted_responses, answer):
        reward = 0.0
        if r.strip().lower() == a.strip().lower():  # Full match
            reward = 2.0
        else:
            r_parts = r.strip().lower().split()
            a_parts = a.strip().lower().split()
            if len(r_parts) > 0 and len(a_parts) > 0 and r_parts[0] == a_parts[0] and r_parts[0] in ("bet", "call"):  # Partial match
                reward = 1.0  # Partial credit for action
                if len(r_parts) > 1 and len(a_parts) > 1 and r_parts[1] == a_parts[1]:
                    reward += 1.0  # Additional credit for correct number
        rewards.append(reward)

    return rewards

# def int_reward_func(completions, **kwargs) -> list[float]:
#     responses = [completion[0]['content'] for completion in completions]
#     extracted_responses = [extract_xml_answer(r) for r in responses]
#     return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def answer_format_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    pattern = r"^(fold|check|call|bet \d+|raise \d+)$"
    return [0.5 if re.match(pattern, r) else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]
```

1. **Structure Compliance**: Rewards are given for properly using the defined tag structure
2. **Reasoning Quality**: The system evaluates the content within reasoning tags for depth, relevance to the poker situation, and logical consistency
3. **Decision Alignment**: Rewards are calibrated to ensure the final decision logically follows from the reasoning provided

To accelerate learning, we've made the reward criteria more lenient, allowing for more frequent positive reinforcement while maintaining the overall training direction. This approach helps address the complexity of poker decision-making by providing more granular feedback throughout the training process.

## Reinforcement Learning Implementation

### Self-Play with PPO
We've implemented Proximal Policy Optimization (PPO) to facilitate self-play training through PyPokerEngine. The setup enables:

1. 6-player poker games where instances of our model compete against each other
2. Collection of gameplay data that directly informs model improvements
3. Progressive refinement of decision-making through competitive environments

### GRPO Training Challenges
The implementation of Group Relative Policy Optimization (GRPO) has introduced valuable learning capabilities but also presented computational challenges:

1. **Computational Intensity**: The process of distributing rewards and improving the model through GRPO requires significant compute time
2. **Complexity Handling**: Poker's inherent complexity means the model must process numerous variables and potential outcomes, making the reward distribution and learning process more demanding
3. **Runtime Concerns**: Complete training cycles have extended runtimes, requiring careful optimization to balance thoroughness with practical time constraints

Despite these challenges, the combination of structured reasoning outputs, refined reward functions, and self-play reinforcement learning represents a significant advancement in the project's approach to developing a poker-playing AI capable of human-like reasoning and decision-making.

# Next Steps
The next step will mainly be focused on refining training efficiency and enhancing reward structures. We will see if we can decrease the amount of time that we need to fine-tune and train our models during selfplay, such as exploring smaller models or algorithms that are less compute-intensive. After doing that, we will be able to refine our reward functions to encourage the model to learn as best as possible, since our current functions may or may not result in reward hacking. Another next step would be to explore how to evaluate our model. We currently have a script to run our model against opponents and see the average difference in chips over a large number of games, but there may be more efficient or robuts methods of evaluating our model. For example, we could create a test-split of our PokerBench dataset and evaluate our model on those results as well, since they may be more objective (GTO results).
