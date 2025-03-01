# Literature Review

## Fine Tuning in Unsloth

Unsloth is an open-source Python framework that speeds up the process of fine-tuning and accessing large language models (LLMs). It does so through the following methods:

1. Optimized Computation Kernels

Standard Approach: In typical fine-tuning frameworks, backpropagation and forward passes are handled by general-purpose libraries (like PyTorch) that are optimized for flexibility rather than maximum speed.
Unsloth's Strategy: Unsloth rewrites key portions of the modeling code into specialized, highly optimized kernels using Triton. By manually deriving the backpropagation steps and implementing them in these low-level kernels, it cuts out much of the overhead that general-purpose implementations carry.
Goal-Oriented Outcome: This results in faster computation during training, as the custom kernels are tailored specifically to the operations performed in language models.

2. Memory Efficiency and Reduced Overhead

Standard Approach: Many systems duplicate memory usage (for example, storing additional copies for gradient calculations or keeping separate caches for inference and training), which slows down processing and increases VRAM usage.
Unsloth's Strategy: Unsloth smartly removes this duplication by sharing memory spaces between the inference engine (vLLM) and the training process. It also employs efficient gradient checkpointing—offloading intermediate activations to system RAM asynchronously. This not only slashes the required VRAM but also minimizes latency by reducing data transfers.
Goal-Oriented Outcome: With a leaner memory footprint, the model can process larger batches or longer sequences without the overhead that slows down traditional approaches.

3. Integrated Fast Inference

Standard Approach: Typically, the training and inference pipelines are distinct, which can lead to inefficiencies when switching contexts or reloading models.
Unsloth's Strategy: By integrating fast inference directly into the training pipeline (using tools like vLLM), Unsloth allows for simultaneous fine-tuning and efficient generation. This avoids the additional overhead of moving data between separate processes or systems.
Goal-Oriented Outcome: The result is a smoother, faster training process where the benefits of optimizations carry through both training and inference stages.

## GRPO

GRPO (Group Relative Policy Optimization) was developed by DeepSeek. Instead of training a model solely on next-token prediction (which simply teaches it to mimic data), GRPO trains a model to optimize a reward function. This reward function is designed to capture qualities like correctness, logical reasoning, and even stylistic or structural attributes.

Group-Based Comparison:
The “group relative” aspect comes from how GRPO evaluates multiple responses at once. For each input, the model generates several candidate responses (for example, 8 variations). Rather than judging each answer in isolation, GRPO compares them within the group:

Scoring: Each response is scored based on a reward function (or set of functions). For instance, in a simple arithmetic task, the reward function might add points for a correct answer and deduct points for errors.

Relative Reinforcement: The model then reinforces responses that score above the group average. In effect, it’s not only learning what a “good” answer looks like in absolute terms, but what distinguishes a better answer from an average one.

Why It Matters:
This process helps the model learn not just the final answer, but the underlying chain-of-thought or reasoning process leading to that answer. By emphasizing the quality of the reasoning, GRPO pushes the model to “think” more deeply about its outputs.

## LoRA

## Introduction
Large-scale fine-tuning of transformer models like GPT-3 (175B) is computationally prohibitive due to the high memory and parameter overhead. LoRA (Low-Rank Adaptation) mitigates this by **freezing** pre-trained model weights and introducing **trainable low-rank decomposition matrices**, significantly reducing the number of parameters to optimize.

## Methodology
### Low-Rank Decomposition
Given a weight matrix $W_0 \in \mathbb{R}^{d \times k}$ in a Transformer layer, LoRA models its adaptation as:
$W = W_0 + \Delta W, \quad \text{where } \Delta W = BA$,
where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ with rank $r \ll \min(d, k)$. The LoRA adaptation only updates $A$ and $B$ while keeping $W_0$ frozen, reducing the number of trainable parameters from $O(dk)$ to $O(r(d+k))$.

### Gradient Flow and Optimization
During training, the forward pass is modified as:
$h = W_0 x + BAx$,
where $BAx$ represents the low-rank adaptation. The gradients are computed only for $A$ and $B$, reducing the optimization complexity. LoRA also introduces a scaling factor $\alpha/r$ to stabilize training:
$\Delta W = \frac{\alpha}{r} BA$.

## Notable Empirical Results
LoRA achieves comparable or superior performance to full fine-tuning across various benchmarks:
- **GLUE Benchmark (RoBERTa-large, DeBERTa-XXL)**: LoRA achieves **89.0** (RoBERTa) and **91.3** (DeBERTa) avg. accuracy, matching full fine-tuning despite having $10^3$ fewer trainable parameters.
- **GPT-3 175B (MNLI, WikiSQL, SAMSum)**: LoRA achieves **91.7% (MNLI)**, **74.0% (WikiSQL)**, and **53.8 ROUGE-1 (SAMSum)** with $10,000\times$ fewer parameters.
- **Reduced Memory Overhead**: LoRA reduces GPU memory usage by **3×** and model checkpoint size by **10,000×**.

## Theoretical Justification
The effectiveness of LoRA is rooted in the **intrinsic rank-deficiency** of weight updates in fine-tuning (Aghajanyan et al., 2020). Empirical evidence shows that $\Delta W$ exhibits low-rank properties:
- The top singular vectors of $\Delta W$ overlap significantly with different runs, indicating a stable low-rank adaptation space.
- Projection of $W_0$ onto the learned $\Delta W$ space shows that LoRA **amplifies underutilized feature directions** rather than replicating existing model behaviors.

## Conclusion
LoRA enables efficient fine-tuning by leveraging **low-rank reparametrization**, achieving strong performance with minimal computational overhead. It is a **scalable, latency-free alternative to full fine-tuning**, making large-scale deployment feasible.



# Testing
## PyPokerEngine
### Brief Introduction
PyPokerEngine allows us to create player instances that exist in a game environment. They can receive game data and output moves, and our model (after some input parsing/cleaning) will read in the info and output a valid move. It provides a flexible and customizable framework for developing and testing poker-playing AI agents. The engine is particularly useful for reinforcement learning (RL) research, game theory applications, and AI-driven decision-making in imperfect information games. Poker is a complex game characterized by incomplete information, stochasticity, and strategic decision-making. Unlike games such as chess or Go, where all information is available to players, poker requires reasoning under uncertainty and managing risk. Poker AI research gained prominence with the success of DeepStack and Libratus, which leveraged counterfactual regret minimization (CFR) and deep reinforcement learning to defeat human professionals. PyPokerEngine offers a simplified environment to experiment with such strategies without needing extensive game infrastructure.

Example:

```python
from pypokerengine.api.game import setup_config, start_poker

config = setup_config(max_round=200, initial_stack=200, small_blind_amount=5)
config.register_player(name="p1", algorithm=PokerZero())
config.register_player(name="p2", algorithm=PokerZero())
config.register_player(name="p3", algorithm=PokerZero())
game_result = start_poker(config, verbose=1)
```

We can then save game logs to observe performance or use it as training data to feed back into our model.

### Reinforcement Learning

PyPokerEngine provides a powerful Emulator class specifically designed for reinforcement learning applications. Here's how we'll utilize it in our implementation:

1. **Game State Management**

    - The Emulator class allows us to simulate poker games and track game states
    - We can restore and manipulate game states, enabling exploration of different scenarios
    - Perfect for training our model through experience replay and state exploration

2. **Player Implementation**

    ```python
    class PokerZero(BasePokerPlayer):
        def receive_game_start_message(self, game_info):
            # Initialize emulator with game parameters
            self.emulator = Emulator()
            self.emulator.set_game_rule(
                player_num=game_info["player_num"],
                max_round=game_info["rule"]["max_round"],
                small_blind_amount=game_info["rule"]["small_blind_amount"],
                ante_amount=game_info["rule"]["ante"]
            )

            # Register player models for simulation
            for player in game_info["seats"]["players"]:
                self.emulator.register_player(player["uuid"], UnslothModel())

        def declare_action(self, valid_actions, hole_card, round_state):
            # Convert game state for model input
            game_state = restore_game_state(round_state)

            # Use emulator to simulate possible outcomes
            # Returns will be used for model training and action selection
            future_states = []
            for action in valid_actions:
                next_state, events = self.emulator.apply_action(game_state, action)
                future_states.append((action, next_state))

            # Model selects best action based on simulated outcomes
            return self.model.select_action(future_states)
    ```

3. **Training Process**

    - Use self-play with multiple instances of our UnslothPokerPlayer
    - Collect game experiences through emulator simulations
    - Update model weights based on game outcomes and reward signals
    - Leverage the emulator's ability to run complete game simulations for better policy learning

4. **Advantages of This Approach**
    - Efficient state simulation and exploration
    - Built-in game rule enforcement
    - Support for multiple players in training scenarios
    - Perfect for implementing self-play training methodology

This integration will allow us to create a robust training environment for our poker AI, combining PyPokerEngine's poker-specific functionality with our Unsloth-based reinforcement learning model.

