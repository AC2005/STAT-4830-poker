# Literature Review: Reinforcement Learning and Policy Optimization

## Introduction
Reinforcement Learning (RL) is a machine learning paradigm where an agent interacts with an environment, receiving feedback in the form of rewards to optimize its behavior. This review covers key RL principles, including Markov Decision Processes (MDPs), Q-learning, Deep Q-learning, and policy optimization methods such as Proximal Policy Optimization (PPO) and the recently proposed Guaranteed Reward Policy Optimization (GRPO). Finally, we discuss how these methods can be applied to our poker project, where we aim to train a reasoning model for predicting optimal poker moves.

## Foundations of Reinforcement Learning
### Markov Decision Processes (MDPs)
An MDP is defined by:
- **States (s):** Represent the environment’s possible configurations.
- **Actions (a):** Choices available to the agent in each state.
- **Reward (r):** Feedback signal that evaluates the quality of an action.
- **Transition Function (T):** Defines state transitions given an action.
- **Policy ($pi$)**: A function dictating the agent’s action selection.

An agent learns by following a policy that maps states to actions, with the goal of maximizing cumulative reward.

## TinyZero
TinyZero is a streamlined reinforcement learning (RL) framework inspired by the success of AlphaZero, a groundbreaking algorithm developed by DeepMind. While AlphaZero demonstrated remarkable performance, its computational demands—requiring thousands of TPUs and GPUs—make it impractical for many real-world applications. TinyZero addresses this limitation by optimizing the AlphaZero architecture for smaller-scale problems, making it accessible for researchers and practitioners with limited computational resources.
- TinyZero reduces the complexity of the neural network used in AlphaZero by employing smaller models with fewer layers and parameters.
- TinyZero incorporates optimizations to the MCTS algorithm, such as reduced simulation depth and adaptive exploration strategies, to balance computational efficiency with decision-making accuracy.
- TinyZero emphasizes on-policy learning, where the agent learns directly from its interactions with the environment, reducing the need for extensive self-play iterations and lowering computational costs. In comparison, AlphaZero relies on self-play to generate training data.

### veRL
veRL, or Variational Explicit Reinforcement Learning, is a flexible, efficient and production-ready RL training library for large language models (LLMs). It trains AI by balancing exploration and exploitation. It uses probability to “guess” the best strategies and improve them over time. veRL is useful in the easy extension of diverse RL algorithms by allowing users to build RL dataflows with a few lines of code, seamless integration of existing LLM infra with modular APIs, flexible device mapping and parallelism, and readily integration with popular HuggingFace models. It is fast with state-of-the-art throughput and efficient actor model resharding with 3D-HybridEngine by eliminating memory redundancy and reduces communication overhead. veRL may be useful in poker optimization due to uncertainty handling, better generalization, and scalability since veRL uses probability-based reasoning. TinyZero is built upon veRL.

### Reinforcement Learning in Large Language Models (LLMs)
In LLMs, the state corresponds to the current text sequence, and the action is the next token prediction. A reward function assesses the quality of the generated token, enabling RL-based fine-tuning.

## Q-Learning and Deep Q-Learning
### Q-Learning
Q-learning is an off-policy algorithm that learns the **Q-value function**, which estimates the expected cumulative reward from taking action \(a\) in state \(s\) and following the optimal policy thereafter:

$$Q(s,a) = E[R | s, a]$$

Using the **Bellman equation**, Q-values are iteratively updated:

$$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_a Q(s',a) - Q(s,a)]$$

### Deep Q-Learning
To handle high-dimensional state spaces, **Deep Q Networks (DQN)** approximate the Q-function using a neural network. DQN employs:
1. **Experience Replay:** Stores past transitions to break correlation between sequential samples.
2. **Target Networks:** Uses a periodically updated Q-network to stabilize learning.
3. **Loss Function:** The mean squared error (MSE) between Q-value estimates and target Q-values.

## Policy Optimization Methods
### Policy Gradient Methods
Policy-based methods optimize a policy \($pi_\theta$\) directly by maximizing expected return:

$$J(\theta) = E_{\tau \sim \pi_\theta} [R(\tau)]$$

where ($tau$) is a trajectory. The gradient is estimated using the **policy gradient theorem** and updated via stochastic gradient ascent.

### Proximal Policy Optimization (PPO)
PPO improves policy gradient methods by introducing a **clipped surrogate objective** to prevent large, destabilizing updates:

$$L(\theta) = E\left[ \min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t) \right]$$

where ($r_t(\theta)$) is the probability ratio between new and old policies, and \( $A_t$ \) is the advantage function. PPO balances computational efficiency with stability by avoiding second-order derivatives used in Trust Region Policy Optimization (TRPO).

### Guaranteed Reward Policy Optimization (GRPO)
GRPO introduces a **reward propagation operator** to improve policy optimization:
- **Theoretical Guarantees:** Ensures that each policy update yields a monotonic improvement in expected reward.
- **Reward-Centric Update Rule:** Unlike PPO’s clipping or TRPO’s hard constraints, GRPO explicitly adjusts updates based on the policy's impact on long-term reward.
- **Empirical Performance:** Demonstrates improved stability and efficiency in benchmark reinforcement learning tasks, outperforming PPO and TRPO.

## Progress and Applications on Our Poker Model
Our overarching goal is to develop or fine tune a **reasoning model for poker** using RL to predict optimal poker moves. Thus far:
1. We tried to load in a distilled version of DeepSeek (1.5B Parameters) to finetune on our Poker Bench data, but Colab does not have enough GPU vRAM to support it. If we gain access to more compute, we plan on running that model and evaluating its performance
our data. 
We are currently working on:
1. Using Unsloth to apply GRPO to Llama 3B to develop a reasoning model for our poker dataset. This includes creating desining custom reward functions that reward outputs of proper poker actions, and apply bigger rewards for correctly predicting the 
optimal output. 

## Conclusion
Reinforcement Learning has evolved through Q-learning, deep Q-networks, and policy gradient methods such as PPO and GRPO. These advances provide robust frameworks for training intelligent agents in complex decision-making tasks. Our poker project will leverage these methods to develop a model capable of reasoning and making optimal poker decisions, contributing to AI-driven game strategies.
