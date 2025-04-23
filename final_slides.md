---
marp: true
theme: default
paginate: true
---

# **Problem Definition**

- **What are we solving?**
  - We aim to build a **reasoning model** for No-Limit Hold’em poker capable of decision-making under **incomplete information** and **adversarial conditions**.

- **Why is it important or interesting?**
  - **Poker Complexity**: Real-time strategic thinking, risk assessment, and adaptation.
  - **Real-World Applications**: 
    - Decision-making under uncertainty (finance, security, negotiation).
    - Insights into **behavioral strategies** and **optimal play**.

---
# **Success Criteria**
- **Specific, measurable outcomes**:
  - **Stack size / money won**.
  - **Win rate** (hands won, or profit over time).
  - **Performance against GTO (Game Theory Optimal) strategies** or baseline bots.
  - **Final ranking** in tournament-style simulations.

---
# Literature Review  
_Reinforcement Learning & Policy Optimization_

- Overview of key RL foundations  
- Baseline methods and recent advances  
- Application to our No-Limit Hold’em poker project  
- Emphasis on algorithmic innovations like GRPO and efficient fine-tuning techniques
- Traditional GTO methods and Pluribus

---

# Introduction

- **Reinforcement Learning (RL):**  
  A machine learning paradigm where an agent interacts with an environment to learn optimal behavior through trial-and-error feedback.

- **Core Concepts Covered:**  
  - Markov Decision Processes (MDPs)
  - Q-Learning & Deep Q-Learning (DQN)
  - Policy Optimization Methods (PPO, GRPO)
  - Efficient fine-tuning with LoRA

- **Application Context:**  
  Our poker project leverages these techniques to develop an adaptive reasoning model that maximizes Expected Value (EV) under uncertain and adversarial conditions.

---

# Current Poker Bots and GTO
- For multi-player non-zero sum games, Nash equilbiria is hard to compute
- Pluribus is not developed to adhere strictly to GTO, instead it uses a ML approach that can improve as it decides which actions have better outcomes (Monte Carlo Counterfactual Regret Minimization)
- Neither ChatGPT nor GPT-4 are GTO players, ChatGPT plays conservatively while GPT-4 plays aggressively

# Foundations of Reinforcement Learning

## Markov Decision Processes (MDPs)

- **States (s):** Possible configurations of the environment.
- **Actions (a):** Choices available to the agent.
- **Reward (r):** Immediate feedback assessing the quality of an action.
- **Transition Function (T):** Probability of moving from one state to another.
- **Policy (𝜋):** Strategy used by the agent to select actions.

**Objective:**  
Learn a policy that maximizes the cumulative reward over time.

---

# Q-Learning & Deep Q-Learning (DQN)

## Q-Learning
- **Off-Policy Algorithm:**  
  Estimates the action-value function \( Q(s,a) \) representing the expected cumulative reward.
  
- **Bellman Equation Update:**  

  $$Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\right]$$

- **Key Point:**  
  Uses temporal difference updates to iteratively refine Q-value estimates.

---

## Deep Q-Learning (DQN)
- **Techniques Employed:**
  - **Experience Replay:**  
    Reduces sample correlation by storing and reusing experiences.
  - **Target Networks:**  
    Stabilizes learning by maintaining a separate network to generate target Q-values.
  - **Loss Function:**  
    Minimizes the Mean Squared Error (MSE) between predicted Q-values and the target values.

- **Outcome:**  
  Efficient approximation of Q-values with neural networks for real-world applications.

---

# Policy Optimization Methods

## From PPO to GRPO

- **Policy Gradient Methods:**  
  Directly optimize the policy by maximizing the expected reward.

- **PPO (Proximal Policy Optimization):**  
  Balances exploration and exploitation with clipping or penalty methods to ensure stable updates.
---
- **GRPO (Guaranteed Reward Policy Optimization):**
  - **Monotonic Improvement:**  
    Provides theoretical guarantees for steady policy improvement.
  - **Reward-Centric Updates:**  
    Focuses on adjusting policies based on long-term reward estimates.
  - **Empirical Advantages:**  
    Demonstrates superior performance compared to PPO and TRPO in several benchmarks.

- **Relevance to Poker:**  
  Enables strategic adaptation and robust performance in complex, adversarial settings.

---

# Unsloth & Low Rank Adaptation (LoRA)

## Unsloth Framework
- **Purpose:**  
  An open-source Python framework optimized for fast fine-tuning and deployment of large language models.
- **Key Features:**
  - High-performance PyTorch code with handwritten GPU kernels.
  - Improved memory utilization through typecasting.
  - Scalability: Fine-tuning 8B parameter models on modest GPU setups (e.g., Colab T4).
---
## LoRA (Low Rank Adaptation)
- **Concept:**  
  Introduces low-rank matrices into pretrained model layers to achieve efficient fine-tuning.
- **Benefit:**  
  Significant performance gains with a minimal increase in parameters—ideal for adapting large models in resource-constrained environments.

---

## Mathematical Formulation of LoRA

- **Pretrained Weight Matrix:**  
  Let $W \in \mathbb{R}^{d \times k}$ be a pretrained weight matrix.

- **Low-Rank Update:**  
  Approximate the weight update as:
  
  $$\Delta W = BA$$
  
  where:
  - $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll \min(d,k)$

- **Adapted Weights:**  
  The new weight matrix is given by:
  
  $$W' = W + BA$$

---

# Summary & Application to Poker

- **Literature Insights:**  
  - Reinforcement learning provides a solid foundation for reasoning under uncertainty.
  - Advanced techniques (DQN, GRPO) and fine-tuning methods (LoRA) enable state-of-the-art performance.

- **In Our Poker Project:**  
  - **MDPs and Q-Learning/DQN** provide the groundwork for modeling decision-making processes.
  - **Policy Optimization (GRPO)** helps refine strategies in adversarial play.
  - **Efficient Adaptation (LoRA & Unsloth)** ensures practicality in real-time computational environments.

- **Goal:**
  Leverage these techniques to build a robust, adaptive model that maximizes expected value in competitive poker settings.

# Modeling the Problem

**Mathematical Formulation & Justification**

- **Stochastic Nature of Poker:**
  - Poker is inherently random, with incomplete information and unpredictable outcomes.
  - **Expected Value (EV) Definition:**
    $$EV = \sum_{h \in H} P(h) \times R(h)$$
    where \( P(h) \) is the probability of a hand outcome \( h \) and \( R(h) \) is the corresponding reward.
    
- **Decision-Making Framework:**
  - **States:** Game configurations such as card distribution, betting history, and stack sizes.
  - **Actions:** Possible moves (bet, raise, fold, call).
  - **Objective:** Maximize EV over a sequence of decisions, balancing risk and reward.

- **Modeling Rationale:**
  - Reinforcement Learning (RL) is naturally suited to handle sequential decision-making under uncertainty.
  - Maximizing EV directly addresses the core objective in poker—optimizing long-term profitability.

---

## Optimization & Reward Functions

- **Reward Function Design:**
  - **For Initial Training:**
    - **Negative Reward:** Apply penalties for outputs that violate constraints.
    - **Zero Reward:** No reward for clearly incorrect moves.
  - **Partial Rewards:**
    - Reward for executing a correct action.
    - Additional reward for an almost correct action (e.g., bet size within ±20% of the optimal).
  - **Maximum Reward:**
    - Full reward for both the correct action and optimal bet sizing.
---
- **Why This Approach?**
  - Allows gradual, nuanced learning instead of an all-or-nothing reward.
  - Helps the model learn the subtleties of decision-making in an environment where perfect play is rare.

---

# Algorithm Choice, Tuning, & Implementation

**Selecting GRPO & Refining the Model**

- **Algorithm Choice: Guaranteed Reward Policy Optimization (GRPO)**
  - **Justification:**
    - GRPO provides theoretical guarantees for steady policy improvement.
    - Suitable for poker's continuous and complex environment where isolated wins do not ensure overall success.
    - Aligns with core RL concepts discussed in our course, focusing on cumulative, long-term reward maximization.

- **Tuning Procedure & Hyperparameters:**
  - **Reward Function Tuning:**
    - Began with rewards only for exact matches, but feedback was sparse.
    - Introduced partial credit for near-miss outputs (e.g., valid poker moves, near-optimal bet amounts).
  - **Hyperparameter Exploration:**
    - Systematic grid search over reward thresholds and learning rates.
    - Iterative refinement based on model performance and stability.

- **Implementation Choices:**
  - Leveraged the **Unsloth** framework to optimize GRPO training.
  - Utilized PyTorch for model development and integration, taking advantage of its efficient computation and GPU support.

---

# Limits Encountered & Adaptations

**Practical Constraints & Solutions**

- **Computational Resources:**
  - Initially limited to a T4 GPU on Colab, leading to frequent disconnections and slow iteration.
  - Challenges in accessing scalable GPU resources on platforms like Google Cloud.

- **Impact on Model Training:**
  - Slow training and iteration speeds forced us to adjust our training framework.
  - Required tuning reward functions to provide a denser, more continuous feedback signal.

- **Adaptation Strategies:**
  - **Unsloth** played a critical role in speeding up our training cycles.
  - Optimization of training loops and hyperparameter searches to work within computational constraints.
  - Adoption of incremental learning strategies to mitigate resource limitations while still aiming for optimal performance.

---

# Result

## **Results Overview**
Present your key quantitative results clearly (e.g., graphs, tables).

How do your results compare to baseline methods or the literature?

Provide an interpretation of your results. What do they mean in the context of the problem?

---

# Demo

Showcase a demo or compelling visualization if applicable.

Compare expected progress with actual progress. Explain discrepancies.

---

# **Project Reflection**

## **Technical/Conceptual difficulties**

- Understanding the complexity of poker strategies and how to model them effectively.
- Implementing reinforcement learning algorithms, especially in the context of self-play.

---

## **What part of the project workflow was easier than expected? Harder?**

### **Easier**
- Implementing basic reinforcement learning algorithms using **unsloth** directly within Colab.

### **Harder**
- Debugging and tuning reinforcement learning models to converge effectively in self-play scenarios.

---

## **How the project evolved**

- Trained initial model using **GRPO** and attempted self-play reinforcement learning with **PPO** to generate initial neural layers.
- **Challenge**: Model wasn't converging.
  - Shifted to using the self-play environment to generate additional training data for **GRPO**.
- Focused more on **self-play**, with multiple iterations of PokerZero playing each other to measure performance improvements.

---

## **How did AI tools assist your project?**

### Specific Examples:
- **Literature Review**: Helped in understanding initial concepts and strategies for reinforcement learning.
- **Debugging**: Assisted in understanding complex algorithms and generating code snippets for reinforcement learning tasks.

---

Individual Contributions (Each person must answer):

What was the most surprising result or finding during the project?

Which specific lecture topic, concept, or technique from the course was most useful for your project? Explain how.

How has your perspective on applying optimization in practice changed since the beginning of the course?

If you had two more weeks, what specific next step would you take on the project?

If you could restart the project, what is the single biggest change you would make to your approach or plan?
