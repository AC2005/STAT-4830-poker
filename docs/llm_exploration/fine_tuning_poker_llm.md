# Fine-Tuning a Poker LLM - Discussion Log

## Problem Setup

### **Problem Statement**
The goal is to fine-tune a **large language model (LLM)** to consistently make optimal poker decisions. Specifically, we aim to train the model to analyze a given **game state** (such as the board, hand strength, position, and prior actions) and predict the best possible move. The challenge lies in capturing the complex decision-making process that expert poker players use, which involves **probability estimation, opponent modeling, and strategic betting patterns**.

### **Mathematical Formulation**
We define the problem as a **sequence prediction task**, where:
- The **input** is the structured game state and natural language instruction (e.g., "You are on the button with AK offsuit. The action folds to you. What is the best move?").
- The **output** is the optimal poker action (e.g., "Raise 3BB" or "Check").
- The objective is to **maximize accuracy** in predicting the optimal move compared to expert decisions.

Mathematically, we seek to optimize the probability of the correct action **\( a^* \)** given the game state **\( s \)**:

\[
P(a^* | s) = \text{argmax}_{a \in A} P(a | s, \theta)
\]

where:
- \( A \) is the action space (Fold, Call, Raise, Bet sizes, etc.)
- \( \theta \) represents the model parameters being fine-tuned.

Loss Function: We will use **cross-entropy loss** to measure the discrepancy between the predicted and correct actions.

### **Data Requirements**
For fine-tuning, we require a **high-quality dataset** that consists of:
1. **Training Data:** Structured game states and corresponding optimal actions based on expert or solver-generated strategies.
   - Example features:
     - **Player positions** (BTN, SB, BB, etc.)
     - **Hole cards** (e.g., "Ace of Spades, King of Diamonds")
     - **Board state** (Flop, Turn, River)
     - **Bet sizes & pot odds**
     - **Opponent actions**
   - Labels: The **correct poker decision** (Fold, Call, Raise + size).
  
2. **Test Data:** A separate dataset to evaluate model performance on unseen game states.
   - Test cases should include **edge scenarios** (e.g., short stacks, multi-way pots) to measure robustness.
   - **Baseline Comparison:** The model should be benchmarked against human decisions or solver outputs.

### **Success Metrics**
To evaluate model performance, we define the following metrics:

1. **Prediction Accuracy:**  
   - Measures how often the model selects the correct action.
   - Formula:  
     \[
     \text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}
     \]
   - Higher accuracy indicates better decision-making.

2. **Expected Value (EV) Analysis:**  
   - Compare the **expected return** of model-selected actions versus expert moves.

3. **Loss Function Improvement:**  
   - Track the **cross-entropy loss** over training epochs to measure learning progress.

4. **Edge Case Performance:**  
   - Test performance in **difficult scenarios** (e.g., facing all-ins, bluff-catching situations).

5. **Generalization Across Hands:**  
   - Ensure the model is not just memorizing common decisions but can generalize to unseen situations.

---

## **Implementation**
1. **Load the Dataset**  
   - The dataset consists of structured poker game states and optimal decision labels.

2. **Preprocess the Data**  
   - Tokenize and structure data for fine-tuning.

3. **Select a Suitable Model**  
   - **GPT-2 was too small**, so we attempted **Falcon-7B**, but Colab ran out of memory.

4. **Fine-Tune the Model**  
   - Used **causal language modeling** for training.

5. **Evaluate Model Performance**  
   - Compared pre-finetuned and post-finetuned model results.

---

## **Issues and Debugging**
### **Model Repeating the Prompt**
- The model was **echoing input instead of generating decisions**.
- **Fix:** Ensured the model was **trained on outputs only** by separating `instruction` from `output`.

### **Colab Running Out of Memory**
- Falcon-7B required more RAM than available on Colab.
- **Fix:** Switched to **lighter models like Mistral-7B**, but faced gated model access issues.

### **Evaluation Misalignment**
- **Fix:** Updated tokenization to properly separate `instruction` and `output`.

---

## **Next Steps**
1. **Use Larger Models:**  
   - GPT-2 is too small, while **GPT-4 easily understands prompts** and outputs single-word actions.  
   - We need to test larger models like **LLaMA-7B or Falcon-7B**.

2. **Gain Access to More Compute:**  
   - Falcon-7B **exceeded Colab RAM limits**.
   - We need a **high-RAM cloud instance** or **local multi-GPU setup**.

3. **Test Performance on Specialized Reasoning Models:**  
   - Compare against **TinyZero, RAGen**, or other **reasoning-focused architectures**.  
   - Evaluate how these models compare to standard LLMs for poker decision-making.

4. **Dataset Expansion:**  
   - Improve dataset **diversity and edge case coverage**.

5. **Fine-Tune with Reinforcement Learning:**  
   - Move from supervised fine-tuning to **RL-based training** to enhance poker strategy.

---

## **Final Thoughts**
The project involves:
- **Fine-tuning an LLM** to make optimal poker decisions.
- Overcoming **model size and memory constraints**.
- Ensuring **proper dataset structuring** for training.
- Evaluating **baseline models vs fine-tuned models**.

We need to explore **larger models, better compute, and alternative training methods** to improve results. 🚀

