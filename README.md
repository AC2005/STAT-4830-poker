# STAT 4830 Final Project: Poker Zero
Andrew Chang, Wesley Liu, Luke Tong, Megan Zhang, Amy Zheng

## High-Level Summary
This project aims to build a reasoning model for No-Limit Hold'em poker capable of decision-making under incomplete information and adversarial conditions. Traditional poker solvers attempt to converge to game theory optimal (GTO), but it is often infeasible to calculate for multi-way poker games due to their massive deicsion treets. So this project explores whether RL and LLMs can provide a more computationally efficient alternative. This project uses GRPO and the Unsloth fine-tuning framework, which applies LoRA techniques to reduce GPU memory usage, making it feasible to train large models with limited computational resources. We trained on the PokerBench datatset, which consists of 500,000 poker hands. Then, we used PyPokerEngine for self-play, allowing the model to improve by playing against earlier versions of itself. We introduced reward functions that reinforce correct actions and near-optimal decisions. Although greater compute resources may be needed, over the successive rounds of self-play, the model's win rate and reward accumlation improved, indicating that LLMs could be viable poker bots. 

## Repository Structure Overview
```
your-repo/
├── README.md                    # This file
├── report.md                    # Final project report
├── archive/                     # Contains all intermediate work
├── notebooks/                   # Jupyter notebooks
├── src/                         # Source code
└── docs/                        # Final presentation and other final docs
```

## Setup Instructions

## Running the Code

## Executable Demo Link