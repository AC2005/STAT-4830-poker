# High-Level Goals From Last Week

- Revise our self-play implementation to not use PPO, but rather use PyPokerEngine as a means of generating more data for our original GRPO fine-tuning
- Revise our reward functions to encourage more rewards and avoid reward sparsity
- Experiment with different ways to deal with our challenges with compute

# Implementation Updates

### Conversion to Smaller Model
For this week, we first decided to downsize our model from the Llama 8B parameter model to the Qwen 2.5 3B Instruct model. This choice was made because we were limited on compute power, and a smaller model would allow us to iterate faster while still exploring the potential of LLMs to improve their reasoning and exhibit some potential for learning poker before investing more time and compute. We are sticking to using Unsloth's GRPO training notebooks to fine-tune the model.

### Reward Function Revisions
We also made minor adjustments to our reward function in order to combat reward sparsity. Responses now get a larger reward for answers that follow the correct output format based on the following regex: "^(fold|call|raise \d+)$"

We have also adjusted the correctness reward function so that responses that correctly return "raise _" receive partial rewards if the number they return is within around 20% of the true answer. These changes have been made so that more rewards are given and the model learns quicker.

### Self-Play Data Generation with PyPokerEngine
Next, we implemented self-play data generation using PyPokerEngine. Using PyPokerEngine, we simulate 6-player poker games where each player is an instance of our fine-tuned LLM. At the end of each hand, we extract the game state up until the winning player's last move, and convert the PyPokerEngine game state to a prompt similar to those found in our PokerBench dataset. We then keep the winning player's last move as the label for that entry. Doing this required us to make rather drastic changes to the game state as well as our labels in our PokerBench dataset during fine-tuning. For example, in PyPokerEngine, "check" is represented by "call:0", and "bet 20" is represented by "raise:20". Although these swaps are essentially the same, we had to adjust the formatting of our PyPokerEngine output as well as get rid of "check" and "bet" in our PokerBench dataset. Otherwise, the difference in prompts across our samples may make the model confused and inconsistent, as well as make our reward functions useless for certain samples. We then continue training our model with GRPO on the new data that we generate.

Thus, our model pipeline is as follows:
- Fine-tune our base Qwen model on our original PokerBench data
- Generate more data through self-play with our fine-tuned models through PyPokerEngine
- Continue fine-tuning our already-fine-tuned Qwen model on the new data generated through self-play
- Save the LoRA weights to HuggingFace as well as our new dataset

# Results

- Our pipeline is able to successfully generate and convert data from PyPokerEngine using our fine-tuned model. For example, the following is an output from our self-play: 

"""

You are a specialist in playing 6-handed No Limit Texas Holdem. The following will be a game scenario and you need to make the optimal decision.

Here is a game summary:
The small blind is 10 chips and the big blind is 20 chips. Everyone started with 1000 chips.
The player positions involved in this game are UTG, HJ, CO, BTN, SB, BB.
In this hand, your position is BTN, and your holding is ['C8', 'DQ'].
Before the flop, TransformerPlayer4 declared fold:0; TransformerPlayer5 declared raise:30; TransformerPlayer6 declared fold:0; TransformerPlayer1 declared fold:0; TransformerPlayer2 declared raise:40; TransformerPlayer3 declared call:40; TransformerPlayer5 declared raise:50; TransformerPlayer2 declared fold:0; TransformerPlayer3 declared raise:60; TransformerPlayer5 declared raise:70; TransformerPlayer3 declared call:70.
The flop comes CT, H5, S2.
Now it is your turn to make a move.
To remind you, the current pot size is {'main': {'amount': 180}, 'side': []} chips, and your holding is ['C8', 'DQ'].

Decide on an action based on the strength of your hand on this board, your position, and actions before you. Do not explain your answer.
Your optimal action is:
Response: call

"""
- We also notice a clear trend of the model outputs heading towards the desired format of response and answer, which is supported by the regular rewards given by our answer_format_reward_func and correctness_reward_func

### Challenges and Concerns
Again, one challenge is the computational power. With the Qwen 2.5 3B Instruct model, doing 250 steps with GRPO (essentially training on 250 samples), takes around 1.5 hours on a T4 GPU in Colab. To make our struggles worse, Unsloth's optimizations are not compatible with the A100 GPU offered in Colab, which forced us to stick to using a T4 GPU (or else we would sacrifice the optimizations made in the Unsloth notebooks).

Another challenge/concern was the mismatch in conventions between our PokerBench dataset and PyPokerEngine. While we converted all "check" to "call" and all "bet _" to "raise _", this may confuse the model depending on what data it was initially trained on, despite us fine-tuning on this dataset. It is difficult for us to convert our data the other way around (take in the PyPokerEngine data and add "check" and "bet _" to it), since this would require us to make complex logic changes to the PyPokerEngine code to account for the nuances between these swaps. We will likely stick to removing those two moves unless we notice significant disadvantages.



# Next Steps
The next step will be fleshing out some methods of validation and evaluation of how our model is improving. For example, we will implement logic to graph the different rewards given by our different reward functions throughout our training runs so that we can visualize the change, and hopefully see an increasing rate of rewards being given. If not, we will make changes to the reward functions. We will also implement logic to compare the fine-tuned model on just the PokerBench data and the fine-tuned model on the additional data through self-play. One possible method of implementing this is by pitting the two models against each other in a game through PyPokerEngine, and then calculate the average difference of ending chip amounts of each bot over time. It would also be beneficial to try experimenting with using other bots and evaluating them against our own (or using them to train in PyPokerEngine). Many bots are unfortunately not open source or not built for 6-player games, but we have been exploring a possible version of Pluribus.