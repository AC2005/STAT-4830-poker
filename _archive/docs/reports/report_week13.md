# High-Level Goals From Last Week
- Move from T4 to A100 training to efficiently train in large amounts
- Implement the ability for models to play against older iterations to add a larger variety of players to the self-play training process
- Implement validation methods to evaluate how our model is improving over time

# Implementation Updates

### Movement from T4 Compatibility to A100 Compatibility
We decided to revise our model training notebook from Unsloth to make our training compatible with A100s. Last week, our model only allowed for training with a T4, which was far too slow (taking about an hour and a half to train on a few hundred examples).

### Self-Play with Older Iterations
We also further enhanced our method of data-generation through self-play. Last week, our pipeline included a single fine-tuning process with our PokerBench dataset, and then another single iteration of data-generation and self-play through a 6-player PyPokerEngine game with 6 of the same fine-tuned models. This week, we changed the self-play process to include multiple different rounds of data-generation. After each round of data-generation, the model trains on the newly-generated examples, and then save the LoRA weights as a new iteration of the model. The next round then chooses 5 models at random from all of our saved models, as well as the latest model weights as the 6th player in the PyPokerEngine game. This introduces diversity in our game and ensures that we do not diverge too far. For example, if one round of data-generation leads to a slightly worse model, we do not continue spiraling into even worse models, since data will still include gameplay from older models.

### Evaluation Methodology
We also implemented a comprehensive evaluation methodology to measure how our model improves over time. This system tracks several key metrics:

Reward Function Tracking: We created a graphing mechanism to visualize how different reward values change throughout training iterations. This allows us to monitor whether our models are receiving increasing rewards for format compliance, reasoning quality, and decision alignment over time.

Stack Size Analysis: We implemented tracking of average chip stack sizes at the end of each game to quantify performance improvements. By comparing the ending chip counts of different model versions, we can numerically assess whether newer iterations outperform older ones.

Model Comparison Framework: We developed a framework for pitting different model versions against each other in controlled environments. This involved comparing the base model (fine-tuned only on PokerBench) against versions that underwent additional self-play training.

### Challenges and Concerns
One challenge is ensuring that our model is actually learning "useful" information and not facing collapse. What commonly happens with models trained on synthetic data (or self-generated data) is that data quality might not be optimal, and lead to weird behaviors or suboptimal training. For example, there were past attempts to create poker bots that involved self-play, but ended up playing in ways that humans never would (and thus the bots were not very useful in real-life games). Since Pluribus is not open-source and there are no readily-available 6-player bots, it is difficult to evaluate whether our model is actually improving in general poker play, or just improving in playing against LLM bots.

# Next Steps
One next step may be to try to find some way to evaluate the bot against human players. This could involve interfacing with some online poker platform and evaluating the bots' win/loss rate and stack sizes throughout games. Another method could be finding some other GTO-optimal 6-player bot (if it exists), and then integrating that into our training to ensure our model learns real poker gameplay.