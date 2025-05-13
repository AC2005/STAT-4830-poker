# Self Critique
# Observe
- Notebook is pretty straightforward and just have to run it (and put in huggingface key)
- Report is a little bit lengthy, could break it down by using bullet points to make it more readable
- Code is printing a lot of information, could reduce
# Orient
### Strengths
- Clear explanation of changes made since last week
- Next-steps seem achievable
### Areas for Improvement
- More notions of validation methods should be explored
- Match between PokerBench data and PyPokerEngine converted data should be closer
### Critical Risks/Assumption
- Assuming a 3B parameter model is enough to see some knowledge gains or learning of poker reasoning
- Assuming that labels of just the optimal move is enough to teach the LLM some reasoning
- Assuming that fine-tuned LLM off of PokerBench provides enough foundational learning so that self-play data is not low-quality and only detrimental to our training
# Decide
### Concrete Next Actions
- Write graphing methods for reward functions
- Compare different model versions through PyPokerEngine by observing stack differences over large numbers of games
- Change logic so that wording and match between datasets are closer
# Act
### Resource Needs
- Need to learn how to make the Pluribus installs work and integrate it into our code
- Need to explore more about how synthetic data generation works and tricks with that (since our PyPokerEngine dataset is essentially synthetic data)
- Need to possibly port our training code away from Unsloth in order to take advantage of A100 GPUs