# Literature Review

## Fine Tuning in Unsloth

## LoRA

## Introduction
Large-scale fine-tuning of transformer models like GPT-3 (175B) is computationally prohibitive due to the high memory and parameter overhead. LoRA (Low-Rank Adaptation) mitigates this by **freezing** pre-trained model weights and introducing **trainable low-rank decomposition matrices**, significantly reducing the number of parameters to optimize.

## Methodology
### Low-Rank Decomposition
Given a weight matrix $W_0 \in \mathbb{R}^{d \times k}$ in a Transformer layer, LoRA models its adaptation as:
$
W = W_0 + \Delta W, \quad \text{where } \Delta W = BA,
$
where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ with rank $r \ll \min(d, k)$. The LoRA adaptation only updates $A$ and $B$ while keeping $W_0$ frozen, reducing the number of trainable parameters from $O(dk)$ to $O(r(d+k))$.

### Gradient Flow and Optimization
During training, the forward pass is modified as:
$
h = W_0 x + BAx,
$
where $BAx$ represents the low-rank adaptation. The gradients are computed only for $A$ and $B$, reducing the optimization complexity. LoRA also introduces a scaling factor $\alpha/r$ to stabilize training:
$
\Delta W = \frac{\alpha}{r} BA.
$

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
