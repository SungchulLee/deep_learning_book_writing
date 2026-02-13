# Evaluation Protocols

## Task-Incremental

Evaluate per task with task ID provided. $A_{final} = \frac{1}{T}\sum_t a_{T,t}$.

## Class-Incremental

Unified test set with examples from all seen classes, no task ID. Significantly harder — better reflects real-world deployment.

## Metrics

**Average incremental accuracy**: $\bar{A} = \frac{1}{T}\sum_{k=1}^T A_k$. **Forgetting**: $F = \frac{1}{T-1}\sum_{i=1}^{T-1} \max_k(a_{k,i} - a_{T,i})$. **Intransigence**: $I_k = a_k^* - a_{k,k}$ (penalty from sequential training).

## Controlled Comparisons

For fair comparison, control for: memory budget (replay buffer size, parameter count), backbone architecture, compute (FLOPs and training time), epochs per task, and exemplar selection strategy (random vs herding vs class-balanced).
