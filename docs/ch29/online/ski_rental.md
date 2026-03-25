# Ski Rental Problem

Should you rent equipment each time you use it, or pay a large upfront cost to own it? This dilemma — balancing recurring costs against a one-time investment when the duration of need is unknown — is the **ski rental problem**, the canonical example of a **rent-or-buy** online problem. Despite its simplicity, it captures a fundamental tradeoff that appears in cloud computing (renting VMs vs buying servers), software licensing, and even deep learning (recomputing activations vs storing them in memory).

## Problem Formulation

A skier goes skiing for an unknown number of days $n$. Each day, the skier can either:

- **Rent** skis for \$1, or
- **Buy** skis for \$b (and never pay again).

Once the skier buys, all future days are free. The skier does not know $n$ in advance and must decide each day whether to rent or buy. The goal is to minimize total cost relative to the offline optimum.

The offline optimum is:

$$
C_{\text{OPT}}(n) = \min(n, b)
$$

If $n < b$, it is cheaper to rent every day (cost $n$). If $n \geq b$, it is cheaper to buy on day 1 (cost $b$).

## Deterministic Strategy

The natural deterministic strategy is: **rent for exactly $b - 1$ days, then buy on day $b$**.

**Analysis.** Consider two cases:

- **Case 1: $n \leq b - 1$.** The skier rents every day for a total cost of $n$. The offline optimum is also $n$. Competitive ratio: 1.
- **Case 2: $n \geq b$.** The skier rents for $b - 1$ days (cost $b - 1$) and then buys (cost $b$), for a total of $2b - 1$. The offline optimum is $b$. Competitive ratio:

$$
\frac{2b - 1}{b} = 2 - \frac{1}{b} < 2
$$

**Theorem.** The strategy "rent $b - 1$ times, then buy" is strictly 2-competitive. Moreover, no deterministic algorithm can achieve a competitive ratio better than $2 - 1/b$.

*Proof of lower bound.* Consider any deterministic algorithm $A$ that buys on day $d$ (if it ever buys). An adversary sets $n = d$:

- If $A$ buys on day $d$: $C_A = (d - 1) + b$ and $C_{\text{OPT}} = \min(d, b)$.
- If $d \leq b$: $C_A/C_{\text{OPT}} = (d - 1 + b)/d$. This is maximized near $d = b$, giving $(2b - 1)/b$.
- If $A$ never buys: the adversary sets $n$ arbitrarily large, making $C_A/C_{\text{OPT}}$ unbounded.

Therefore no deterministic algorithm achieves ratio better than $2 - 1/b$. $\square$

## Randomized Strategy

Randomization improves the competitive ratio. The skier chooses a random day $D$ on which to buy, drawn from a carefully designed distribution.

**Theorem.** The optimal randomized competitive ratio against an oblivious adversary is:

$$
c^* = \frac{e}{e - 1} \approx 1.5820
$$

### Construction of the Optimal Distribution

Define the probability of buying on day $d$ as:

$$
P(D = d) = \begin{cases} \frac{1}{b} \left(\frac{b-1}{b}\right)^{d-1} & \text{if } 1 \leq d \leq b \\ 0 & \text{if } d > b \end{cases}
$$

This is (approximately) a truncated geometric distribution. The expected cost of this strategy satisfies:

$$
\frac{\mathbb{E}[C_A(n)]}{C_{\text{OPT}}(n)} \leq \frac{e}{e - 1}
$$

for all $n$.

!!! tip "Intuition for $e/(e-1)$"
    The ratio $e/(e-1)$ arises because the optimal distribution balances the probability of buying too early (wasting money if the season is short) against buying too late (overpaying rent). The geometric distribution achieves this balance, and the ratio $e/(e-1)$ emerges in the continuous limit as $b \to \infty$.

## Generalizations

### Multi-Slope Ski Rental

Instead of a binary rent-or-buy decision, the skier faces multiple options with different costs and durations (e.g., daily rental, weekly pass, season pass). The competitive ratio generalizes accordingly.

### TCP Acknowledgment Problem

A server accumulates acknowledgments and must decide when to send them: one at a time (rent) or in a batch with fixed overhead (buy). This is equivalent to ski rental and has the same competitive ratio.

### Bahncard Problem

A commuter decides whether to buy a discount card (fixed cost) that reduces future trip prices. This generalization includes partial savings rather than free future use.

## Connection to Deep Learning

The rent-or-buy tradeoff appears in several deep learning settings:

- **Cloud computing**: training a model requires GPU hours. Renting on-demand instances costs more per hour than reserved instances, but reserved instances require upfront commitment. The ski rental framework helps decide the break-even point.
- **Gradient checkpointing**: storing activations (buy memory) vs recomputing them during backpropagation (rent compute). The tradeoff depends on how many backward passes will use the stored activation.
- **Model compilation**: JIT-compiling a model graph has upfront cost but speeds up future executions. The decision to compile depends on how many times the graph will be executed.

## Summary

The ski rental problem distills the rent-or-buy dilemma into its purest form. The deterministic strategy of renting $b - 1$ times then buying achieves a tight competitive ratio of $2 - 1/b$. Randomization improves this to $e/(e-1) \approx 1.58$, demonstrating the power of randomized online algorithms. The problem's simplicity makes it an ideal starting point for understanding competitive analysis while its generalizations connect to real-world infrastructure decisions.

## References

- [Online Computation and Competitive Analysis (Borodin and El-Yaniv)](https://www.amazon.com/dp/0521619467)
- [Data Streams: Algorithms and Applications (Muthukrishnan)](https://www.cs.rutgers.edu/~muthu/stream-1-1.ps)
