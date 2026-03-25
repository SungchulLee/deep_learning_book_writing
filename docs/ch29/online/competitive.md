# Competitive Analysis

When designing algorithms for real-world systems, decisions often must be made without knowledge of future inputs. A web cache decides which pages to keep without knowing future requests; an operating system scheduler assigns tasks without knowing what arrives next. **Competitive analysis** provides a rigorous framework for evaluating such **online algorithms** by comparing their performance against an omniscient optimal offline algorithm that sees the entire input sequence in advance.

## Online vs Offline Algorithms

An **online algorithm** receives input elements one at a time and must make irrevocable decisions after each element, without knowledge of future elements. An **offline algorithm** receives the entire input sequence upfront and can compute the globally optimal solution.

!!! example "Online vs Offline Intuition"
    Consider a ski resort scenario: each day you decide whether to rent skis (\$50) or buy them (\$500). An online algorithm decides day by day. An offline algorithm knows in advance how many days you will ski and chooses the cheapest strategy from the start.

Formally, let $\sigma = \sigma_1, \sigma_2, \ldots, \sigma_n$ be a request sequence. An online algorithm $A$ processes $\sigma_i$ before seeing $\sigma_{i+1}, \ldots, \sigma_n$, producing a cost $C_A(\sigma)$. The optimal offline algorithm $\text{OPT}$ produces cost $C_{\text{OPT}}(\sigma)$.

## The Competitive Ratio

The **competitive ratio** of an online algorithm $A$ is the smallest constant $c$ such that for every request sequence $\sigma$,

$$
C_A(\sigma) \leq c \cdot C_{\text{OPT}}(\sigma) + b
$$

where $b$ is a constant independent of $\sigma$. Algorithm $A$ is called **$c$-competitive**.

When $b = 0$, the algorithm is **strictly $c$-competitive**. The competitive ratio captures the worst-case multiplicative overhead of not knowing the future.

!!! tip "Additive Constant"
    The additive constant $b$ accounts for startup costs or boundary effects. For most analyses, $b$ can be ignored because it becomes negligible for long request sequences.

## Deterministic vs Randomized Competitive Ratios

For **deterministic** online algorithms, the competitive ratio is defined over worst-case input sequences:

$$
c = \sup_{\sigma} \frac{C_A(\sigma)}{C_{\text{OPT}}(\sigma)}
$$

For **randomized** online algorithms, the cost $C_A(\sigma)$ becomes a random variable and we take the expectation:

$$
\mathbb{E}[C_A(\sigma)] \leq c \cdot C_{\text{OPT}}(\sigma) + b
$$

Randomization often yields strictly better competitive ratios than any deterministic algorithm. The adversary model matters:

- **Oblivious adversary**: chooses the request sequence without seeing the algorithm's random choices. This is the standard model for randomized competitive analysis.
- **Adaptive adversary**: observes the algorithm's decisions and adapts future requests accordingly. Against an adaptive adversary, randomization provides no benefit.

## Lower Bounds via Adversary Arguments

To prove that no online algorithm can achieve a competitive ratio better than $c$, one constructs an **adversary strategy** that forces any algorithm to pay at least $c$ times the offline optimum.

??? example "Lower Bound Technique"
    The adversary adaptively constructs a request sequence based on the algorithm's decisions, always choosing the request that maximizes the ratio $C_A(\sigma) / C_{\text{OPT}}(\sigma)$. If the adversary can guarantee this ratio is at least $c$ for every deterministic algorithm, then $c$ is a lower bound on the deterministic competitive ratio.

    For randomized algorithms against an oblivious adversary, **Yao's minimax principle** provides lower bounds: any lower bound on the expected cost of deterministic algorithms against a random input distribution also lower-bounds the competitive ratio of randomized algorithms.

## Amortized Competitive Analysis

Some analyses use a **potential function** $\Phi$ to compare the online algorithm's state to the offline optimum's state. Define the amortized cost of serving request $\sigma_i$ as:

$$
a_i = c_i + \Phi(s_i) - \Phi(s_{i-1})
$$

where $c_i$ is the actual cost and $s_i$ is the algorithm's state after request $i$. If we can show $a_i \leq c \cdot c_i^*$ for the offline cost $c_i^*$ at each step, and $\Phi$ is non-negative with $\Phi(s_0) = 0$, then the algorithm is $c$-competitive.

## Example: Deterministic List Accessing

Consider a linked list of $n$ elements serving access requests. Each access to element $x$ at position $i$ costs $i$. After an access, the algorithm may rearrange the list by moving $x$ toward the front.

The **Move-to-Front (MTF)** strategy moves the accessed element to position 1 after every access. Using amortized analysis with an appropriate potential function:

$$
\text{MTF is 2-competitive}
$$

This means MTF never pays more than twice the cost of the optimal offline list arrangement strategy.

## Connection to Deep Learning

Competitive analysis concepts appear in several deep learning contexts:

- **Online learning** algorithms like online gradient descent are analyzed via regret bounds, the online learning analogue of competitive ratios
- **Caching and prefetching** in GPU memory management use online algorithms whose quality is measured by competitive analysis
- **Adaptive learning rate schedules** make irrevocable decisions about step sizes, trading off exploration and exploitation in a manner analogous to online algorithms

## Summary

Competitive analysis measures how much an online algorithm's lack of future knowledge costs compared to an omniscient offline optimum. The competitive ratio provides worst-case guarantees that hold for any input sequence, making it particularly valuable for systems where adversarial or unpredictable inputs are the norm. Randomization, potential functions, and adversary arguments form the core toolkit for establishing and proving competitive ratios.

## References

- [Online Computation and Competitive Analysis (Borodin and El-Yaniv)](https://www.amazon.com/dp/0521619467)
- [Data Streams: Algorithms and Applications (Muthukrishnan)](https://www.cs.rutgers.edu/~muthu/stream-1-1.ps)
