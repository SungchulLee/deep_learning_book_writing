# 32.9.1 Epsilon-Greedy Exploration

## Overview

**ε-greedy** is the simplest and most widely used exploration strategy. With probability $\epsilon$, select a random action; otherwise, select the greedy (best known) action.

## Definition

$$\pi_\epsilon(a|s) = \begin{cases} 1 - \epsilon + \frac{\epsilon}{|\mathcal{A}|} & \text{if } a = \arg\max_{a'} Q(s, a') \\ \frac{\epsilon}{|\mathcal{A}|} & \text{otherwise} \end{cases}$$

Every action has minimum probability $\frac{\epsilon}{|\mathcal{A}|}$, ensuring continued exploration.

## Decaying ε

Constant ε never fully exploits. Common decay schedules:

- **Linear decay**: $\epsilon_t = \max(\epsilon_{\min}, \epsilon_0 - t \cdot \text{decay\_rate})$
- **Exponential decay**: $\epsilon_t = \max(\epsilon_{\min}, \epsilon_0 \cdot \lambda^t)$
- **Inverse**: $\epsilon_t = \frac{1}{1 + c \cdot t}$

For GLIE convergence: $\epsilon_t \to 0$ but $\sum_t \epsilon_t = \infty$.

## Advantages and Limitations

| Advantage | Limitation |
|-----------|-----------|
| Simple to implement | Explores all actions equally (undirected) |
| Guaranteed exploration | Wastes time on clearly bad actions |
| Works with any value method | No adaptation to uncertainty |
| Well-understood theory | Fixed exploration rate (if not decayed) |

## Financial Application

In trading, ε-greedy means the agent usually follows its best-known strategy but occasionally tries random actions (alternative trades). This is analogous to a trader's practice of "paper trading" new ideas while primarily following the main strategy. The key tuning question is how much to explore vs. exploit as market conditions change.
