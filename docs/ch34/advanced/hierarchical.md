# 34.5.2 Hierarchical Reinforcement Learning

## Introduction

Hierarchical RL (HRL) decomposes complex tasks into a hierarchy of subtasks, enabling temporal abstraction and transfer learning. By operating at multiple time scales, HRL agents can tackle long-horizon problems that are intractable for flat RL methods.

## Options Framework

The Options framework (Sutton et al., 1999) formalizes temporal abstraction. An **option** $\omega = (I_\omega, \pi_\omega, \beta_\omega)$ consists of an initiation set $I_\omega$, an intra-option policy $\pi_\omega$, and a termination function $\beta_\omega(s) \in [0, 1]$.

A high-level policy selects options; each option executes its internal policy until termination. This creates a semi-Markov decision process (SMDP) at the higher level.

## Feudal Networks and Goal-Conditioned HRL

**Feudal Networks** (Vezhnevets et al., 2017) use a Manager-Worker hierarchy:
- **Manager**: Operates at lower temporal resolution, sets subgoals in a learned latent space
- **Worker**: Executes primitive actions to achieve subgoals

The manager's goal $g_t$ modulates the worker's policy via directional cosine similarity:

$$\pi_\text{worker}(a|s, g) \propto \exp(d_\text{cos}(\nabla_\theta h(s), g))$$

## HIRO: Data-Efficient Hierarchical RL

HIRO (Nachum et al., 2018) uses off-policy corrections to train both levels:
- **High-level**: Sets subgoals for the low-level every $c$ steps
- **Low-level**: Goal-conditioned policy $\pi_\text{low}(a|s, g)$
- **Relabeling**: Off-policy corrections relabel subgoals for sample efficiency

## Key Challenges

1. **Non-stationarity**: The low-level policy changes, making the high-level environment non-stationary
2. **Credit assignment**: Determining which level of the hierarchy is responsible for success/failure
3. **Subgoal representation**: Learning useful subgoal spaces
4. **Exploration**: Efficient exploration across time scales

## Applications in Finance

HRL naturally maps to multi-scale financial decision making:
- **High level**: Strategic asset allocation (monthly/quarterly)
- **Low level**: Tactical execution and timing (daily/intraday)
- **Options**: Different trading strategies activated by market conditions

## Summary

Hierarchical RL addresses long-horizon decision making through temporal abstraction. While powerful in principle, practical HRL remains challenging due to non-stationarity and credit assignment across levels.
