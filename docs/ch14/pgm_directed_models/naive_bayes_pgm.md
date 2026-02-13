# Naive Bayes as a PGM

## Structure

Class $C$ is parent of all features $X_i$; no edges between features. This encodes the "naive" conditional independence assumption.

## Joint Distribution

$P(C, X_1, \ldots, X_n) = P(C) \prod_i P(X_i | C)$

Parameter count: $O(n|\mathcal{X}|)$ with the assumption vs $O(|\mathcal{X}|^n)$ without — a dramatic reduction enabling training with very little data.

## Variants

Gaussian NB ($\mathcal{N}(\mu_{ic}, \sigma_{ic}^2)$ per feature-class pair), Multinomial NB (word counts), Bernoulli NB (binary features).

## Why It Works Despite the Assumption

Classification only requires correct ranking of $P(C|X)$, not accurate probabilities. Estimation error from limited data often exceeds bias from the independence assumption. Robust to irrelevant features.

## Connection to Log-Linear Models

Naive Bayes is equivalent to a log-linear model without feature interactions: $\log P(C=c|X) = b_c + \sum_i w_{ic}\phi_i(X_i)$.
