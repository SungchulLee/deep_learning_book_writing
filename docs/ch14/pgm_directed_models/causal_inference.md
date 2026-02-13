# Causal Inference

## Overview

Causal inference extends the probabilistic framework of Bayesian networks to distinguish between correlation and causation. While standard BNs represent conditional independence structure, causal BNs additionally represent the effects of interventions.

## Observational vs Interventional

- **Observational**: $P(Y \mid X = x)$ — what is $Y$ when we *observe* $X = x$?
- **Interventional**: $P(Y \mid do(X = x))$ — what is $Y$ when we *set* $X = x$?

## do-Calculus

Pearl's do-calculus provides rules for computing interventional distributions from observational data. The key operation is the **truncated factorization**:

$$P(X_1, \ldots, X_n \mid do(X_i = x_i)) = \prod_{j \neq i} P(X_j \mid \text{Pa}(X_j)) \bigg|_{X_i = x_i}$$

The factor $P(X_i \mid \text{Pa}(X_i))$ is removed because the intervention overrides the natural mechanism generating $X_i$.

## Adjustment Formula

If we can identify a set of confounders $Z$ that satisfies the **backdoor criterion**, the causal effect is:

$$P(Y \mid do(X = x)) = \sum_z P(Y \mid X = x, Z = z) P(Z = z)$$

## Counterfactuals

Structural Causal Models (SCMs) formalize counterfactuals:

$$X := f_X(\text{Pa}(X), U_X)$$

where $U_X$ are exogenous noise variables.

## Relevance to Deep Learning

Causal reasoning is increasingly important in ML for robust prediction under distribution shift, fair decision-making (identifying discriminatory causal pathways), and understanding model behavior through causal analysis of learned representations.
