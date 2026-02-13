# EM for PGMs

## Overview

The Expectation-Maximization (EM) algorithm is the standard approach for learning parameters of PGMs with latent variables. It iteratively computes expected sufficient statistics (E-step) and maximizes the expected log-likelihood (M-step).

## General Framework

Given observed data $X$, latent variables $Z$, and parameters $\theta$:

**E-step**: Compute the posterior over latent variables:
$$Q(Z) = P(Z \mid X, \theta^{(t)})$$

**M-step**: Maximize the expected complete-data log-likelihood:
$$\theta^{(t+1)} = \arg\max_\theta \mathbb{E}_{Q(Z)}[\log P(X, Z \mid \theta)]$$

## EM for Bayesian Networks

For a BN with latent variables, the M-step decomposes by parameter group:

$$\theta^{(t+1)}_{X_i \mid \text{Pa}(X_i)} = \frac{\mathbb{E}[N(X_i, \text{Pa}(X_i))]}{\mathbb{E}[N(\text{Pa}(X_i))]}$$

where $N(\cdot)$ denotes sufficient statistics (counts for discrete variables, moments for Gaussian).

## EM for Specific Models

### Gaussian Mixture Models
- E-step: compute responsibility $\gamma_{nk} = P(z_n = k \mid x_n, \theta)$
- M-step: update $\mu_k, \Sigma_k, \pi_k$ using weighted statistics

### Hidden Markov Models (Baum-Welch)
- E-step: forward-backward algorithm to compute $\gamma_t(i), \xi_t(i,j)$
- M-step: update transition and emission parameters

### Latent Dirichlet Allocation
- E-step: variational inference for per-document topic distributions
- M-step: update topic-word distributions

## Convergence

EM guarantees monotonic increase of the log-likelihood (or ELBO for variational EM). However, it converges to a local maximum, so multiple random restarts are recommended.

## Variants

- **Hard EM** (Viterbi EM): use $\arg\max$ instead of expectation in E-step
- **Variational EM**: use an approximate posterior when exact inference is intractable
- **Online EM**: update parameters incrementally as data arrives
