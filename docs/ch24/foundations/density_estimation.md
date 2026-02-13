# Density Estimation Perspective

## Overview

Autoregressive models are fundamentally density estimators: they learn to assign probability values to data points. This perspective connects them to the broader framework of probabilistic modeling and enables applications beyond generation.

## Exact Log-Likelihood

The log-probability of a data point is:

$$\log p(x) = \sum_{i=1}^{d} \log p(x_i \mid x_{<i})$$

Each term is the output of a neural network, making the total log-likelihood a sum of neural network outputs. This is computed in a single forward pass (parallel), unlike sampling which is sequential.

## As Compression

The connection between density estimation and compression is fundamental (Shannon's source coding theorem):

$$\text{Expected code length} \geq H(p^*) = -\mathbb{E}_{p^*}[\log p^*(x)]$$

An autoregressive model trained to minimize negative log-likelihood is simultaneously learning an optimal compressor. Bits per dimension directly measures compression efficiency.

## Anomaly Detection

High-quality density estimation enables anomaly detection: data points with low $\log p(x)$ are anomalous. However, Nalisnick et al. (2019) showed that deep generative models can assign higher likelihood to out-of-distribution data than in-distribution data, so density alone is insufficient for robust anomaly detection.

## Comparison of Density Estimators

| Method | Exact Density | Parametric Assumptions | Scalability |
|--------|--------------|----------------------|-------------|
| Histogram | Yes | Binning | Poor in high-d |
| KDE | Yes | Kernel choice | Poor in high-d |
| GMM | Yes | Gaussian components | Moderate |
| Autoregressive NN | Yes | Network architecture | Excellent |
| Normalizing Flow | Yes | Invertible transforms | Good |
| VAE | Lower bound | Encoder/decoder | Good |

Autoregressive models achieve the best density estimation among neural approaches, at the cost of slow sampling.
