# Dynamic Bayesian Networks

## Overview

DBNs extend Bayesian networks for temporal processes: prior network $B_0$ for $t=0$ and transition network $B_{\to}$ defining $P(X_{t+1} | X_t)$.

## Relation to HMMs

An HMM is a specific DBN: one hidden $Z_t$ (depends only on $Z_{t-1}$), one observed $X_t$ (depends only on $Z_t$). DBNs generalize to multiple state variables, direct observation-to-observation edges, and higher-order dependencies.

## Inference

Filtering: $P(X_t | Y_{1:t})$. Smoothing: $P(X_t | Y_{1:T})$. Prediction: $P(X_{t+k} | Y_{1:t})$. For linear-Gaussian DBNs: Kalman filter/smoother. For discrete: forward-backward algorithm.

## Applications

Speech recognition (phoneme states → acoustics), financial regime detection (regime → returns/volatility), gene regulatory networks, robot localization.

## Learning

Known structure: EM algorithm (Baum-Welch for HMMs). Unknown structure: structure learning with temporal constraints (edges only forward or within same time slice).
