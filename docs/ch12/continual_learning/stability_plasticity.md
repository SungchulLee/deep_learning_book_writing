# Stability-Plasticity Dilemma

## Overview

The fundamental tension in continual learning: a model must be stable enough to retain old knowledge while remaining plastic enough to learn new tasks. These conflict because gradient updates for new tasks may overwrite parameters important for earlier tasks.

## Measuring the Tradeoff

**Average Accuracy**: $A_k = \frac{1}{k}\sum_{i=1}^k a_{k,i}$ after learning task $k$.

**Backward Transfer**: $\text{BWT} = \frac{1}{n-1}\sum_{i=1}^{n-1}(a_{n,i} - a_{i,i})$. Negative BWT indicates forgetting.

**Forward Transfer**: $\text{FWT} = \frac{1}{n-1}\sum_{i=2}^n (a_{i-1,i} - \bar{a}_i)$ measures positive transfer to future tasks.

## Approaches

Regularization (EWC, SI): penalize changes to important parameters — reduced plasticity. Replay: rehearse old examples — memory overhead. Architecture: allocate new capacity — growing model size. Distillation: match old model outputs — extra forward pass. No method perfectly resolves the dilemma.
