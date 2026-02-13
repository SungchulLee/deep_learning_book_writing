# LUCIR

## Overview

Learning a Unified Classifier Incrementally via Rebalancing (Hou et al., 2019) addresses the classifier bias problem in class-incremental learning.

## The Bias Problem

Old class weights receive no gradient updates while new classes are trained with abundant data, causing systematic bias toward predicting new classes.

## Key Components

**Cosine normalization**: $y_c = \eta \cdot w_c^T x / (\|w_c\| \|x\|)$ removes magnitude bias. **Less-Forget Constraint**: cosine distance distillation in normalized feature space. **Inter-Class Separation**: margin ranking loss ensuring old and new class features stay well-separated.

$$\mathcal{L} = \mathcal{L}_{CE} + \lambda_1 \mathcal{L}_{LF} + \lambda_2 \mathcal{L}_{MR}$$

Significantly reduces the accuracy gap between old and new classes compared to LwF.
