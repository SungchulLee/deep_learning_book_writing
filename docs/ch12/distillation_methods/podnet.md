# PODNet

## Overview

Pooled Outputs Distillation Network (Douillard et al., 2020) extends knowledge distillation by constraining intermediate features at every layer, not just final logits.

## Pooled Distillation Loss

For each feature map $h_l$, compute width-pooled and height-pooled statistics. The distillation loss constrains normalized pooled features:

$$\mathcal{L}_{POD} = \sum_l \left\| \frac{\hat{h}_l^{old}}{\|\hat{h}_l^{old}\|} - \frac{\hat{h}_l^{new}}{\|\hat{h}_l^{new}\|} \right\|^2$$

## Key Components

Layer-wise constraints prevent representation drift at all levels. Pooling reduces dimensionality for computational efficiency. Cosine classifier (NME) reduces bias toward new classes.

## Results

Consistently outperforms LwF and EWC on CIFAR-100 and ImageNet class-incremental benchmarks, particularly in long task sequences.
