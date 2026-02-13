# SSL and Representation Learning

## Desirable Properties

**Informativeness**: preserve task-relevant information while discarding noise (Information Bottleneck principle). **Invariance**: unchanged by nuisance transformations (crops, color jitter). **Equivariance**: changes predictably with meaningful transformations. **Disentanglement**: individual dimensions correspond to independent factors.

## Collapse Problem

The fundamental challenge: the model maps all inputs to the same point $f(x) = c$. Different methods prevent collapse differently: SimCLR uses negative examples, MoCo uses a momentum encoder for stable negatives, BYOL uses predictor asymmetry, Barlow Twins decorrelates features, VICReg adds explicit variance/covariance regularization, and DINO uses centering and sharpening.

## Evaluation of Representations

Linear probing (frozen features + linear classifier), fine-tuning (update all parameters), kNN evaluation (no training needed), and cross-dataset transfer.

## SSL vs Supervised Pre-training

SSL representations outperform supervised pre-training when downstream tasks have limited labels, tasks are diverse (SSL features are more general), or the pre-training label set is not representative. Supervised pre-training biases features toward the label taxonomy.
