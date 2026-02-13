# Large-Batch Training

## Linear Scaling Rule

When increasing batch size by factor $k$, scale learning rate by $k$: $\text{lr}_{\text{new}} = k \cdot \text{lr}_{\text{base}}$. Preserves effective update magnitude per epoch but breaks down at very large batch sizes.

## Square Root Scaling

More conservative alternative for very large batches: $\text{lr}_{\text{new}} = \sqrt{k} \cdot \text{lr}_{\text{base}}$. Often more stable for batch sizes above 8K.

## LAMB Optimizer

Layer-wise Adaptive Moments for Batch training normalizes gradient updates per layer, enabling extreme batch sizes. LAMB was used to train BERT in 76 minutes on 1024 TPUs.

## Practical Batch Size Guidelines

Fine-tuning: 16-64. Pre-training (small): 256-1024. Pre-training (large): 2K-8K with gradient accumulation. Pre-training (extreme): 32K-64K with LAMB.

## Gradient Accumulation

Most accessible approach: $\text{effective\_batch} = \text{micro\_batch} \times \text{accum\_steps} \times \text{num\_gpus}$. Algorithmically identical to true large batches; wall-clock time increases linearly with accumulation steps.
