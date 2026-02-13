# Fine-Tuning Evaluation

## Overview

Fine-tuning evaluation updates all model parameters on the downstream task, measuring how well the self-supervised model serves as an initialization rather than as a fixed feature extractor.

## Fine-Tuning vs Linear Probing

Linear probing measures frozen feature quality; fine-tuning measures initialization quality. Fine-tuning typically shows a smaller gap to supervised baselines. Risk of overfitting is higher, especially on small datasets.

## Low-Data Regime

SSL shows the largest advantage at 1-10% label fractions. Standard protocol: evaluate at 1%, 10%, and 100% of available labels, comparing against supervised pre-training and random initialization.

## Best Practices

Use discriminative learning rates (lower for pre-trained backbone, higher for new head). Apply warmup for the first few epochs. Use strong augmentation during fine-tuning (unlike linear probing). Report results with confidence intervals across multiple random seeds.
