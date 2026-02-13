# Pretext Task Taxonomy

## Predictive Methods

The model predicts part of the input from the rest. Examples: masked token prediction (BERT), next sentence prediction, image rotation prediction (RotNet), jigsaw puzzle solving, and colorization.

## Contrastive Methods

The model distinguishes similar (positive) from dissimilar (negative) pairs:

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(z_i, z_j^+) / \tau)}{\sum_k \exp(\text{sim}(z_i, z_k) / \tau)}$$

Key design choices: augmentation strategy for positives, negative source (in-batch, memory bank, momentum encoder), and temperature $\tau$. Examples: SimCLR, MoCo, BYOL (no explicit negatives), Barlow Twins.

## Generative Methods

The model reconstructs input from a corrupted version. Examples: denoising autoencoders, MAE (mask 75% of patches, reconstruct pixels), BEiT (mask patches, predict visual tokens), Data2Vec (mask input, predict teacher features).

## Selection Guidelines

NLP: masked language modeling dominates. Vision: contrastive (SimCLR, DINO) or masked image modeling (MAE). Time series: contrastive with temporal augmentations. Multi-modal: contrastive alignment (CLIP). The pretext task should encourage features relevant to downstream tasks without introducing shortcuts.
