# 29.7.2 Drug Discovery

## Overview
GNNs accelerate drug discovery by predicting drug-target interactions, molecular generation, and ADMET properties from molecular graphs.

## Tasks
- **Virtual screening**: Predict binding affinity between drug molecules and protein targets
- **Drug-drug interaction**: Predict adverse interactions between co-administered drugs
- **Molecular generation**: Generate novel molecules with desired properties (using graph VAEs/GANs)
- **ADMET prediction**: Absorption, distribution, metabolism, excretion, toxicity

## Protein-Ligand Interaction
Proteins and ligands as graphs; predict binding using bipartite GNNs or cross-attention mechanisms.

## Transfer Learning
Pre-train on large molecular databases (ZINC, ChEMBL), fine-tune on specific targets with limited data. Self-supervised pre-training (masked atom prediction, contrastive learning) significantly improves few-shot performance.
