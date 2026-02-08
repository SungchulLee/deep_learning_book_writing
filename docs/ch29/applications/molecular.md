# 29.7.1 Molecular Property Prediction

## Overview
Molecules as graphs: atoms=nodes, bonds=edges. GNNs predict properties like solubility, toxicity, and binding affinity.

## Node Features
Atom type (one-hot), degree, charge, hybridization, aromaticity, hydrogen count.

## Edge Features
Bond type (single/double/triple/aromatic), conjugation, ring membership, stereochemistry.

## Architectures
MPNN, SchNet (3D-aware), GIN, Graphormer. Pre-training with masked attributes or contrastive learning boosts performance.

## Benchmarks
QM9 (quantum properties), ZINC (solubility), MoleculeNet (BBBP, HIV, Tox21), OGB-Mol.
