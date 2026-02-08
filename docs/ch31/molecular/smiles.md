# 31.5.2 SMILES-Based Molecular Generation

## Overview

SMILES (Simplified Molecular-Input Line-Entry System) encodes molecular graphs as strings, reducing molecular generation to **sequence generation**. This enables direct application of powerful language modeling techniques—RNNs, Transformers, and their training paradigms—to chemistry. SMILES-based approaches dominated early molecular generation and remain competitive due to their simplicity and the maturity of sequence modeling infrastructure.

## SMILES Syntax

SMILES encodes molecular structure through a depth-first traversal of the molecular graph:

- **Atoms**: uppercase letters for elements—`C`, `N`, `O`, `F`, `S`; brackets for unusual valence: `[NH3+]`
- **Bonds**: `-` (single, usually implicit), `=` (double), `#` (triple), `:` (aromatic)
- **Branches**: parentheses `(...)` for side chains off the main chain
- **Rings**: matched digit pairs mark ring openings and closings: `C1CCCCC1` = cyclohexane
- **Aromaticity**: lowercase letters denote aromatic atoms: `c1ccccc1` = benzene
- **Stereochemistry**: `/` and `\` for cis/trans; `@`/`@@` for chirality

Examples:

| Molecule | SMILES |
|----------|--------|
| Ethanol | `CCO` |
| Benzene | `c1ccccc1` |
| Aspirin | `CC(=O)Oc1ccccc1C(=O)O` |
| Caffeine | `Cn1c(=O)c2c(ncn2C)n(C)c1=O` |

A critical property of SMILES is that the mapping from molecules to SMILES is **one-to-many**: a single molecule can have multiple valid SMILES representations depending on the traversal order. **Canonical SMILES** (as defined by RDKit or OpenBabel) provide a unique, deterministic representation for each molecule.

## Character-Level SMILES Language Model

The simplest approach treats SMILES as a sequence of characters and trains an autoregressive language model:

$$p(\text{SMILES}) = \prod_{t=1}^{T} p(c_t \mid c_1, \ldots, c_{t-1})$$

where $c_t$ is the character at position $t$. Training maximizes the log-likelihood of the training set SMILES. At generation time, characters are sampled one at a time with temperature control.

**Architecture choices**:
- **LSTM/GRU**: The original approach (Segler et al., 2018). Two-layer LSTM with 512 hidden units is a common baseline.
- **Transformer**: Self-attention captures long-range dependencies better (e.g., matching ring-closure digits far apart in the string). GPT-style causal Transformers are now standard.

**Temperature sampling**: At each step, logits are divided by temperature $\tau$ before softmax:

$$p(c_t = k) = \frac{\exp(z_k / \tau)}{\sum_j \exp(z_j / \tau)}$$

Low $\tau$ (0.5–0.7) produces conservative, high-validity molecules. High $\tau$ (1.0–1.5) increases diversity but reduces validity.

## SELFIES: A Robust Alternative

Self-Referencing Embedded Strings (SELFIES) is a molecular string representation designed so that **every valid SELFIES string maps to a valid molecule**. This eliminates the validity problem entirely at the representation level.

Key differences from SMILES:
- SELFIES uses derivation rules that ensure valency constraints are always satisfied
- Ring closures and branches are handled with explicit state tokens (`[Ring1]`, `[Branch1_1]`)
- Any random string of SELFIES tokens produces a valid molecule

This means a SELFIES language model has **100% validity by construction**, allowing the model to focus entirely on learning the distribution of useful molecules rather than spending capacity on chemical grammar.

The trade-off is that SELFIES strings are typically longer than SMILES (more tokens per molecule) and the distribution may be harder to model because the bijection between molecules and SELFIES is less intuitive than canonical SMILES.

## Transfer Learning and Fine-Tuning

A powerful paradigm for molecular generation combines:

1. **Pre-training**: Train a large SMILES language model on millions of molecules (e.g., ChEMBL, ZINC) to learn general chemical grammar and common structural motifs.

2. **Fine-tuning**: Adapt the pre-trained model to a specific target distribution—e.g., molecules active against a particular protein target, molecules with specific property ranges, or molecules similar to a known drug.

3. **Reinforcement learning**: After pre-training, use policy gradient methods to optimize the generator for a reward function (property score, docking score, synthesizability). The REINFORCE algorithm with a baseline is standard:

$$\nabla_\theta J = \mathbb{E}_{s \sim \pi_\theta} \left[ (R(s) - b) \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(c_t \mid c_{<t}) \right]$$

where $R(s)$ is the reward for generated SMILES $s$ and $b$ is a baseline (e.g., running mean of rewards).

## Conditional Generation

To generate molecules with specific properties, condition the model on desired property values:

$$p(c_t \mid c_{<t}, y)$$

where $y$ is a vector of target properties (logP, QED, binding affinity, etc.). Implementation approaches include:

- **Concatenation**: Embed the property vector and concatenate with the character embedding at each time step.
- **Cross-attention**: Use a Transformer decoder with cross-attention to property embeddings.
- **Classifier-free guidance**: Train with and without conditioning, then interpolate at inference for controllable generation.

## Limitations of SMILES-Based Generation

Despite their success, SMILES-based methods have fundamental limitations:

**Fragility**: Small changes in the SMILES string (e.g., flipping one character) can produce drastically different or invalid molecules. The discrete string space does not reflect continuous chemical similarity.

**Non-uniqueness**: Multiple SMILES encode the same molecule, so the model must learn equivalence classes rather than a clean bijection. Data augmentation with randomized SMILES can partially address this.

**No 3D information**: SMILES encodes 2D topology only. For applications where 3D geometry matters (protein-ligand docking, material properties), graph-based or point-cloud approaches are superior.

**Evaluation gap**: A model may achieve high validity and novelty on SMILES metrics while generating molecules that are trivial variations of training data rather than genuinely novel chemical matter.
