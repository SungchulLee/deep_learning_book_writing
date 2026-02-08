# 31.5.3 3D Molecule Generation

## Overview

While 2D molecular graphs capture topology (which atoms are connected), many chemical and biological properties depend critically on **3D geometry**—the spatial arrangement of atoms. Protein-ligand binding, catalytic activity, and crystal packing all require accurate 3D molecular representations. 3D molecule generation extends graph generation to jointly produce both the molecular graph and atomic coordinates $\mathbf{R} \in \mathbb{R}^{n \times 3}$.

## 3D Molecular Representation

A 3D molecule is described by:

$$\mathcal{M} = (\mathbf{X}, \mathbf{E}, \mathbf{R})$$

where $\mathbf{X}$ encodes atom types, $\mathbf{E}$ encodes bond types, and $\mathbf{R} = [\mathbf{r}_1, \ldots, \mathbf{r}_n]^\top$ gives Cartesian coordinates. Key physical considerations include:

**Bond lengths**: Each bond type has a characteristic equilibrium length (e.g., C–C single ≈ 1.54 Å, C=C double ≈ 1.34 Å).

**Bond angles**: Atoms adopt specific geometries depending on hybridization (sp³ → tetrahedral ≈ 109.5°, sp² → trigonal planar ≈ 120°).

**Dihedral angles**: Rotation around single bonds creates conformational diversity, critical for drug-receptor complementarity.

**Steric clashes**: Atoms cannot overlap; van der Waals radii impose minimum inter-atomic distances.

## Symmetry Requirements: SE(3) Equivariance

A fundamental requirement for 3D molecular generation is **SE(3) equivariance**: the model's predictions must transform correctly under rotations and translations of the coordinate system. If we rotate all atoms by a matrix $\mathbf{Q}$ and translate by $\mathbf{t}$:

$$\mathbf{r}_i \mapsto \mathbf{Q}\mathbf{r}_i + \mathbf{t}$$

then the model's output (e.g., predicted forces, score functions) must transform consistently. For scalar properties (energy, probability), the model must be **SE(3) invariant**:

$$f(\mathbf{Q}\mathbf{R} + \mathbf{t}) = f(\mathbf{R})$$

For vector fields (forces, score functions for coordinates), the model must be **SE(3) equivariant**:

$$f(\mathbf{Q}\mathbf{R} + \mathbf{t}) = \mathbf{Q} f(\mathbf{R})$$

## Equivariant Neural Networks

Several architectures satisfy SE(3) equivariance by design:

**SchNet**: Uses continuous-filter convolutional layers on inter-atomic distances $d_{ij} = \|\mathbf{r}_i - \mathbf{r}_j\|$. Since distances are invariant to rotation and translation, all operations built on distances inherit invariance. Messages between atoms are:

$$\mathbf{m}_{ij} = \mathbf{h}_j \odot W(d_{ij})$$

where $W$ is a learnable filter parameterized by radial basis functions.

**EGNN (E(n) Equivariant Graph Neural Network)**: Updates both node features $\mathbf{h}_i$ (invariant) and coordinates $\mathbf{r}_i$ (equivariant) at each layer:

$$\mathbf{m}_{ij} = \phi_m(\mathbf{h}_i, \mathbf{h}_j, d_{ij}^2, a_{ij})$$

$$\mathbf{r}_i' = \mathbf{r}_i + C \sum_{j \neq i} (\mathbf{r}_i - \mathbf{r}_j) \cdot \phi_r(\mathbf{m}_{ij})$$

$$\mathbf{h}_i' = \phi_h\left(\mathbf{h}_i, \sum_{j} \mathbf{m}_{ij}\right)$$

The coordinate update uses the direction vectors $(\mathbf{r}_i - \mathbf{r}_j)$, which transform equivariantly under rotation.

**Equivariant Transformers (e.g., TorchMD-NET)**: Combine attention mechanisms with equivariant message passing, using tensor products of irreducible representations to handle higher-order geometric features (not just scalars and vectors but also quadrupole-like terms).

## Equivariant Diffusion Models (EDM)

The state of the art for 3D molecule generation uses **equivariant diffusion models** that denoise both atom types and coordinates simultaneously.

**Forward process**: Add Gaussian noise to coordinates and categorical noise to atom types:

$$q(\mathbf{R}_t \mid \mathbf{R}_0) = \mathcal{N}(\mathbf{R}_t; \sqrt{\bar\alpha_t}\,\mathbf{R}_0,\; (1 - \bar\alpha_t)\,\mathbf{I})$$

$$q(\mathbf{X}_t \mid \mathbf{X}_0) = \text{Cat}(\mathbf{X}_t;\; \bar\alpha_t\,\mathbf{X}_0 + (1 - \bar\alpha_t)\,\mathbf{u})$$

where $\mathbf{u}$ is the uniform distribution over atom types.

**Reverse process**: An equivariant neural network $\epsilon_\theta(\mathbf{X}_t, \mathbf{R}_t, t)$ predicts the noise to remove at each step. The coordinate denoising must be SE(3) equivariant:

$$\epsilon_\theta(\mathbf{Q}\mathbf{R}_t + \mathbf{t}, \mathbf{X}_t, t) = \mathbf{Q}\,\epsilon_\theta(\mathbf{R}_t, \mathbf{X}_t, t)$$

**Center of mass subtraction**: To remove translational degrees of freedom, coordinates are centered at each diffusion step:

$$\mathbf{R}_t \leftarrow \mathbf{R}_t - \frac{1}{n}\sum_{i=1}^{n} \mathbf{r}_{i,t}$$

This projects the diffusion process onto the zero center-of-mass subspace.

## E(3) Equivariant Diffusion Model (EDM)

The EDM architecture (Hoogeboom et al., 2022) is a landmark model for 3D molecule generation:

1. **Joint diffusion** over continuous coordinates $\mathbf{R}$ and categorical atom features $\mathbf{X}$
2. **EGNN backbone** for the denoising network, ensuring equivariance
3. **Learned noise schedule** that adapts the diffusion speed to the data
4. **Atom number prediction**: A separate network predicts the number of atoms $n$, then generation proceeds conditionally on $n$

The training objective combines coordinate denoising loss and atom type cross-entropy:

$$\mathcal{L} = \mathbb{E}_{t, \mathbf{R}_0, \mathbf{X}_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(\mathbf{R}_t, \mathbf{X}_t, t)\|^2 + \lambda\,\text{CE}(\mathbf{X}_0, \hat{\mathbf{X}}_0)\right]$$

## Conditional 3D Generation

For drug design, molecules must be generated to fit a protein binding pocket. Conditional 3D generation conditions on the pocket structure:

$$p(\mathcal{M} \mid \text{pocket}) = p(\mathbf{X}, \mathbf{R} \mid \mathbf{X}_\text{pocket}, \mathbf{R}_\text{pocket})$$

**Pocket conditioning**: The protein pocket atoms are included as context nodes in the graph neural network. During diffusion, only the ligand atoms are noised and denoised while the pocket remains fixed.

**Inpainting**: Generate missing parts of a molecule (e.g., a linker between two fragments) by fixing known atoms and diffusing only the unknown region.

## Evaluation of 3D Molecular Generation

Beyond 2D validity metrics, 3D generation is evaluated on:

**Atom stability**: Fraction of atoms whose local geometry (bond lengths, angles) falls within physically reasonable ranges.

**Molecule stability**: Fraction of generated molecules where all atoms are stable simultaneously.

**Strain energy**: Total energy of the generated conformation relative to the relaxed (energy-minimized) geometry. Lower strain indicates more realistic 3D structures.

**Property distributions**: Compare distributions of molecular properties (energy, HOMO-LUMO gap, dipole moment) between generated and test sets.
