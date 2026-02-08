# 31.5.1 Molecular Graphs

## Overview

Molecular graph generation is the most mature and impactful application of graph generation, with direct applications in **drug discovery**, **materials science**, and **chemical engineering**. A molecule is naturally represented as a graph where atoms are nodes (with element types as features) and chemical bonds are edges (with bond types—single, double, triple, aromatic). The generation task is to produce novel, valid, and useful molecules by learning from databases of known chemical structures.

## Molecular Graph Representation

A molecule $\mathcal{M}$ is represented as an attributed graph $G = (\mathbf{X}, \mathbf{E})$ where:

- **Node features** $\mathbf{X} \in \{1, \ldots, K\}^n$: atom types (C, N, O, S, F, Cl, Br, etc.)
- **Edge features** $\mathbf{E} \in \{0, 1, 2, 3, 4\}^{n \times n}$: bond types (none, single, double, triple, aromatic)
- **Atom properties**: formal charge, chirality, number of hydrogens, aromaticity flag
- **3D coordinates** $\mathbf{R} \in \mathbb{R}^{n \times 3}$ (optional): atomic positions in Euclidean space

The key physical constraint is **chemical valency**: each atom type has a maximum number of bonds it can form. For common organic atoms:

| Atom | Valency | Common in |
|------|---------|-----------|
| C    | 4       | Organic backbone |
| N    | 3       | Amines, heterocycles |
| O    | 2       | Alcohols, ethers |
| S    | 2 (or 6) | Thiols, sulfonates |
| F, Cl, Br | 1   | Halogens |

A valid molecular graph must satisfy:

$$\sum_{j} \text{bond\_order}(i, j) \leq \text{valency}(\text{atom\_type}(i)) \quad \forall\, i$$

## Common Molecular Datasets

| Dataset | Size | Max Atoms | Description |
|---------|------|-----------|-------------|
| QM9 | 134K | 9 heavy | Small organics with quantum properties |
| ZINC250K | 250K | 38 heavy | Drug-like molecules from ZINC |
| MOSES | 1.9M | varies | Molecular benchmarking suite |
| GuacaMol | 1.6M | varies | Generation benchmarks |
| ChEMBL | 2M+ | varies | Bioactivity database |

## Molecular Validity Metrics

Beyond general graph generation metrics (MMD, uniqueness, novelty), molecular generation evaluates domain-specific quality:

**Validity**: Fraction of generated molecules that are chemically valid—correct valency, no impossible substructures, parseable by chemistry toolkits.

**Drug-likeness (QED)**: Quantitative Estimate of Drug-likeness, a composite score capturing molecular weight, lipophilicity (logP), hydrogen bond donors/acceptors, polar surface area, and structural alerts.

**Lipinski's Rule of Five**: A drug candidate should have MW ≤ 500, logP ≤ 5, HBD ≤ 5, HBA ≤ 10. Violations correlate with poor oral bioavailability.

**Synthetic Accessibility (SA)**: Score from 1 (easy) to 10 (hard) estimating how difficult a molecule is to synthesize in a lab. Penalizes exotic ring systems and rare functional groups.

**Fréchet ChemNet Distance (FCD)**: Analogous to FID for images—measures distributional similarity between generated and reference molecules in the activation space of a pretrained molecular neural network.

## Molecular Fingerprints

Fingerprints are fixed-length binary or count vectors encoding molecular substructures, widely used for similarity computation:

**Morgan/ECFP (Extended Connectivity Fingerprints)**: Circular fingerprints that enumerate atom neighborhoods up to a specified radius. ECFP4 (radius 2) is standard for drug discovery.

**MACCS Keys**: 166 predefined structural keys (e.g., "contains a 6-membered ring," "has a nitrogen–oxygen bond").

**Topological fingerprints**: Based on paths of specified lengths through the molecular graph.

The **Tanimoto similarity** between two fingerprints $A$ and $B$ is:

$$T(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

This is the standard metric for molecular similarity and is used in novelty and diversity evaluation.

## Molecular Feature Extraction

```python
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


# ============================================================
# Atom and Bond Feature Definitions
# ============================================================

ATOM_TYPES = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "Si"]
BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]

VALENCY_MAP = {
    "C": 4, "N": 3, "O": 2, "S": 2, "F": 1,
    "P": 5, "Cl": 1, "Br": 1, "I": 1, "Si": 4,
}


def one_hot_encode(value: str, vocabulary: List[str]) -> List[int]:
    """One-hot encode a value given a vocabulary."""
    encoding = [0] * len(vocabulary)
    if value in vocabulary:
        encoding[vocabulary.index(value)] = 1
    return encoding


class MolecularGraph:
    """
    Represents a molecule as a graph with atom/bond features.
    
    Provides methods for constructing molecular graphs from
    atom lists, computing validity, and extracting features
    suitable for graph neural networks.
    """
    
    def __init__(
        self,
        atom_types: List[str],
        bond_list: List[Tuple[int, int, str]],
        formal_charges: Optional[List[int]] = None,
    ):
        self.atom_types = atom_types
        self.bond_list = bond_list  # (i, j, bond_type)
        self.formal_charges = formal_charges or [0] * len(atom_types)
        self.num_atoms = len(atom_types)
    
    def get_node_features(self) -> torch.Tensor:
        """
        Build node feature matrix [num_atoms, num_features].
        
        Features per atom:
          - one-hot atom type (len(ATOM_TYPES))
          - formal charge (1)
          - degree (1)
        """
        degrees = [0] * self.num_atoms
        for i, j, _ in self.bond_list:
            degrees[i] += 1
            degrees[j] += 1
        
        features = []
        for idx in range(self.num_atoms):
            feat = one_hot_encode(self.atom_types[idx], ATOM_TYPES)
            feat.append(self.formal_charges[idx])
            feat.append(degrees[idx])
            features.append(feat)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def get_adjacency_matrix(self) -> torch.Tensor:
        """Build adjacency matrix [num_atoms, num_atoms]."""
        adj = torch.zeros(self.num_atoms, self.num_atoms)
        for i, j, _ in self.bond_list:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
        return adj
    
    def get_bond_type_matrix(self) -> torch.Tensor:
        """
        Build bond-type matrix [num_atoms, num_atoms] where entries
        encode bond order: 0=none, 1=single, 2=double, 3=triple, 4=aromatic.
        """
        bond_order = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 4}
        mat = torch.zeros(self.num_atoms, self.num_atoms, dtype=torch.long)
        for i, j, btype in self.bond_list:
            order = bond_order.get(btype, 1)
            mat[i, j] = order
            mat[j, i] = order
        return mat
    
    def get_edge_index(self) -> torch.Tensor:
        """Edge index in COO format [2, num_edges] for PyG."""
        sources, targets = [], []
        for i, j, _ in self.bond_list:
            sources.extend([i, j])
            targets.extend([j, i])
        return torch.tensor([sources, targets], dtype=torch.long)
    
    def check_valency(self) -> Tuple[bool, List[int]]:
        """
        Check whether all atoms satisfy valency constraints.
        
        Returns:
            (is_valid, list_of_violating_atom_indices)
        """
        bond_order_map = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 1.5}
        bond_sums = [0.0] * self.num_atoms
        
        for i, j, btype in self.bond_list:
            order = bond_order_map.get(btype, 1)
            bond_sums[i] += order
            bond_sums[j] += order
        
        violations = []
        for idx in range(self.num_atoms):
            max_val = VALENCY_MAP.get(self.atom_types[idx], 4)
            if bond_sums[idx] > max_val + 0.01:  # tolerance for aromatic
                violations.append(idx)
        
        return len(violations) == 0, violations


def compute_molecular_properties(smiles: str) -> Optional[Dict[str, float]]:
    """
    Compute standard molecular properties from a SMILES string.
    
    Uses RDKit if available; returns None if the molecule is invalid
    or RDKit is not installed.
    
    Properties returned:
        mw         – molecular weight
        logp       – octanol-water partition coefficient
        hbd        – hydrogen bond donors
        hba        – hydrogen bond acceptors
        tpsa       – topological polar surface area
        qed        – quantitative estimate of drug-likeness
        num_rings  – number of rings
        rotatable  – number of rotatable bonds
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, QED
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        return {
            "mw": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "hbd": float(Descriptors.NumHDonors(mol)),
            "hba": float(Descriptors.NumHAcceptors(mol)),
            "tpsa": Descriptors.TPSA(mol),
            "qed": QED.qed(mol),
            "num_rings": float(Descriptors.RingCount(mol)),
            "rotatable": float(Descriptors.NumRotatableBonds(mol)),
        }
    except ImportError:
        return None


def compute_fingerprint_similarity(
    smiles_a: str, smiles_b: str, radius: int = 2, n_bits: int = 2048,
) -> Optional[float]:
    """
    Compute Tanimoto similarity between two molecules using
    Morgan (ECFP) fingerprints.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
        
        mol_a = Chem.MolFromSmiles(smiles_a)
        mol_b = Chem.MolFromSmiles(smiles_b)
        if mol_a is None or mol_b is None:
            return None
        
        fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, radius, nBits=n_bits)
        fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, radius, nBits=n_bits)
        
        return DataStructs.TanimotoSimilarity(fp_a, fp_b)
    except ImportError:
        return None


# ============================================================
# Evaluation Metrics for Molecular Generation
# ============================================================

class MolecularMetrics:
    """
    Evaluates a set of generated SMILES strings on standard
    molecular generation benchmarks.
    """
    
    def __init__(self, reference_smiles: Optional[List[str]] = None):
        self.reference_smiles = set(reference_smiles) if reference_smiles else set()
    
    def validity(self, generated: List[str]) -> float:
        """Fraction of generated SMILES that parse to valid molecules."""
        try:
            from rdkit import Chem
        except ImportError:
            return float("nan")
        
        valid = sum(1 for s in generated if Chem.MolFromSmiles(s) is not None)
        return valid / len(generated) if generated else 0.0
    
    def uniqueness(self, generated: List[str]) -> float:
        """Fraction of unique canonical SMILES among valid molecules."""
        try:
            from rdkit import Chem
        except ImportError:
            return float("nan")
        
        canonical = set()
        valid_count = 0
        for s in generated:
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                valid_count += 1
                canonical.add(Chem.MolToSmiles(mol))
        
        return len(canonical) / valid_count if valid_count > 0 else 0.0
    
    def novelty(self, generated: List[str]) -> float:
        """Fraction of valid, unique molecules not in the reference set."""
        if not self.reference_smiles:
            return float("nan")
        
        try:
            from rdkit import Chem
        except ImportError:
            return float("nan")
        
        canonical = set()
        for s in generated:
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                canonical.add(Chem.MolToSmiles(mol))
        
        novel = canonical - self.reference_smiles
        return len(novel) / len(canonical) if canonical else 0.0
    
    def evaluate(self, generated: List[str]) -> Dict[str, float]:
        """Run all metrics and return a summary dict."""
        return {
            "validity": self.validity(generated),
            "uniqueness": self.uniqueness(generated),
            "novelty": self.novelty(generated),
            "count": len(generated),
        }


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    # Build a simple molecular graph: ethanol (C-C-O)
    mol = MolecularGraph(
        atom_types=["C", "C", "O"],
        bond_list=[(0, 1, "SINGLE"), (1, 2, "SINGLE")],
    )
    
    print("=== Molecular Graph: Ethanol ===")
    print(f"Node features shape: {mol.get_node_features().shape}")
    print(f"Adjacency matrix:\n{mol.get_adjacency_matrix()}")
    print(f"Bond type matrix:\n{mol.get_bond_type_matrix()}")
    print(f"Edge index:\n{mol.get_edge_index()}")
    
    is_valid, violations = mol.check_valency()
    print(f"Valency valid: {is_valid}, violations: {violations}")
    
    # Demonstrate metrics (requires RDKit)
    test_smiles = ["CCO", "c1ccccc1", "CC(=O)O", "invalid_smiles", "CC(C)CC"]
    metrics = MolecularMetrics(reference_smiles=["CCO", "c1ccccc1"])
    results = metrics.evaluate(test_smiles)
    print(f"\nGeneration metrics: {results}")
    
    # Property computation
    props = compute_molecular_properties("CC(=O)Oc1ccccc1C(=O)O")
    if props:
        print(f"\nAspirin properties: {props}")
    
    # Fingerprint similarity
    sim = compute_fingerprint_similarity("CCO", "CCCO")
    if sim is not None:
        print(f"\nEthanol vs Propanol similarity: {sim:.4f}")
```

## Molecular Validity in Generative Models

Enforcing chemical validity during generation is a central challenge. Three main strategies exist:

**Hard constraints (autoregressive)**: At each generation step, mask out invalid actions—atoms that would exceed valency, bonds to non-existent atoms. This guarantees 100% validity but restricts the model's exploration of chemical space.

**Soft constraints (one-shot / diffusion)**: Add a validity penalty $\lambda \cdot \mathcal{L}_{\text{val}}$ to the training loss. This encourages but does not guarantee validity, requiring post-hoc filtering of invalid outputs.

**Post-hoc correction**: Generate molecules freely, then apply chemical correction rules—adjust bond orders to satisfy valency, add or remove implicit hydrogens, kekulize aromatic systems. Tools like RDKit's `SanitizeMol` perform this automatically.

The choice among these approaches reflects a fundamental trade-off between **expressiveness** (the model's ability to explore diverse molecular space) and **guaranteed correctness** (ensuring every output is a real molecule). In practice, many state-of-the-art systems combine soft constraints during training with post-hoc correction during inference.
