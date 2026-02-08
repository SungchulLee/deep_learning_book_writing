"""
31.5.1 Molecular Graphs — Implementation

Molecular graph representation, feature extraction, valency checking,
property computation, fingerprint similarity, and generation metrics.
"""

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


# ============================================================
# Molecular Graph Data Structure
# ============================================================

class MolecularGraph:
    """
    Represents a molecule as a graph with atom/bond features.

    Provides methods for constructing molecular graphs from
    atom lists, computing validity, and extracting features
    suitable for graph neural networks.

    Args:
        atom_types: List of atom element symbols.
        bond_list: List of (i, j, bond_type) tuples.
        formal_charges: Optional list of formal charges per atom.
    """

    def __init__(
        self,
        atom_types: List[str],
        bond_list: List[Tuple[int, int, str]],
        formal_charges: Optional[List[int]] = None,
    ):
        self.atom_types = atom_types
        self.bond_list = bond_list
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
        """Build binary adjacency matrix [num_atoms, num_atoms]."""
        adj = torch.zeros(self.num_atoms, self.num_atoms)
        for i, j, _ in self.bond_list:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
        return adj

    def get_bond_type_matrix(self) -> torch.Tensor:
        """
        Build bond-type matrix [num_atoms, num_atoms].
        Entries: 0=none, 1=single, 2=double, 3=triple, 4=aromatic.
        """
        bond_order = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 4}
        mat = torch.zeros(self.num_atoms, self.num_atoms, dtype=torch.long)
        for i, j, btype in self.bond_list:
            order = bond_order.get(btype, 1)
            mat[i, j] = order
            mat[j, i] = order
        return mat

    def get_edge_index(self) -> torch.Tensor:
        """Edge index in COO format [2, num_edges] for PyTorch Geometric."""
        sources, targets = [], []
        for i, j, _ in self.bond_list:
            sources.extend([i, j])
            targets.extend([j, i])
        return torch.tensor([sources, targets], dtype=torch.long)

    def check_valency(self) -> Tuple[bool, List[int]]:
        """
        Check whether all atoms satisfy valency constraints.

        Returns:
            Tuple of (is_valid, list_of_violating_atom_indices).
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
            if bond_sums[idx] > max_val + 0.01:
                violations.append(idx)

        return len(violations) == 0, violations


# ============================================================
# Property Computation (RDKit-based)
# ============================================================

def compute_molecular_properties(smiles: str) -> Optional[Dict[str, float]]:
    """
    Compute standard molecular descriptors from a SMILES string.

    Requires RDKit. Returns None if the molecule is invalid or
    RDKit is not installed.

    Properties:
        mw, logp, hbd, hba, tpsa, qed, num_rings, rotatable
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
    smiles_a: str,
    smiles_b: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> Optional[float]:
    """
    Compute Tanimoto similarity using Morgan (ECFP) fingerprints.

    Args:
        smiles_a: First molecule SMILES.
        smiles_b: Second molecule SMILES.
        radius: Morgan fingerprint radius (2 = ECFP4).
        n_bits: Fingerprint bit vector length.

    Returns:
        Tanimoto similarity in [0, 1], or None if invalid.
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
# Generation Evaluation Metrics
# ============================================================

class MolecularMetrics:
    """
    Evaluates generated SMILES on standard molecular benchmarks:
    validity, uniqueness, and novelty.

    Args:
        reference_smiles: Optional set of training SMILES for novelty.
    """

    def __init__(self, reference_smiles: Optional[List[str]] = None):
        self.reference_smiles = set(reference_smiles) if reference_smiles else set()

    def validity(self, generated: List[str]) -> float:
        """Fraction of SMILES that parse to valid molecules."""
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
        """Fraction of valid unique molecules not in the reference set."""
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
        """Run all metrics and return a summary dictionary."""
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
    # ---- Build a simple molecular graph: ethanol (C-C-O) ----
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

    # ---- Invalid molecule: carbon with 5 single bonds ----
    bad_mol = MolecularGraph(
        atom_types=["C", "H", "H", "H", "H", "H"],
        bond_list=[
            (0, 1, "SINGLE"), (0, 2, "SINGLE"), (0, 3, "SINGLE"),
            (0, 4, "SINGLE"), (0, 5, "SINGLE"),
        ],
    )
    is_valid_bad, violations_bad = bad_mol.check_valency()
    print(f"\nOvervalent carbon valid: {is_valid_bad}, violations: {violations_bad}")

    # ---- Molecular metrics (requires RDKit) ----
    test_smiles = ["CCO", "c1ccccc1", "CC(=O)O", "invalid_smiles", "CC(C)CC"]
    metrics = MolecularMetrics(reference_smiles=["CCO", "c1ccccc1"])
    results = metrics.evaluate(test_smiles)
    print(f"\nGeneration metrics: {results}")

    # ---- Property computation ----
    props = compute_molecular_properties("CC(=O)Oc1ccccc1C(=O)O")
    if props:
        print(f"\nAspirin properties:")
        for k, v in props.items():
            print(f"  {k}: {v:.4f}")
    else:
        print("\nRDKit not available — skipping property computation.")

    # ---- Fingerprint similarity ----
    sim = compute_fingerprint_similarity("CCO", "CCCO")
    if sim is not None:
        print(f"\nEthanol vs Propanol Tanimoto: {sim:.4f}")
    else:
        print("\nRDKit not available — skipping fingerprint similarity.")
