"""
31.5.4 Property Optimization — Implementation

Latent space optimization (gradient ascent, Bayesian optimization),
RL-based fine-tuning, genetic algorithms for molecules, and
multi-objective optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass


# ============================================================
# Property Oracle (Simulated)
# ============================================================

class PropertyOracle:
    """
    Simulated molecular property oracle for demonstration.

    In practice, this would wrap RDKit descriptors, docking
    simulations, or learned property predictors.

    Supports: qed, logp, sa_score, binding_affinity (simulated).
    """

    def __init__(self, target_properties: Optional[Dict[str, Tuple[float, float]]] = None):
        """
        Args:
            target_properties: Dict of {property: (target_value, tolerance)}.
        """
        self.target_properties = target_properties or {
            "qed": (0.9, 0.1),
            "logp": (2.5, 1.0),
        }
        self.call_count = 0

    def _simulate_property(self, smiles: str, prop_name: str) -> float:
        """Simulate a property value from SMILES (deterministic hash)."""
        hash_val = hash(smiles + prop_name) % 10000
        np.random.seed(hash_val)

        if prop_name == "qed":
            return np.random.beta(3, 2)  # range [0, 1]
        elif prop_name == "logp":
            return np.random.normal(2.0, 1.5)  # typical range [-2, 6]
        elif prop_name == "sa_score":
            return np.random.uniform(1.0, 10.0)  # 1=easy, 10=hard
        elif prop_name == "binding_affinity":
            return -np.random.exponential(2.0)  # negative (more negative = better)
        else:
            return np.random.randn()

    def evaluate(self, smiles: str) -> Dict[str, float]:
        """Evaluate all target properties for a molecule."""
        self.call_count += 1
        return {
            prop: self._simulate_property(smiles, prop)
            for prop in self.target_properties
        }

    def reward(self, smiles: str) -> float:
        """
        Compute a scalar reward based on how close properties
        are to their targets.

        Uses a Gaussian reward centered on each target.
        """
        props = self.evaluate(smiles)
        total_reward = 0.0

        for prop_name, (target, tolerance) in self.target_properties.items():
            if prop_name in props:
                value = props[prop_name]
                r = np.exp(-0.5 * ((value - target) / tolerance) ** 2)
                total_reward += r

        return total_reward / len(self.target_properties)

    def multi_objective_reward(self, smiles: str) -> Dict[str, float]:
        """Return individual objective scores (for Pareto analysis)."""
        props = self.evaluate(smiles)
        rewards = {}
        for prop_name, (target, tolerance) in self.target_properties.items():
            if prop_name in props:
                rewards[prop_name] = np.exp(
                    -0.5 * ((props[prop_name] - target) / tolerance) ** 2
                )
        return rewards


# ============================================================
# Latent Space Optimization
# ============================================================

class LatentSpaceOptimizer:
    """
    Optimize molecular properties in a VAE latent space.

    Supports gradient ascent and Bayesian optimization
    (via surrogate model).

    Args:
        decoder_fn: Function mapping latent vector z → SMILES.
        oracle: Property oracle.
        latent_dim: Dimension of the latent space.
    """

    def __init__(
        self,
        decoder_fn: Callable[[torch.Tensor], str],
        oracle: PropertyOracle,
        latent_dim: int = 64,
    ):
        self.decoder_fn = decoder_fn
        self.oracle = oracle
        self.latent_dim = latent_dim

    def gradient_ascent(
        self,
        property_predictor: nn.Module,
        z_init: Optional[torch.Tensor] = None,
        num_steps: int = 100,
        lr: float = 0.01,
        noise_scale: float = 0.01,
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Optimize latent vector via gradient ascent on a
        differentiable property predictor.

        Args:
            property_predictor: Maps z → predicted property score.
            z_init: Starting latent vector.
            num_steps: Number of gradient steps.
            lr: Step size.
            noise_scale: Gaussian noise for exploration.

        Returns:
            (optimized_z, history_of_predicted_scores)
        """
        if z_init is None:
            z = torch.randn(1, self.latent_dim, requires_grad=True)
        else:
            z = z_init.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([z], lr=lr)
        history = []

        for step in range(num_steps):
            optimizer.zero_grad()
            score = property_predictor(z)
            loss = -score.mean()  # maximize score
            loss.backward()
            optimizer.step()

            # Add exploration noise
            with torch.no_grad():
                z.add_(torch.randn_like(z) * noise_scale)

            history.append(score.item())

        return z.detach(), history

    def bayesian_optimization(
        self,
        num_iterations: int = 50,
        initial_points: int = 10,
        acquisition: str = "ucb",
        kappa: float = 2.0,
    ) -> Tuple[List[str], List[float]]:
        """
        Bayesian optimization over the latent space using a
        Gaussian process surrogate.

        Simplified implementation using a nearest-neighbor surrogate.

        Args:
            num_iterations: Total optimization budget.
            initial_points: Random initial evaluations.
            acquisition: 'ucb' or 'ei' (Expected Improvement).
            kappa: UCB exploration parameter.

        Returns:
            (best_smiles_list, corresponding_rewards)
        """
        # Collect initial random points
        Z_observed = []
        y_observed = []

        for _ in range(initial_points):
            z = torch.randn(1, self.latent_dim)
            smiles = self.decoder_fn(z)
            reward = self.oracle.reward(smiles)
            Z_observed.append(z.numpy().flatten())
            y_observed.append(reward)

        Z_observed = np.array(Z_observed)
        y_observed = np.array(y_observed)

        best_smiles = []
        best_rewards = []

        for iteration in range(num_iterations - initial_points):
            # Simple surrogate: predict based on distance-weighted neighbors
            candidates = []
            candidate_acq = []

            for _ in range(100):  # random candidate pool
                z_cand = np.random.randn(self.latent_dim)
                dists = np.linalg.norm(Z_observed - z_cand, axis=1)
                weights = np.exp(-dists)
                weights /= weights.sum() + 1e-8

                # Predicted mean and std
                mu = (weights * y_observed).sum()
                var = (weights * (y_observed - mu) ** 2).sum()
                sigma = np.sqrt(max(var, 1e-8))

                # Acquisition function
                if acquisition == "ucb":
                    acq = mu + kappa * sigma
                else:  # Expected Improvement
                    best_so_far = y_observed.max()
                    improvement = mu - best_so_far
                    if sigma > 1e-8:
                        z_score = improvement / sigma
                        from scipy.stats import norm
                        acq = improvement * norm.cdf(z_score) + sigma * norm.pdf(z_score)
                    else:
                        acq = max(improvement, 0.0)

                candidates.append(z_cand)
                candidate_acq.append(acq)

            # Select best candidate
            best_idx = np.argmax(candidate_acq)
            z_next = candidates[best_idx]

            z_tensor = torch.tensor(z_next, dtype=torch.float32).unsqueeze(0)
            smiles = self.decoder_fn(z_tensor)
            reward = self.oracle.reward(smiles)

            Z_observed = np.vstack([Z_observed, z_next])
            y_observed = np.append(y_observed, reward)

            best_smiles.append(smiles)
            best_rewards.append(reward)

        # Return top results
        sorted_idx = np.argsort(y_observed)[::-1]
        top_smiles = [self.decoder_fn(torch.tensor(Z_observed[i]).unsqueeze(0).float())
                      for i in sorted_idx[:5]]
        top_rewards = y_observed[sorted_idx[:5]].tolist()

        return top_smiles, top_rewards


# ============================================================
# Genetic Algorithm for Molecular Optimization
# ============================================================

@dataclass
class Molecule:
    """Simple molecule representation for GA."""
    smiles: str
    fitness: float = 0.0


class MolecularGeneticAlgorithm:
    """
    Genetic algorithm for molecular property optimization.

    Operates on SMILES strings with crossover and mutation
    operations.

    Args:
        oracle: Property oracle for fitness evaluation.
        population_size: Number of molecules per generation.
        mutation_rate: Probability of mutating each molecule.
        elite_fraction: Fraction of top molecules kept each generation.
    """

    # Simple mutation vocabulary
    ATOMS = ["C", "N", "O", "S", "F"]
    BONDS = ["", "=", "#"]

    def __init__(
        self,
        oracle: PropertyOracle,
        population_size: int = 100,
        mutation_rate: float = 0.3,
        elite_fraction: float = 0.2,
    ):
        self.oracle = oracle
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_size = max(1, int(population_size * elite_fraction))

    def _random_molecule(self) -> str:
        """Generate a random small molecule SMILES."""
        length = np.random.randint(2, 8)
        parts = []
        for _ in range(length):
            atom = np.random.choice(self.ATOMS)
            bond = np.random.choice(self.BONDS) if parts else ""
            parts.append(bond + atom)
        return "".join(parts)

    def _mutate(self, smiles: str) -> str:
        """Apply a random mutation to a SMILES string."""
        if len(smiles) < 2:
            return smiles

        mutation_type = np.random.choice(["replace", "insert", "delete"])
        chars = list(smiles)

        if mutation_type == "replace":
            pos = np.random.randint(len(chars))
            chars[pos] = np.random.choice(self.ATOMS)
        elif mutation_type == "insert":
            pos = np.random.randint(len(chars) + 1)
            new_atom = np.random.choice(self.ATOMS)
            chars.insert(pos, new_atom)
        elif mutation_type == "delete" and len(chars) > 2:
            pos = np.random.randint(len(chars))
            chars.pop(pos)

        return "".join(chars)

    def _crossover(self, parent_a: str, parent_b: str) -> str:
        """Single-point crossover between two SMILES strings."""
        if len(parent_a) < 2 or len(parent_b) < 2:
            return parent_a

        cut_a = np.random.randint(1, len(parent_a))
        cut_b = np.random.randint(1, len(parent_b))

        child = parent_a[:cut_a] + parent_b[cut_b:]
        return child[:50]  # length limit

    def evolve(
        self,
        initial_population: Optional[List[str]] = None,
        num_generations: int = 50,
        verbose: bool = True,
    ) -> List[Molecule]:
        """
        Run the genetic algorithm.

        Args:
            initial_population: Optional starting molecules.
            num_generations: Number of generations to evolve.
            verbose: Print progress.

        Returns:
            Final population sorted by fitness (best first).
        """
        # Initialize population
        if initial_population:
            pop = [Molecule(s) for s in initial_population[:self.population_size]]
            while len(pop) < self.population_size:
                pop.append(Molecule(self._random_molecule()))
        else:
            pop = [Molecule(self._random_molecule()) for _ in range(self.population_size)]

        # Evaluate initial fitness
        for mol in pop:
            mol.fitness = self.oracle.reward(mol.smiles)

        for gen in range(num_generations):
            # Sort by fitness (descending)
            pop.sort(key=lambda m: m.fitness, reverse=True)

            if verbose and (gen % 10 == 0 or gen == num_generations - 1):
                best = pop[0]
                avg = np.mean([m.fitness for m in pop])
                print(f"Gen {gen:3d}: best={best.fitness:.4f}, "
                      f"avg={avg:.4f}, smiles={best.smiles}")

            # Elitism: keep top molecules
            new_pop = pop[:self.elite_size]

            # Fill rest with crossover + mutation
            while len(new_pop) < self.population_size:
                # Tournament selection
                idx_a = min(
                    np.random.randint(len(pop)),
                    np.random.randint(len(pop)),
                )
                idx_b = min(
                    np.random.randint(len(pop)),
                    np.random.randint(len(pop)),
                )
                parent_a = pop[idx_a]
                parent_b = pop[idx_b]

                # Crossover
                child_smiles = self._crossover(parent_a.smiles, parent_b.smiles)

                # Mutation
                if np.random.rand() < self.mutation_rate:
                    child_smiles = self._mutate(child_smiles)

                child = Molecule(child_smiles)
                child.fitness = self.oracle.reward(child.smiles)
                new_pop.append(child)

            pop = new_pop

        pop.sort(key=lambda m: m.fitness, reverse=True)
        return pop


# ============================================================
# Multi-Objective Optimization Utilities
# ============================================================

def is_pareto_dominated(obj_a: np.ndarray, obj_b: np.ndarray) -> bool:
    """Check if obj_a is dominated by obj_b (all >= and at least one >)."""
    return bool(np.all(obj_b >= obj_a) and np.any(obj_b > obj_a))


def pareto_front(objectives: np.ndarray) -> np.ndarray:
    """
    Compute the Pareto front indices from an array of objective values.

    Args:
        objectives: [N, K] array of K objective values for N solutions.

    Returns:
        Boolean mask of Pareto-optimal solutions.
    """
    n = objectives.shape[0]
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if is_pareto_dominated(objectives[i], objectives[j]):
                is_pareto[i] = False
                break

    return is_pareto


def hypervolume_2d(points: np.ndarray, ref_point: np.ndarray) -> float:
    """
    Compute the 2D hypervolume indicator.

    Args:
        points: [N, 2] Pareto front points.
        ref_point: [2] reference point (worst-case).

    Returns:
        Hypervolume dominated by the Pareto front.
    """
    # Filter points dominating the reference
    valid = np.all(points > ref_point, axis=1)
    pts = points[valid]

    if len(pts) == 0:
        return 0.0

    # Sort by first objective descending
    sorted_idx = np.argsort(-pts[:, 0])
    pts = pts[sorted_idx]

    hv = 0.0
    prev_y = ref_point[1]

    for pt in pts:
        if pt[1] > prev_y:
            hv += (pt[0] - ref_point[0]) * (pt[1] - prev_y)
            prev_y = pt[1]

    return hv


class MultiObjectiveMolecularOptimizer:
    """
    Multi-objective molecular optimization using scalarization
    with varying weights to approximate the Pareto front.

    Args:
        oracle: Multi-objective property oracle.
        optimizer_factory: Factory that creates single-objective optimizers.
        num_weight_vectors: Number of scalarization weights to try.
    """

    def __init__(
        self,
        oracle: PropertyOracle,
        num_weight_vectors: int = 10,
    ):
        self.oracle = oracle
        self.num_objectives = len(oracle.target_properties)
        self.objective_names = list(oracle.target_properties.keys())

        # Generate evenly spaced weight vectors
        if self.num_objectives == 2:
            weights = np.linspace(0, 1, num_weight_vectors)
            self.weight_vectors = np.column_stack([weights, 1 - weights])
        else:
            # Random weights for >2 objectives
            self.weight_vectors = np.random.dirichlet(
                np.ones(self.num_objectives), size=num_weight_vectors,
            )

    def optimize(
        self,
        initial_smiles: List[str],
        num_generations: int = 30,
        verbose: bool = True,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Run multi-objective optimization.

        Returns:
            (pareto_smiles, pareto_objectives)
        """
        all_smiles = []
        all_objectives = []

        for w_idx, weights in enumerate(self.weight_vectors):
            if verbose:
                print(f"\nWeight vector {w_idx}: {weights}")

            # Create weighted oracle
            def weighted_reward(smiles, w=weights):
                obj_scores = self.oracle.multi_objective_reward(smiles)
                values = [obj_scores.get(name, 0.0) for name in self.objective_names]
                return float(np.dot(w, values))

            # Simple GA with weighted reward
            ga = MolecularGeneticAlgorithm(
                oracle=PropertyOracle(self.oracle.target_properties),
                population_size=50,
                mutation_rate=0.3,
            )
            # Override reward function
            ga.oracle.reward = weighted_reward

            result = ga.evolve(
                initial_population=initial_smiles,
                num_generations=num_generations,
                verbose=False,
            )

            # Collect top molecules
            for mol in result[:5]:
                obj_scores = self.oracle.multi_objective_reward(mol.smiles)
                objectives = [obj_scores.get(name, 0.0) for name in self.objective_names]
                all_smiles.append(mol.smiles)
                all_objectives.append(objectives)

        all_objectives = np.array(all_objectives)

        # Extract Pareto front
        pareto_mask = pareto_front(all_objectives)
        pareto_smiles = [s for s, m in zip(all_smiles, pareto_mask) if m]
        pareto_obj = all_objectives[pareto_mask]

        if verbose:
            print(f"\nPareto front: {pareto_mask.sum()} molecules out of {len(all_smiles)}")
            for i, (s, obj) in enumerate(zip(pareto_smiles, pareto_obj)):
                obj_str = ", ".join(f"{name}={v:.3f}"
                                   for name, v in zip(self.objective_names, obj))
                print(f"  [{i}] {s[:30]:30s} → {obj_str}")

        return pareto_smiles, pareto_obj


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    # ---- Property Oracle ----
    print("=== Property Oracle Demo ===")
    oracle = PropertyOracle({"qed": (0.8, 0.15), "logp": (2.5, 1.0)})
    props = oracle.evaluate("CCO")
    print(f"Ethanol properties: {props}")
    print(f"Ethanol reward: {oracle.reward('CCO'):.4f}")

    # ---- Genetic Algorithm ----
    print("\n=== Genetic Algorithm Demo ===")
    ga = MolecularGeneticAlgorithm(
        oracle=oracle,
        population_size=50,
        mutation_rate=0.3,
    )
    final_pop = ga.evolve(num_generations=30)
    print(f"\nTop 5 molecules:")
    for mol in final_pop[:5]:
        print(f"  {mol.smiles:20s} fitness={mol.fitness:.4f}")
    print(f"Oracle calls: {oracle.call_count}")

    # ---- Latent Space Optimization ----
    print("\n=== Latent Space Optimization Demo ===")

    # Simple mock decoder
    def mock_decoder(z: torch.Tensor) -> str:
        z_flat = z.detach().numpy().flatten()
        seed = int(abs(z_flat.sum()) * 1000) % 100000
        np.random.seed(seed)
        atoms = ["C", "N", "O", "S", "F"]
        length = np.random.randint(3, 10)
        return "".join(np.random.choice(atoms, size=length))

    # Simple property predictor
    predictor = nn.Sequential(
        nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid(),
    )

    oracle2 = PropertyOracle({"qed": (0.9, 0.1)})
    lso = LatentSpaceOptimizer(mock_decoder, oracle2, latent_dim=64)

    z_opt, history = lso.gradient_ascent(predictor, num_steps=50, lr=0.05)
    print(f"Gradient ascent — final predicted score: {history[-1]:.4f}")
    decoded = mock_decoder(z_opt)
    print(f"Decoded molecule: {decoded}")

    # ---- Pareto Front ----
    print("\n=== Multi-Objective / Pareto Front Demo ===")
    objectives = np.array([
        [0.8, 0.3],
        [0.7, 0.6],
        [0.5, 0.9],
        [0.6, 0.5],
        [0.3, 0.8],
        [0.9, 0.1],
    ])
    mask = pareto_front(objectives)
    print(f"Objectives:\n{objectives}")
    print(f"Pareto mask: {mask}")
    print(f"Pareto points:\n{objectives[mask]}")

    hv = hypervolume_2d(objectives[mask], ref_point=np.array([0.0, 0.0]))
    print(f"Hypervolume (2D): {hv:.4f}")
