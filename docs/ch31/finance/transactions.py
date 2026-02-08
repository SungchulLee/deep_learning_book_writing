"""
31.6.2 Transaction Graph Generation — Implementation

Rule-based transaction graph generation with entity profiles,
temporal patterns, AML typology injection, and evaluation metrics.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


# ============================================================
# Data Structures
# ============================================================

@dataclass
class Transaction:
    """Single financial transaction."""
    sender: int
    receiver: int
    amount: float
    timestamp: float  # seconds from epoch
    tx_type: str = "wire"  # wire, ach, card, cash
    is_suspicious: bool = False
    label: str = "normal"  # normal, structuring, layering, round_trip


@dataclass
class Entity:
    """Financial entity (account holder)."""
    entity_id: int
    entity_type: str  # individual, small_business, large_corporate
    jurisdiction: str = "US"
    risk_rating: float = 0.0  # 0=low, 1=high
    avg_balance: float = 10000.0
    tx_frequency: float = 5.0  # transactions per day
    counterparty_diversity: int = 10


@dataclass
class TransactionGraph:
    """Complete transaction graph with entities and transactions."""
    entities: List[Entity]
    transactions: List[Transaction]

    @property
    def num_entities(self) -> int:
        return len(self.entities)

    @property
    def num_transactions(self) -> int:
        return len(self.transactions)

    def summary(self) -> Dict[str, float]:
        if not self.transactions:
            return {"num_entities": self.num_entities, "num_transactions": 0}

        amounts = [tx.amount for tx in self.transactions]
        timestamps = [tx.timestamp for tx in self.transactions]
        suspicious = sum(1 for tx in self.transactions if tx.is_suspicious)

        return {
            "num_entities": self.num_entities,
            "num_transactions": self.num_transactions,
            "mean_amount": float(np.mean(amounts)),
            "median_amount": float(np.median(amounts)),
            "max_amount": float(np.max(amounts)),
            "total_volume": float(np.sum(amounts)),
            "time_span_days": (max(timestamps) - min(timestamps)) / 86400,
            "suspicious_fraction": suspicious / len(self.transactions),
        }

    def get_adjacency(self) -> np.ndarray:
        """Build weighted adjacency (total flow between entities)."""
        n = self.num_entities
        adj = np.zeros((n, n))
        for tx in self.transactions:
            adj[tx.sender, tx.receiver] += tx.amount
        return adj

    def get_degree_sequence(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (out_degree, in_degree) arrays."""
        n = self.num_entities
        out_deg = np.zeros(n)
        in_deg = np.zeros(n)
        for tx in self.transactions:
            out_deg[tx.sender] += 1
            in_deg[tx.receiver] += 1
        return out_deg, in_deg


# ============================================================
# Entity Profile Generator
# ============================================================

class EntityProfileGenerator:
    """
    Generate realistic entity profiles with type-specific
    behavioral parameters.

    Args:
        num_individuals: Number of individual accounts.
        num_small_biz: Number of small business accounts.
        num_large_corp: Number of large corporate accounts.
    """

    PROFILES = {
        "individual": {
            "avg_balance_mean": 5000, "avg_balance_std": 8000,
            "tx_freq_mean": 3.0, "tx_freq_std": 2.0,
            "counterparty_mean": 8, "counterparty_std": 5,
        },
        "small_business": {
            "avg_balance_mean": 50000, "avg_balance_std": 40000,
            "tx_freq_mean": 15.0, "tx_freq_std": 8.0,
            "counterparty_mean": 25, "counterparty_std": 15,
        },
        "large_corporate": {
            "avg_balance_mean": 5000000, "avg_balance_std": 3000000,
            "tx_freq_mean": 100.0, "tx_freq_std": 50.0,
            "counterparty_mean": 100, "counterparty_std": 50,
        },
    }

    def __init__(self, num_individuals=70, num_small_biz=25, num_large_corp=5):
        self.counts = {
            "individual": num_individuals,
            "small_business": num_small_biz,
            "large_corporate": num_large_corp,
        }

    def generate(self) -> List[Entity]:
        entities = []
        eid = 0
        for etype, count in self.counts.items():
            profile = self.PROFILES[etype]
            for _ in range(count):
                entity = Entity(
                    entity_id=eid,
                    entity_type=etype,
                    avg_balance=max(100, np.random.normal(
                        profile["avg_balance_mean"], profile["avg_balance_std"])),
                    tx_frequency=max(0.1, np.random.normal(
                        profile["tx_freq_mean"], profile["tx_freq_std"])),
                    counterparty_diversity=max(1, int(np.random.normal(
                        profile["counterparty_mean"], profile["counterparty_std"]))),
                    risk_rating=np.random.beta(2, 10),  # most are low risk
                )
                entities.append(entity)
                eid += 1
        return entities


# ============================================================
# Transaction Graph Generator
# ============================================================

class TransactionGraphGenerator:
    """
    Rule-based synthetic transaction graph generator.

    Generates normal transaction activity based on entity profiles,
    then optionally injects AML typologies.

    Args:
        entities: List of Entity objects.
        days: Number of days to simulate.
        base_timestamp: Starting timestamp (seconds).
    """

    def __init__(
        self,
        entities: List[Entity],
        days: int = 30,
        base_timestamp: float = 1700000000.0,
    ):
        self.entities = entities
        self.days = days
        self.base_timestamp = base_timestamp
        self.num_entities = len(entities)

    def _select_counterparty(self, sender: Entity) -> int:
        """Select a transaction counterparty based on entity profile."""
        # Prefer counterparties of compatible type
        weights = np.ones(self.num_entities)
        weights[sender.entity_id] = 0  # no self-transactions

        # Corporates transact more with other corporates and businesses
        for i, e in enumerate(self.entities):
            if sender.entity_type == "large_corporate":
                if e.entity_type == "large_corporate":
                    weights[i] *= 3.0
                elif e.entity_type == "small_business":
                    weights[i] *= 2.0
            elif sender.entity_type == "individual":
                if e.entity_type == "small_business":
                    weights[i] *= 2.0  # retail spending

        weights /= weights.sum()
        return int(np.random.choice(self.num_entities, p=weights))

    def _sample_amount(self, sender: Entity) -> float:
        """Sample transaction amount based on entity profile."""
        base = sender.avg_balance * 0.05
        amount = np.random.lognormal(mean=np.log(max(base, 10)), sigma=0.8)

        # Round number bias (20% of transactions)
        if np.random.rand() < 0.2:
            round_values = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
            amount = float(np.random.choice(round_values))

        return round(max(0.01, amount), 2)

    def _sample_tx_type(self, amount: float) -> str:
        """Assign transaction type based on amount."""
        if amount < 50:
            return np.random.choice(["card", "ach"], p=[0.7, 0.3])
        elif amount < 5000:
            return np.random.choice(["ach", "card", "wire"], p=[0.5, 0.3, 0.2])
        else:
            return np.random.choice(["wire", "ach"], p=[0.6, 0.4])

    def generate_normal(self) -> List[Transaction]:
        """Generate normal (non-suspicious) transactions."""
        transactions = []

        for entity in self.entities:
            # Expected total transactions over the period
            expected_count = int(entity.tx_frequency * self.days)
            actual_count = max(1, np.random.poisson(expected_count))

            # Select a counterparty pool
            pool_size = min(entity.counterparty_diversity, self.num_entities - 1)
            all_others = [i for i in range(self.num_entities) if i != entity.entity_id]
            if pool_size < len(all_others):
                counterparty_pool = np.random.choice(
                    all_others, size=pool_size, replace=False
                ).tolist()
            else:
                counterparty_pool = all_others

            for _ in range(actual_count):
                receiver = int(np.random.choice(counterparty_pool))
                amount = self._sample_amount(entity)
                # Timestamp: uniform within the simulation period + weekly pattern
                day_offset = np.random.uniform(0, self.days)
                # Reduce weekend activity
                weekday = int(day_offset) % 7
                if weekday >= 5 and np.random.rand() < 0.6:
                    day_offset = int(day_offset) - weekday + np.random.randint(0, 5)

                hour = np.random.normal(13, 3)  # peak at 1 PM
                hour = np.clip(hour, 6, 22)
                timestamp = self.base_timestamp + day_offset * 86400 + hour * 3600

                tx_type = self._sample_tx_type(amount)

                transactions.append(Transaction(
                    sender=entity.entity_id,
                    receiver=receiver,
                    amount=amount,
                    timestamp=timestamp,
                    tx_type=tx_type,
                    is_suspicious=False,
                    label="normal",
                ))

        # Sort by timestamp
        transactions.sort(key=lambda tx: tx.timestamp)
        return transactions

    def inject_structuring(
        self, transactions: List[Transaction],
        num_patterns: int = 5,
        threshold: float = 10000.0,
    ) -> List[Transaction]:
        """
        Inject structuring (smurfing) patterns.

        Splits a large amount into sub-threshold transactions
        from the same sender within a short time window.
        """
        injected = list(transactions)

        for _ in range(num_patterns):
            sender_id = np.random.randint(self.num_entities)
            total_amount = np.random.uniform(15000, 50000)
            num_splits = int(np.ceil(total_amount / (threshold * 0.95)))

            base_time = self.base_timestamp + np.random.uniform(0, self.days) * 86400
            receivers = np.random.choice(
                [i for i in range(self.num_entities) if i != sender_id],
                size=min(num_splits, self.num_entities - 1),
                replace=True,
            )

            remaining = total_amount
            for k, recv in enumerate(receivers):
                split_amount = min(remaining, np.random.uniform(
                    threshold * 0.7, threshold * 0.95))
                split_amount = round(split_amount, 2)
                remaining -= split_amount

                injected.append(Transaction(
                    sender=sender_id,
                    receiver=int(recv),
                    amount=split_amount,
                    timestamp=base_time + k * np.random.uniform(1800, 7200),
                    tx_type="cash",
                    is_suspicious=True,
                    label="structuring",
                ))

                if remaining <= 0:
                    break

        injected.sort(key=lambda tx: tx.timestamp)
        return injected

    def inject_layering(
        self, transactions: List[Transaction],
        num_patterns: int = 3,
        chain_length: int = 4,
    ) -> List[Transaction]:
        """
        Inject layering patterns: money moves through a chain
        of intermediary accounts.
        """
        injected = list(transactions)

        for _ in range(num_patterns):
            amount = np.random.uniform(20000, 100000)
            chain = np.random.choice(
                self.num_entities, size=min(chain_length + 1, self.num_entities),
                replace=False,
            ).tolist()

            base_time = self.base_timestamp + np.random.uniform(0, self.days) * 86400

            for hop in range(len(chain) - 1):
                # Small fee deducted at each hop
                hop_amount = round(amount * (1 - 0.01 * hop), 2)
                injected.append(Transaction(
                    sender=chain[hop],
                    receiver=chain[hop + 1],
                    amount=hop_amount,
                    timestamp=base_time + hop * np.random.uniform(3600, 86400),
                    tx_type="wire",
                    is_suspicious=True,
                    label="layering",
                ))

        injected.sort(key=lambda tx: tx.timestamp)
        return injected

    def inject_round_tripping(
        self, transactions: List[Transaction],
        num_patterns: int = 3,
        cycle_length: int = 3,
    ) -> List[Transaction]:
        """
        Inject round-tripping: money leaves and returns to the
        same account through intermediaries.
        """
        injected = list(transactions)

        for _ in range(num_patterns):
            amount = np.random.uniform(10000, 80000)
            participants = np.random.choice(
                self.num_entities,
                size=min(cycle_length, self.num_entities),
                replace=False,
            ).tolist()

            base_time = self.base_timestamp + np.random.uniform(0, self.days) * 86400

            # Form a cycle: A→B→C→...→A
            for hop in range(len(participants)):
                sender = participants[hop]
                receiver = participants[(hop + 1) % len(participants)]
                hop_amount = round(amount * (1 - 0.005 * hop), 2)

                injected.append(Transaction(
                    sender=sender,
                    receiver=receiver,
                    amount=hop_amount,
                    timestamp=base_time + hop * np.random.uniform(7200, 172800),
                    tx_type="wire",
                    is_suspicious=True,
                    label="round_trip",
                ))

        injected.sort(key=lambda tx: tx.timestamp)
        return injected

    def generate(self, inject_aml: bool = True) -> TransactionGraph:
        """
        Generate a complete transaction graph.

        Args:
            inject_aml: Whether to inject AML typologies.

        Returns:
            TransactionGraph with entities and transactions.
        """
        transactions = self.generate_normal()

        if inject_aml:
            transactions = self.inject_structuring(transactions)
            transactions = self.inject_layering(transactions)
            transactions = self.inject_round_tripping(transactions)

        return TransactionGraph(
            entities=self.entities,
            transactions=transactions,
        )


# ============================================================
# Transaction Graph Evaluation
# ============================================================

class TransactionGraphEvaluator:
    """
    Evaluate quality of synthetic transaction graphs by comparing
    statistical properties with a reference graph.
    """

    @staticmethod
    def amount_distribution_stats(graph: TransactionGraph) -> Dict[str, float]:
        """Compute transaction amount distribution statistics."""
        amounts = np.array([tx.amount for tx in graph.transactions])
        if len(amounts) == 0:
            return {}
        return {
            "mean": float(amounts.mean()),
            "std": float(amounts.std()),
            "median": float(np.median(amounts)),
            "p95": float(np.percentile(amounts, 95)),
            "p99": float(np.percentile(amounts, 99)),
            "skewness": float(
                ((amounts - amounts.mean()) ** 3).mean() / (amounts.std() ** 3 + 1e-10)
            ),
        }

    @staticmethod
    def temporal_stats(graph: TransactionGraph) -> Dict[str, float]:
        """Compute temporal distribution statistics."""
        timestamps = np.array(sorted([tx.timestamp for tx in graph.transactions]))
        if len(timestamps) < 2:
            return {}

        inter_arrival = np.diff(timestamps)
        # Convert to hours
        inter_hours = inter_arrival / 3600

        return {
            "mean_inter_arrival_hrs": float(inter_hours.mean()),
            "std_inter_arrival_hrs": float(inter_hours.std()),
            "median_inter_arrival_hrs": float(np.median(inter_hours)),
            "tx_per_day": float(len(timestamps) / max(
                (timestamps[-1] - timestamps[0]) / 86400, 1)),
        }

    @staticmethod
    def compare(
        real: TransactionGraph, synthetic: TransactionGraph,
    ) -> Dict[str, float]:
        """Compare real and synthetic transaction graphs."""
        real_amounts = TransactionGraphEvaluator.amount_distribution_stats(real)
        synth_amounts = TransactionGraphEvaluator.amount_distribution_stats(synthetic)

        real_temporal = TransactionGraphEvaluator.temporal_stats(real)
        synth_temporal = TransactionGraphEvaluator.temporal_stats(synthetic)

        comparison = {}
        for key in real_amounts:
            if key in synth_amounts:
                comparison[f"amount_{key}_diff"] = abs(
                    real_amounts[key] - synth_amounts[key]
                )
        for key in real_temporal:
            if key in synth_temporal:
                comparison[f"temporal_{key}_diff"] = abs(
                    real_temporal[key] - synth_temporal[key]
                )

        return comparison


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)

    # ---- Generate entities ----
    print("=== Entity Generation ===")
    profile_gen = EntityProfileGenerator(
        num_individuals=50, num_small_biz=15, num_large_corp=5,
    )
    entities = profile_gen.generate()
    print(f"Entities: {len(entities)}")
    for etype in ["individual", "small_business", "large_corporate"]:
        count = sum(1 for e in entities if e.entity_type == etype)
        avg_bal = np.mean([e.avg_balance for e in entities if e.entity_type == etype])
        print(f"  {etype}: {count} entities, avg balance=${avg_bal:,.0f}")

    # ---- Generate transaction graph ----
    print("\n=== Transaction Graph Generation ===")
    gen = TransactionGraphGenerator(entities, days=30)
    graph = gen.generate(inject_aml=True)
    print(graph.summary())

    # ---- Label distribution ----
    label_counts = defaultdict(int)
    for tx in graph.transactions:
        label_counts[tx.label] += 1
    print(f"\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} ({100*count/graph.num_transactions:.1f}%)")

    # ---- Evaluate ----
    print("\n=== Distribution Statistics ===")
    amount_stats = TransactionGraphEvaluator.amount_distribution_stats(graph)
    print(f"Amount stats: {amount_stats}")

    temporal_stats = TransactionGraphEvaluator.temporal_stats(graph)
    print(f"Temporal stats: {temporal_stats}")

    # ---- Adjacency ----
    adj = graph.get_adjacency()
    density = (adj > 0).sum() / (graph.num_entities * (graph.num_entities - 1))
    print(f"\nAdjacency density: {density:.4f}")

    out_deg, in_deg = graph.get_degree_sequence()
    print(f"Mean out-degree: {out_deg.mean():.1f}")
    print(f"Mean in-degree: {in_deg.mean():.1f}")
    print(f"Max out-degree: {out_deg.max():.0f}")
