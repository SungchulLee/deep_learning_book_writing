# 31.6.2 Transaction Graph Generation

## Overview

Transaction graphs record the flow of money between entities over time. Unlike static financial networks, transaction graphs are inherently **temporal** and **dynamic**: each edge represents a specific transaction at a specific time, with an amount, currency, and often additional metadata (transaction type, memo, risk flags). Generating realistic synthetic transaction graphs is essential for anti-money laundering (AML) model development, fraud detection training, privacy-preserving data sharing, and stress testing payment systems.

## Transaction Graph Structure

A transaction graph $\mathcal{G} = (V, E, T)$ consists of:

- **Nodes** $V$: entities (accounts, individuals, businesses)
  - Node features: entity type (individual/corporate), jurisdiction, account age, risk rating, average balance
- **Edges** $E$: directed transactions from sender to receiver
  - Edge features: amount, timestamp, currency, transaction type (wire, ACH, card), channel
- **Temporal ordering** $T$: transactions are ordered in time, and temporal patterns (frequency, periodicity, bursts) carry critical information

Key structural patterns that generators must capture:

**Fan-in / Fan-out**: Legitimate business accounts receive many small payments (fan-in) or distribute payments to many recipients (fan-out). Money laundering often involves **layering** (complex fan-out patterns to obscure origin).

**Cyclical patterns**: Salary payments, rent, subscriptions create regular temporal patterns. Anomalous cycles may indicate round-tripping (money returning to its origin through intermediaries).

**Amount distributions**: Transaction amounts follow heavy-tailed distributions with spikes at round numbers ($100, $500, structuring thresholds). Currency transaction reports (CTRs) are triggered above $10,000, leading to "structuring" where launderers split transactions to stay below the threshold.

**Temporal burstiness**: Transactions cluster in time—paydays, month-ends, holiday seasons. Legitimate accounts show predictable periodicity; fraudulent accounts may show unusual burst patterns.

## Synthetic Transaction Generation Approaches

### Rule-Based Generation

Define entity types and behavioral rules, then simulate:

1. **Entity profiles**: Each account has a type (individual, small business, large corporate) with associated behavioral parameters (transaction frequency, typical amounts, counterparty diversity).

2. **Transaction templates**: Define prototypical transaction patterns—salary deposits, bill payments, B2B invoices—with parameterized timing and amounts.

3. **Anomaly injection**: After generating a baseline of normal activity, inject specific money laundering typologies (structuring, layering, round-tripping) at controlled rates.

This approach gives full control over the anomaly rate and types, but the "normal" transactions may lack the complexity and correlations of real data.

### Deep Generative Approaches

**Temporal Point Process Models**: Model transactions as events in continuous time using a conditional intensity function:

$$\lambda(t \mid \mathcal{H}_t) = f_\theta(t, \mathcal{H}_t)$$

where $\mathcal{H}_t$ is the history of transactions up to time $t$. Neural Hawkes processes and Transformer-based point processes can capture complex temporal dependencies.

**Graph-Level Temporal Generation**: Combine graph generation (for topology) with temporal modeling (for transaction timing and amounts). At each time step, decide which pairs of nodes transact, what amount, and what type—conditioned on the network history.

**GAN-Based Transaction Generation**: Train a GAN where the generator produces sequences of transactions and the discriminator distinguishes real from synthetic transaction histories. The key challenge is preserving the relational structure (transactions link specific entities) while maintaining realistic temporal patterns.

## Privacy-Preserving Synthetic Data

A major motivation for transaction graph generation is creating **privacy-preserving synthetic data** that preserves statistical properties of real transaction data without exposing individual transactions:

**Differential privacy**: Add calibrated noise to the generation process to provide formal privacy guarantees. The challenge is maintaining data utility (realistic patterns) under privacy constraints.

**Synthetic data validation**: Evaluate synthetic transaction data on three axes:
1. **Fidelity**: Do aggregate statistics (amount distributions, temporal patterns, degree distributions) match the real data?
2. **Utility**: Do ML models trained on synthetic data perform comparably to those trained on real data (e.g., for fraud detection)?
3. **Privacy**: Can an adversary re-identify individuals or infer sensitive attributes from the synthetic data?

## AML Typology Injection

For training AML detection models, synthetic data must include realistic money laundering patterns:

**Structuring (Smurfing)**: Splitting large transactions into amounts below reporting thresholds ($10,000 in the US). Multiple deposits of $9,500 across different branches within a short window.

**Layering**: Moving funds through a complex chain of intermediary accounts to obscure the origin. The transaction graph shows a path from source to destination through multiple hops.

**Round-tripping**: Funds leave an account and return to the same account (or a related account) through a circuitous path. Detected as cycles in the transaction graph.

**Trade-based laundering**: Over- or under-invoicing in trade transactions to move value across borders. Appears as transactions with amounts inconsistent with the goods described.

## Evaluation Metrics

Beyond general graph statistics, transaction graph quality is measured by:

**Temporal fidelity**: Compare inter-arrival time distributions, daily/weekly volume patterns, and autocorrelation functions between real and synthetic data.

**Amount fidelity**: Compare transaction amount distributions (mean, variance, quantiles, round-number spikes).

**Downstream utility**: Train a fraud/AML detection model on synthetic data and evaluate on real data. The closer the performance to a model trained on real data, the higher the synthetic data quality.

**Graph structure**: Compare degree distributions, clustering coefficients, connected component sizes, and community structure.
