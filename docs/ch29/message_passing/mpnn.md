# 29.2.4 Message Passing Neural Network (MPNN)

## Introduction

The **Message Passing Neural Network (MPNN)** framework, proposed by Gilmer et al. (2017) in *"Neural Message Passing for Quantum Chemistry"*, unifies many graph neural network architectures under a single abstract formulation. It provides the theoretical and practical foundation for designing and analyzing GNNs.

## MPNN Formulation

The MPNN framework consists of two phases:

### Message Passing Phase

For $T$ time steps (layers), each node updates its hidden state:

$$\mathbf{m}_v^{t+1} = \sum_{u \in \mathcal{N}(v)} M_t\left(\mathbf{h}_v^t, \mathbf{h}_u^t, \mathbf{e}_{vu}\right)$$

$$\mathbf{h}_v^{t+1} = U_t\left(\mathbf{h}_v^t, \mathbf{m}_v^{t+1}\right)$$

where:
- $M_t$: Message function at step $t$
- $U_t$: Update function at step $t$ (e.g., GRU)
- $\mathbf{e}_{vu}$: Edge features between $v$ and $u$

### Readout Phase

For graph-level tasks, a readout function produces a graph-level representation:

$$\hat{y} = R\left(\{\mathbf{h}_v^T : v \in V\}\right)$$

The readout $R$ must be invariant to node permutation. Common choices include sum, mean, or attention-based pooling.

## Existing Models as MPNN Instances

### GCN (Kipf & Welling, 2017)
$$M_t(\mathbf{h}_u) = \frac{1}{\sqrt{d_u d_v}} W^t \mathbf{h}_u^t, \quad U_t(\mathbf{h}_v, \mathbf{m}_v) = \sigma(\mathbf{m}_v)$$
Self-loops provide self-information; no explicit separate update.

### GraphSAGE (Hamilton et al., 2017)
$$M_t(\mathbf{h}_u) = \mathbf{h}_u^t, \quad U_t(\mathbf{h}_v, \mathbf{m}_v) = \sigma\left(W^t [\mathbf{h}_v^t \| \mathbf{m}_v]\right)$$
Explicit separation of self and neighbor information.

### GAT (Velickovic et al., 2018)
$$M_t(\mathbf{h}_u, \mathbf{h}_v) = \alpha_{uv} W^t \mathbf{h}_u^t$$
$$\alpha_{uv} = \text{softmax}_u\left(\text{LeakyReLU}\left(\mathbf{a}^T [W\mathbf{h}_v \| W\mathbf{h}_u]\right)\right)$$

### GIN (Xu et al., 2019)
$$M_t(\mathbf{h}_u) = \mathbf{h}_u^t, \quad U_t(\mathbf{h}_v, \mathbf{m}_v) = \text{MLP}\left((1+\epsilon^t) \mathbf{h}_v^t + \mathbf{m}_v\right)$$
Provably as powerful as the 1-WL test.

### Edge-Conditioned Convolution (Simonovsky & Komodakis, 2017)
$$M_t(\mathbf{h}_u, \mathbf{e}_{vu}) = \Theta(\mathbf{e}_{vu}) \mathbf{h}_u^t$$
Edge features parameterize the message function via a neural network $\Theta$.

## MPNN for Graph-Level Prediction

For tasks like molecular property prediction, the MPNN pipeline is:

1. **Input**: Graph $G = (V, E)$ with node features $\mathbf{x}_v$ and edge features $\mathbf{e}_{uv}$
2. **Initialize**: $\mathbf{h}_v^0 = \mathbf{x}_v$
3. **Message passing**: Run $T$ rounds of message passing
4. **Readout**: $\hat{y} = R(\{\mathbf{h}_v^T\})$
5. **Loss**: $\mathcal{L}(\hat{y}, y)$ (e.g., MSE for regression, CE for classification)

## Theoretical Properties

### Expressiveness
- MPNNs are bounded by the 1-WL test in distinguishing graphs
- GIN achieves this upper bound
- Higher-order GNNs (k-WL) can be more expressive but more expensive

### Computation
- Time complexity per layer: $O(|E| \cdot d)$ where $d$ is the feature dimension
- Memory: $O(|V| \cdot d + |E| \cdot d)$
- Parallelizable across edges (GPU-friendly)

### Limitations
- Cannot detect certain structural properties (e.g., cycles of specific lengths)
- Long-range dependencies require many layers (over-smoothing risk)
- Expressiveness limited by 1-WL bound

## Quantitative Finance MPNN Applications

The MPNN framework naturally models financial problems:

- **Portfolio risk assessment**: Message passing aggregates correlated risks through asset networks
- **Fraud detection**: Transaction patterns propagate through account networks
- **Credit scoring**: Counterparty risk flows through lending networks
- **Market microstructure**: Order flow information propagates through trading networks

## Summary

The MPNN framework provides a unified language for understanding and designing graph neural networks. By specifying the message function, aggregation, update rule, and readout, one can instantiate most existing GNN architectures or design new ones tailored to specific applications.
