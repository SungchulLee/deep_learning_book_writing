# Neural Collaborative Filtering

## Learning Objectives

- Understand why replacing the dot product with a neural network increases expressiveness
- Analyze the architecture of neural collaborative filtering (NCF)
- Implement NCF in PyTorch with dropout regularization
- Compare NCF to matrix factorization theoretically and empirically
- Understand the NeuMF architecture that combines MF and MLP paths

## Beyond the Dot Product

### Limitation of Bilinear Models

Both MF and MF-bias predict ratings using a **bilinear** interaction:

$$\hat{R}_{ui} = \mathbf{p}_u^\top \mathbf{q}_i + (\text{bias terms})$$

The dot product is a **linear** function of each embedding given the other. This means:

1. **Fixed interaction structure**: The contribution of each latent dimension is independent and additive: $\sum_{k=1}^d p_{uk} \cdot q_{ik}$.
2. **No feature crossing**: Dimension $k$ of the user embedding only interacts with dimension $k$ of the item embedding — there are no cross-dimensional interactions.
3. **Monotonic similarity**: If $\mathbf{p}_u$ and $\mathbf{q}_i$ have a large dot product, the model always predicts a high rating. There is no way to model non-monotonic relationships.

### The NCF Idea

**Replace the dot product with a learned function.** Instead of:

$$\hat{R}_{ui} = \mathbf{p}_u^\top \mathbf{q}_i$$

use:

$$\hat{R}_{ui} = f_\theta(\mathbf{p}_u, \mathbf{q}_i)$$

where $f_\theta$ is a neural network. The simplest choice: concatenate the embeddings and pass through an MLP.

!!! note "Key Architectural Insight"
    In the MF model, user and item embeddings must have the **same** dimension because they are combined via dot product. In NCF, embeddings are concatenated, so they can have **different** dimensions. This adds a design degree of freedom.

## Architecture

### General NCF Framework

The NCF architecture from He et al. (2017) consists of:

1. **Embedding layer**: Map user ID $u$ to $\mathbf{p}_u \in \mathbb{R}^{d_u}$ and item ID $i$ to $\mathbf{q}_i \in \mathbb{R}^{d_i}$.
2. **Concatenation**: Form $\mathbf{x} = [\mathbf{p}_u ; \mathbf{q}_i] \in \mathbb{R}^{d_u + d_i}$.
3. **Hidden layers**: $\mathbf{h}_1 = \sigma(W_1 \mathbf{x} + \mathbf{b}_1)$, $\mathbf{h}_2 = \sigma(W_2 \mathbf{h}_1 + \mathbf{b}_2)$, etc.
4. **Output**: $\hat{R}_{ui} = \mathbf{w}^\top \mathbf{h}_L + b$ (linear output for regression).

### The `CollabFNet` Implementation

The source code implements a two-layer NCF:

```python
class CollabFNet(nn.Module):
    """
    Neural Collaborative Filtering Network.
    
    Architecture:
        [user_emb | item_emb] → ReLU → Dropout → Linear → ReLU → Linear → output
        
    The concatenation allows cross-dimensional interactions that
    the dot product in MF cannot capture.
    """
    def __init__(self, num_users, num_items, emb_size=100, n_hidden=10):
        super(CollabFNet, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.lin1 = nn.Linear(emb_size * 2, n_hidden)   # 200 → 10
        self.lin2 = nn.Linear(n_hidden, 1)               # 10 → 1
        self.drop1 = nn.Dropout(0.1)

    def forward(self, u, v):
        U = self.user_emb(u)                      # (batch, emb_size)
        V = self.item_emb(v)                      # (batch, emb_size)
        x = F.relu(torch.cat([U, V], dim=1))      # (batch, emb_size * 2)
        x = self.drop1(x)                         # dropout after concatenation
        x = F.relu(self.lin1(x))                  # (batch, n_hidden)
        x = self.lin2(x)                          # (batch, 1)
        return x
```

### Layer-by-Layer Analysis

**Concatenation + ReLU**:

$$\mathbf{x} = \text{ReLU}\bigl([\mathbf{p}_u ; \mathbf{q}_i]\bigr)$$

Since embeddings are initialized to small positive values and the concatenation preserves these, the ReLU here acts as a passthrough initially. As training progresses and some embedding values become negative, ReLU provides nonlinearity at the input level.

**Dropout** ($p = 0.1$): During training, each element of $\mathbf{x}$ is independently zeroed with probability 0.1 and scaled by $\frac{1}{1-0.1}$. This prevents co-adaptation of embedding dimensions and acts as a form of regularization.

**Hidden layer** ($200 \to 10$):

$$\mathbf{h} = \text{ReLU}(W_1 \mathbf{x} + \mathbf{b}_1)$$

The weight matrix $W_1 \in \mathbb{R}^{10 \times 200}$ creates **cross-dimensional interactions**. Each hidden unit computes a weighted combination of all user and item embedding dimensions — something the dot product cannot do.

**Bottleneck structure**: The aggressive reduction from 200 to 10 dimensions forces the network to learn a compact representation of user–item compatibility. This acts as an implicit regularizer.

**Output layer** ($10 \to 1$):

$$\hat{R}_{ui} = \mathbf{w}^\top \mathbf{h} + b$$

A linear projection to a single scalar — the predicted rating. Note there is no activation function on the output, allowing the prediction to take any real value.

!!! warning "`unsqueeze` Requirement"
    The output shape is `(batch, 1)` rather than `(batch,)` as in the MF models. The training code handles this with the `unsqueeze` flag, reshaping the target ratings from `(batch,)` to `(batch, 1)` for the MSE loss computation.

## Expressiveness Analysis

### Universal Approximation

By the universal approximation theorem (Chapter 7), an MLP with sufficient width can approximate any continuous function on a compact domain. Therefore, the NCF model can approximate any continuous rating function $R(u, i)$, given sufficient hidden units.

In contrast, MF is restricted to bilinear functions:

$$\{f : f(\mathbf{p}, \mathbf{q}) = \mathbf{p}^\top A \mathbf{q} + \mathbf{b}^\top \mathbf{p} + \mathbf{c}^\top \mathbf{q} + d\}$$

This is a strict subset of continuous functions.

### Can NCF Recover MF?

Yes. If $d_u = d_i = d$ and the hidden layer has at least $d$ units, NCF can learn to implement the dot product: set $W_1$ to extract corresponding dimension pairs and multiply them, then $W_2$ sums the results. This shows MF is a **special case** of NCF.

### When Does NCF Outperform MF?

NCF shines when the true interaction function is **nonlinear**. Examples:

- **Threshold effects**: A user likes a movie only if it scores high on *both* action and humor dimensions (logical AND, not sum).
- **Complex preferences**: A user's rating depends on the *ratio* of two latent factors, not their sum.
- **Interaction effects**: The importance of one latent factor depends on the value of another.

In practice, the improvement over MF is often modest on standard benchmarks but can be significant on larger, more complex datasets.

## Deeper Architectures

### DeepNCF

The source code also provides a deeper architecture with configurable layers and batch normalization:

```python
class DeepNCF(nn.Module):
    """
    Deeper NCF with tower pattern — each layer halves the dimension.
    """
    def __init__(self, num_users, num_items, emb_size=100,
                 hidden_layers=None, dropout=0.2):
        super(DeepNCF, self).__init__()
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]
        
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        # Build MLP with batch normalization
        layers = []
        input_dim = emb_size * 2
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, u, v):
        U = self.user_emb(u)
        V = self.item_emb(v)
        x = torch.cat([U, V], dim=1)
        return self.mlp(x)
```

The tower pattern `[200 → 128 → 64 → 32 → 1]` creates progressive abstraction of the user–item interaction. However, deeper networks are harder to train with sparse recommendation data — the bottleneck design in `CollabFNet` (200 → 10) is conservative but effective for MovieLens-Small.

## NeuMF: Combining MF and NCF

He et al. (2017) proposed **NeuMF**, which runs MF and MLP paths in parallel and combines their outputs:

$$\hat{R}_{ui} = \sigma\Bigl(\mathbf{w}^\top \bigl[\underbrace{\mathbf{p}_u^{(1)} \odot \mathbf{q}_i^{(1)}}_{\text{GMF path}} ; \underbrace{\text{MLP}([\mathbf{p}_u^{(2)} ; \mathbf{q}_i^{(2)}])}_{\text{MLP path}}\bigr] + b\Bigr)$$

Each path has **separate** embeddings, allowing them to learn different representations. The GMF (Generalized Matrix Factorization) path captures linear interactions while the MLP path captures nonlinear ones.

```python
class NeuMF(nn.Module):
    """
    Neural Matrix Factorization — parallel GMF + MLP paths
    with separate embedding tables.
    """
    def __init__(self, num_users, num_items, gmf_emb_size=32,
                 mlp_emb_size=32, hidden_layers=None, dropout=0.2):
        super(NeuMF, self).__init__()
        if hidden_layers is None:
            hidden_layers = [64, 32, 16]

        # Separate embeddings for each path
        self.gmf_user_emb = nn.Embedding(num_users, gmf_emb_size)
        self.gmf_item_emb = nn.Embedding(num_items, gmf_emb_size)
        self.mlp_user_emb = nn.Embedding(num_users, mlp_emb_size)
        self.mlp_item_emb = nn.Embedding(num_items, mlp_emb_size)

        # MLP layers
        mlp_layers = []
        input_dim = mlp_emb_size * 2
        for hidden_dim in hidden_layers:
            mlp_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            input_dim = hidden_dim
        self.mlp = nn.Sequential(*mlp_layers)

        # Combine GMF output + MLP output → prediction
        self.predict = nn.Linear(gmf_emb_size + hidden_layers[-1], 1)

    def forward(self, u, v):
        # GMF path: element-wise product
        gmf_out = self.gmf_user_emb(u) * self.gmf_item_emb(v)

        # MLP path: concatenation through hidden layers
        mlp_in = torch.cat([self.mlp_user_emb(u), self.mlp_item_emb(v)], dim=1)
        mlp_out = self.mlp(mlp_in)

        # Combine and predict
        combined = torch.cat([gmf_out, mlp_out], dim=1)
        return self.predict(combined)
```

## Parameter Analysis

For `CollabFNet` with `emb_size=100`, `n_hidden=10`:

| Component | Parameters |
|-----------|-----------|
| `user_emb` | $m \times 100$ |
| `item_emb` | $n \times 100$ |
| `lin1` | $200 \times 10 + 10 = 2{,}010$ |
| `lin2` | $10 \times 1 + 1 = 11$ |
| **MLP total** | $2{,}021$ |
| **Embedding total** | $(m + n) \times 100$ |

The MLP adds only ~2K parameters on top of the embeddings — the bottleneck design keeps the model compact.

## Comparison: MF vs MF-Bias vs NCF

| Aspect | MF | MF-Bias | NCF |
|--------|----|---------|----|
| **Interaction** | Dot product | Dot product + bias | MLP |
| **Expressiveness** | Bilinear | Bilinear + affine | Universal |
| **Cross-dim interaction** | No | No | Yes |
| **Bias modeling** | No | Explicit | Implicit (learned) |
| **Embedding dim constraint** | $d_u = d_i$ | $d_u = d_i$ | $d_u \neq d_i$ allowed |
| **Output shape** | `(batch,)` | `(batch,)` | `(batch, 1)` |
| **Regularization** | Weight decay | Weight decay | Dropout + weight decay |
| **Interpretability** | High (factor analysis) | High | Lower |

## Practical Considerations

### Dropout Placement

The source places dropout after concatenation but before the hidden layer. Alternative placements:

- **After hidden layer**: Regularizes the learned representation, common in practice.
- **On embeddings directly**: Zeroes random embedding dimensions, forcing robustness.
- **Multiple dropout layers**: Different rates at different depths.

For small datasets like MovieLens-Small, a single dropout layer with $p = 0.1$ is often sufficient.

### Output Activation

The source uses no output activation (linear output), which is appropriate for regression with unbounded targets. For bounded ratings (e.g., 1–5), you might add:

```python
# Constrain output to [1, 5]
return 1 + 4 * torch.sigmoid(self.lin2(x))
```

This ensures predictions fall within the valid range, but can cause gradient saturation near the boundaries.

## Summary

Neural collaborative filtering replaces the dot product in MF with a neural network, enabling nonlinear user–item interactions and cross-dimensional feature crossing. The `CollabFNet` implementation uses a concatenation-based architecture with a single hidden layer and dropout regularization. While more expressive than MF, the practical gains depend on dataset complexity and careful regularization. The NeuMF architecture combines MF and MLP paths for the best of both worlds.

---

## Exercises

1. **Dot product recovery**: Construct specific weight matrices for `CollabFNet` (with `n_hidden ≥ emb_size`) that exactly implement the dot product $\mathbf{p}_u^\top \mathbf{q}_i$. Verify numerically.

2. **Architecture search**: Experiment with different hidden layer configurations on MovieLens: `[10]`, `[64]`, `[128, 64]`, `[256, 128, 64]`. Report validation MSE and training time. Is deeper always better?

3. **Dropout analysis**: Train NCF with dropout rates $p \in \{0.0, 0.1, 0.2, 0.3, 0.5\}$. Plot training loss and validation loss vs $p$. At what rate does training loss start to significantly diverge from validation loss?

4. **Asymmetric embeddings**: Modify `CollabFNet` to use `user_emb_size=50` and `item_emb_size=150` (total concatenation still 200). Does breaking the symmetry help?

5. **NeuMF implementation**: Train the NeuMF model with separate embeddings for the GMF and MLP paths. Compare against standalone MF and NCF on MovieLens. Does the ensemble help?

6. **Feature importance**: After training NCF, compute the gradient of the output with respect to each embedding dimension. Which dimensions contribute most to predictions? How does this compare to the latent factor analysis in MF?
