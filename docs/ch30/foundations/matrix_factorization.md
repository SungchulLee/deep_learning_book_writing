# Matrix Factorization

## Learning Objectives

- Derive the matrix factorization objective from low-rank approximation
- Connect MF to SVD and explain why direct SVD fails on sparse matrices
- Understand the embedding interpretation and PyTorch implementation
- Derive the bias decomposition for rating prediction and why bias terms substantially reduce prediction error
- Implement both basic MF and biased MF in PyTorch using `nn.Embedding`
- Analyze convergence properties and the effect of embedding dimension

## From Low-Rank Approximation to MF

### The Ideal Case: Complete Matrix

If the rating matrix $R \in \mathbb{R}^{m \times n}$ were fully observed, the best rank-$d$ approximation would be given by the **truncated SVD**:

$$R \approx U_d \Sigma_d V_d^\top$$

where $U_d \in \mathbb{R}^{m \times d}$, $\Sigma_d \in \mathbb{R}^{d \times d}$, $V_d \in \mathbb{R}^{n \times d}$ are the top-$d$ components. By the Eckart–Young–Mirsky theorem, this minimizes:

$$\min_{\text{rank}(X) = d} \|R - X\|_F^2$$

We could define $P = U_d \Sigma_d^{1/2}$ and $Q = V_d \Sigma_d^{1/2}$ so that $R \approx P Q^\top$.

### The Real Case: Sparse Matrix

In practice, most entries of $R$ are missing. Direct SVD fails because:

1. **SVD requires a complete matrix.** Treating missing entries as zeros biases the decomposition toward zero — a movie that a user hasn't rated is not the same as a movie rated 0.
2. **Imputation is circular.** Filling missing values before SVD requires knowing what we're trying to predict.

The solution: optimize **only over observed entries**.

## The MF Objective

### Formulation

Given observed ratings $\Omega = \{(u, i) : R_{ui} \text{ is observed}\}$, learn user embeddings $\mathbf{p}_u \in \mathbb{R}^d$ and item embeddings $\mathbf{q}_i \in \mathbb{R}^d$ by minimizing:

$$\mathcal{L}(\theta) = \sum_{(u,i) \in \Omega} \bigl(R_{ui} - \mathbf{p}_u^\top \mathbf{q}_i\bigr)^2$$

This is a **non-convex** optimization problem (bilinear in $P$ and $Q$), but gradient-based methods work well in practice.

### Gradient Derivation

For a single observed rating $(u, i)$, define the prediction error:

$$e_{ui} = R_{ui} - \mathbf{p}_u^\top \mathbf{q}_i$$

The loss for this pair is $\ell_{ui} = e_{ui}^2$. The gradients are:

$$\frac{\partial \ell_{ui}}{\partial \mathbf{p}_u} = -2\, e_{ui}\, \mathbf{q}_i$$

$$\frac{\partial \ell_{ui}}{\partial \mathbf{q}_i} = -2\, e_{ui}\, \mathbf{p}_u$$

**Derivation:** By the chain rule:

$$\frac{\partial \ell_{ui}}{\partial \mathbf{p}_u} = 2(R_{ui} - \mathbf{p}_u^\top \mathbf{q}_i) \cdot \frac{\partial}{\partial \mathbf{p}_u}\bigl(-\mathbf{p}_u^\top \mathbf{q}_i\bigr) = -2\, e_{ui}\, \mathbf{q}_i$$

since $\frac{\partial}{\partial \mathbf{p}_u}(\mathbf{p}_u^\top \mathbf{q}_i) = \mathbf{q}_i$ (treating $\mathbf{q}_i$ as constant with respect to $\mathbf{p}_u$).

The SGD update rules are:

$$\mathbf{p}_u \leftarrow \mathbf{p}_u + \eta \cdot e_{ui} \cdot \mathbf{q}_i$$

$$\mathbf{q}_i \leftarrow \mathbf{q}_i + \eta \cdot e_{ui} \cdot \mathbf{p}_u$$

where $\eta$ is the learning rate.

### With L2 Regularization

To prevent overfitting (especially with sparse data), add weight decay:

$$\mathcal{L}_{\text{reg}}(\theta) = \sum_{(u,i) \in \Omega} \bigl(R_{ui} - \mathbf{p}_u^\top \mathbf{q}_i\bigr)^2 + \lambda \Bigl(\sum_u \|\mathbf{p}_u\|^2 + \sum_i \|\mathbf{q}_i\|^2\Bigr)$$

The regularized gradients become:

$$\frac{\partial \mathcal{L}_{\text{reg}}}{\partial \mathbf{p}_u} = -2\, e_{ui}\, \mathbf{q}_i + 2\lambda\, \mathbf{p}_u$$

$$\frac{\partial \mathcal{L}_{\text{reg}}}{\partial \mathbf{q}_i} = -2\, e_{ui}\, \mathbf{p}_u + 2\lambda\, \mathbf{q}_i$$

!!! note "Weight Decay in Adam"
    In PyTorch, passing `weight_decay=wd` to `torch.optim.Adam` applies L2 regularization. Note that Adam's weight decay interacts with the adaptive learning rate — for decoupled weight decay, use `AdamW` (see Chapter 2).

## Embedding Interpretation

The matrix $P \in \mathbb{R}^{m \times d}$ can be viewed as an **embedding table**: row $u$ is the $d$-dimensional representation of user $u$. Similarly, $Q \in \mathbb{R}^{n \times d}$ is the item embedding table.

In PyTorch, `nn.Embedding(num_users, d)` stores exactly this table. The forward pass for user $u$ and item $i$:

```python
p_u = self.user_emb(u)   # shape: (batch, d) — lookup row u from P
q_i = self.item_emb(i)   # shape: (batch, d) — lookup row i from Q
rating = (p_u * q_i).sum(1)  # element-wise product, then sum → dot product
```

**Why element-wise product + sum instead of `torch.dot`?** The inputs are batched: `p_u` and `q_i` have shape `(batch_size, d)`. Element-wise multiplication followed by `sum(1)` computes the dot product **independently for each sample** in the batch, yielding a tensor of shape `(batch_size,)`.

!!! warning "Embedding Initialization Matters"
    The implementation initializes embeddings with `uniform_(0, 0.05)`. This ensures small, positive initial values. Other common choices include Xavier initialization or sampling from $\mathcal{N}(0, 0.01)$. Poor initialization (e.g., large values) can cause the optimization to diverge.

## PyTorch Implementation: Basic MF

```python
class MF(nn.Module):
    """
    Matrix Factorization for Collaborative Filtering.
    
    Predicts rating as: r_hat = p_u^T q_i
    where p_u and q_i are learned embedding vectors.
    """
    def __init__(self, num_users, num_items, emb_size=100):
        super(MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        # Small positive initialization
        self.user_emb.weight.data.uniform_(0, 0.05)
        self.item_emb.weight.data.uniform_(0, 0.05)

    def forward(self, u, v):
        u = self.user_emb(u)          # (batch, emb_size)
        v = self.item_emb(v)          # (batch, emb_size)
        return (u * v).sum(1)         # (batch,) — dot product per sample
```

### Walkthrough

1. **`__init__`**: Creates two embedding tables of shape `(num_users, emb_size)` and `(num_items, emb_size)`.
2. **`forward(u, v)`**: Takes integer tensors `u` (user IDs) and `v` (item IDs), looks up embeddings, and returns the dot product as the predicted rating.
3. **Parameter count**: $(m + n) \times d$. For $m = 610$, $n = 9{,}724$, $d = 100$: approximately 1.03M parameters.

## Connection to SVD

Matrix factorization and SVD are closely related but not identical:

| Aspect | Truncated SVD | Learned MF |
|--------|--------------|-----------|
| **Objective** | $\min \|R - PQ^\top\|_F^2$ (all entries) | $\min \sum_{(u,i) \in \Omega}(R_{ui} - \mathbf{p}_u^\top \mathbf{q}_i)^2$ (observed only) |
| **Solution** | Closed-form (Eckart–Young) | Iterative (gradient descent) |
| **Orthogonality** | $P^\top P = I$, $Q^\top Q = I$ | No constraints |
| **Missing data** | Cannot handle natively | Handles naturally |
| **Uniqueness** | Unique (up to sign) | Multiple optima |

The learned MF is sometimes called "Funk SVD" (after Simon Funk's Netflix Prize approach), though it is not a true SVD.

## Effect of Embedding Dimension

The embedding dimension $d$ controls model capacity:

- **$d$ too small**: Underfitting. The model cannot capture the latent structure.
- **$d$ too large**: Overfitting. With sparse data, the model memorizes noise.

**Rule of thumb**: Start with $d \in \{50, 100, 200\}$ and tune via validation loss. For the MovieLens-Small dataset ($\sim$100K ratings), $d = 100$ is a reasonable default.

The number of effective latent factors in the data can be estimated by examining the singular value spectrum of the (imputed) rating matrix: if the singular values decay rapidly, a small $d$ suffices.

## Limitations of Basic MF

The basic MF model has an important limitation: it models each rating as purely a user–item interaction, ignoring systematic effects:

- **User bias**: Some users rate everything highly; others are harsh critics.
- **Item bias**: Some movies are universally loved; others are niche.
- **Global mean**: The average rating across all observations.

For example, if the global mean rating is 3.5, user $u$ tends to rate 0.5 above average, and movie $i$ is 0.3 below average, the baseline prediction should be $3.5 + 0.5 - 0.3 = 3.7$ — before any latent factor interaction.

---

## Matrix Factorization with Bias

### Motivation: Systematic Effects in Ratings

Consider the MovieLens dataset. Some observable patterns have nothing to do with user–item compatibility:

- **User tendency**: User A rates movies 4.2 on average, while User B rates 2.8 on average. This reflects personality, not preference for specific movies.
- **Item quality**: "The Shawshank Redemption" has a 4.5 average rating across all users, while a low-budget film averages 2.1. This reflects quality consensus, not individual taste.
- **Global baseline**: The overall average rating in the dataset might be 3.5.

The basic MF model $\hat{R}_{ui} = \mathbf{p}_u^\top \mathbf{q}_i$ forces the latent factors to capture **all** variation, including these systematic effects. This wastes model capacity and makes the latent factors harder to interpret.

### The Bias Decomposition

Following Koren, Bell, and Volinsky (2009), we decompose each rating into:

$$R_{ui} = \underbrace{\mu}_{\text{global mean}} + \underbrace{b_u}_{\text{user bias}} + \underbrace{b_i}_{\text{item bias}} + \underbrace{\mathbf{p}_u^\top \mathbf{q}_i}_{\text{interaction}} + \underbrace{\epsilon_{ui}}_{\text{noise}}$$

The predicted rating becomes:

$$\hat{R}_{ui} = \mu + b_u + b_i + \mathbf{p}_u^\top \mathbf{q}_i$$

where:

- $\mu = \frac{1}{|\Omega|}\sum_{(u,i) \in \Omega} R_{ui}$ is the global mean (can be precomputed)
- $b_u \in \mathbb{R}$ is the bias for user $u$ (learned)
- $b_i \in \mathbb{R}$ is the bias for item $i$ (learned)
- $\mathbf{p}_u^\top \mathbf{q}_i$ captures the residual user–item interaction

**Interpretation**: The baseline estimate $\mu + b_u + b_i$ captures "expected rating based on who the user is and what the item is." The interaction term $\mathbf{p}_u^\top \mathbf{q}_i$ captures "how much this particular user–item pair deviates from the baseline."

### Optimal Bias Values (Closed Form)

If we fix the latent factors and optimize only the biases, the optimal values minimize:

$$\min_{b_u, b_i} \sum_{(u,i) \in \Omega} \bigl(R_{ui} - \mu - b_u - b_i - \mathbf{p}_u^\top \mathbf{q}_i\bigr)^2 + \lambda\bigl(\sum_u b_u^2 + \sum_i b_i^2\bigr)$$

Taking the derivative with respect to $b_u$ and setting to zero:

$$\frac{\partial}{\partial b_u}: \quad -2 \sum_{i:(u,i) \in \Omega} \bigl(R_{ui} - \mu - b_u - b_i - \mathbf{p}_u^\top \mathbf{q}_i\bigr) + 2\lambda b_u = 0$$

Solving:

$$b_u^* = \frac{\sum_{i:(u,i) \in \Omega} (R_{ui} - \mu - b_i - \mathbf{p}_u^\top \mathbf{q}_i)}{|I_u| + \lambda}$$

where $|I_u|$ is the number of items rated by user $u$. Similarly:

$$b_i^* = \frac{\sum_{u:(u,i) \in \Omega} (R_{ui} - \mu - b_u - \mathbf{p}_u^\top \mathbf{q}_i)}{|U_i| + \lambda}$$

In practice, we don't use these closed-form solutions — instead, we learn all parameters jointly with gradient descent. But the closed forms provide insight: the regularization term $\lambda$ in the denominator **shrinks** the bias toward zero, with stronger shrinkage for users/items with few ratings.

### Objective Function with Biases

The regularized loss for biased MF is:

$$\mathcal{L} = \sum_{(u,i) \in \Omega} \bigl(R_{ui} - \mu - b_u - b_i - \mathbf{p}_u^\top \mathbf{q}_i\bigr)^2 + \lambda\Bigl(\sum_u \|\mathbf{p}_u\|^2 + \sum_i \|\mathbf{q}_i\|^2 + \sum_u b_u^2 + \sum_i b_i^2\Bigr)$$

Define the residual $e_{ui} = R_{ui} - \mu - b_u - b_i - \mathbf{p}_u^\top \mathbf{q}_i$. The gradients for a single observation $(u, i)$:

$$\frac{\partial \mathcal{L}}{\partial b_u} = -2\, e_{ui} + 2\lambda\, b_u$$

$$\frac{\partial \mathcal{L}}{\partial b_i} = -2\, e_{ui} + 2\lambda\, b_i$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{p}_u} = -2\, e_{ui}\, \mathbf{q}_i + 2\lambda\, \mathbf{p}_u$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{q}_i} = -2\, e_{ui}\, \mathbf{p}_u + 2\lambda\, \mathbf{q}_i$$

Note that the bias gradients are scalar, while the embedding gradients are $d$-dimensional vectors.

## PyTorch Implementation: MF with Bias

```python
class MF_bias(nn.Module):
    """
    Matrix Factorization with user and item biases.
    
    Predicts rating as: r_hat = p_u^T q_i + b_u + b_i
    
    Note: The global mean mu can be absorbed into the bias terms
    during training (PyTorch optimizes b_u and b_i to include it),
    or added explicitly as a fixed constant.
    """
    def __init__(self, num_users, num_items, emb_size=100):
        super(MF_bias, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.item_bias = nn.Embedding(num_items, 1)
        # Latent factors: small positive values
        self.user_emb.weight.data.uniform_(0, 0.05)
        self.item_emb.weight.data.uniform_(0, 0.05)
        # Biases: small values centered at zero
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

    def forward(self, u, v):
        U = self.user_emb(u)                # (batch, emb_size)
        V = self.item_emb(v)                # (batch, emb_size)
        b_u = self.user_bias(u).squeeze()   # (batch,)
        b_v = self.item_bias(v).squeeze()   # (batch,)
        return (U * V).sum(1) + b_u + b_v   # (batch,)
```

### Implementation Details

**Bias as `nn.Embedding(n, 1)`**: Each bias is a single scalar per user/item. Using `nn.Embedding` with output dimension 1 stores these scalars in a lookup table, just like the latent factors. The `.squeeze()` call removes the trailing dimension: `(batch, 1)` → `(batch,)`.

**Why not `nn.Parameter`?** You *could* use `self.user_bias = nn.Parameter(torch.zeros(num_users))` and index manually. The `nn.Embedding` approach is cleaner because it handles batched lookups automatically and integrates with the optimizer's weight decay.

**Parameter count**: $(m + n) \times d + (m + n)$. The bias terms add only $m + n$ parameters — negligible compared to the embeddings.

!!! note "Where Is the Global Mean $\mu$?"
    The implementation above does not explicitly include $\mu$. During training, the bias terms $b_u$ and $b_i$ absorb the global mean. If you want to make $\mu$ explicit, precompute it and add it in the forward pass:
    ```python
    return (U * V).sum(1) + b_u + b_v + self.global_mean
    ```
    This can improve convergence since the biases start closer to their optimal values.

### Initialization Asymmetry

Latent factors are initialized in $[0, 0.05]$ (positive), while biases are initialized in $[-0.01, 0.01]$ (centered). This reflects their different roles:

- **Latent factors**: Positive initialization ensures initial dot products are non-negative, which is reasonable for positive rating scales.
- **Biases**: Centered initialization reflects the prior that most users/items have near-average tendencies.

## Why Biases Help: A Quantitative Analysis

Consider a user who rates everything 1 point above average. Without bias:

- The latent factors must encode this tendency, reducing capacity for capturing interactions.
- The model needs higher $d$ to achieve the same accuracy.

With bias:

- $b_u \approx 1.0$ captures the tendency directly.
- The latent factors focus purely on user–item compatibility.
- Lower $d$ suffices, improving generalization.

**Empirical impact**: On MovieLens datasets, adding biases typically reduces validation MSE by 10–20%, with the most significant gains on users/items with extreme average ratings.

## Comparison of MF Variants

| Model | Prediction | Parameters | Captures |
|-------|-----------|-----------|----------|
| Basic MF | $\mathbf{p}_u^\top \mathbf{q}_i$ | $(m+n)d$ | Interactions only |
| MF + Bias | $b_u + b_i + \mathbf{p}_u^\top \mathbf{q}_i$ | $(m+n)(d+1)$ | Biases + interactions |
| MF + Bias + $\mu$ | $\mu + b_u + b_i + \mathbf{p}_u^\top \mathbf{q}_i$ | $(m+n)(d+1) + 1$ | Global + biases + interactions |

## Training Considerations

### Learning Rate Schedule

The source code uses a **staged learning rate** strategy:

```python
train_epochs(model, df_train, df_val, epochs=10, lr=0.1)   # Stage 1: coarse
train_epochs(model, df_train, df_val, epochs=15, lr=0.01)  # Stage 2: refine
train_epochs(model, df_train, df_val, epochs=15, lr=0.001) # Stage 3: polish
```

This is a manual approximation of learning rate decay. The initial high learning rate quickly finds a good region; the lower rates fine-tune within that region.

### Full-Batch vs Mini-Batch Training

The implementation uses **full-batch** gradient descent (all training data per step). This is feasible for MovieLens-Small (~80K training ratings) but should be replaced with mini-batch training for larger datasets:

```python
from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(
    torch.LongTensor(df_train.userId.values),
    torch.LongTensor(df_train.movieId.values),
    torch.FloatTensor(df_train.rating.values)
)
loader = DataLoader(dataset, batch_size=1024, shuffle=True)
```

## Complete Training Pipeline

The following shows the complete data loading, encoding, and training pipeline:

```python
def train_epochs(model, df_train, df_val, epochs=10, lr=0.01, wd=0.0):
    """Train the model for a specified number of epochs."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    for epoch in range(epochs):
        model.train()
        users = torch.LongTensor(df_train.userId.values)
        items = torch.LongTensor(df_train.movieId.values)
        ratings = torch.FloatTensor(df_train.rating.values)

        y_hat = model(users, items)
        loss = F.mse_loss(y_hat, ratings)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Periodic validation
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_users = torch.LongTensor(df_val.userId.values)
                val_items = torch.LongTensor(df_val.movieId.values)
                val_ratings = torch.FloatTensor(df_val.rating.values)
                val_hat = model(val_users, val_items)
                val_loss = F.mse_loss(val_hat, val_ratings).item()
            print(f"  Epoch {epoch+1:3d} | "
                  f"Train MSE: {loss.item():.4f} | Val MSE: {val_loss:.4f}")
```

!!! tip "ID Encoding"
    MovieLens user IDs and movie IDs are not contiguous (e.g., user IDs might jump from 3 to 7). Before using `nn.Embedding`, remap IDs to contiguous integers in $[0, N)$. The source code handles this with `proc_col` and `encode_data` functions. Validation data must be encoded with the **same** mapping as training data, and entries with unseen users/items should be dropped.

## Summary

Matrix factorization learns low-dimensional embeddings for users and items such that their dot product approximates the observed ratings. It is optimized only over observed entries (unlike SVD), uses gradient descent (enabling mini-batch training), and is naturally expressed using `nn.Embedding` in PyTorch. Adding bias terms separates systematic effects (who the user is, what the item is) from interaction effects (how compatible this user–item pair is), meaningfully improving prediction quality with negligible parameter increase. The next section replaces the dot product with a neural network for greater expressiveness.

---

## Exercises

1. **Gradient verification**: Implement the MF model and verify the analytical gradients against PyTorch's autograd using `torch.autograd.gradcheck`.

2. **Rank analysis**: For a $5 \times 5$ rating matrix with rank 2, show that MF with $d = 2$ can perfectly reconstruct all entries. What happens with $d = 1$?

3. **Initialization experiment**: Train the MF model with three initialization schemes: `uniform_(0, 0.05)`, `uniform_(-1, 1)`, and `normal_(0, 1)`. Compare convergence speed and final validation loss.

4. **Embedding dimension sweep**: Train MF with $d \in \{10, 50, 100, 200, 500\}$ on MovieLens-Small. Plot training and validation MSE vs $d$. At what point does overfitting begin?

5. **Mathematical equivalence**: Prove that if $R$ is fully observed and we minimize $\|R - PQ^\top\|_F^2$ without regularization, the global minimum satisfies $PQ^\top = U_d \Sigma_d V_d^\top$ (the truncated SVD).

6. **Bias recovery**: Create a synthetic dataset where $R_{ui} = 3.0 + b_u + b_i + \epsilon_{ui}$ with known biases and no latent factors. Train MF_bias with $d = 1$. Do the learned biases recover the true values?

7. **Ablation study**: Train three models on MovieLens: (a) MF without bias, (b) bias-only model ($d = 0$, just $\mu + b_u + b_i$), (c) MF with bias. Report validation MSE. How much of the prediction power comes from biases alone?

8. **Regularization sensitivity**: Train MF_bias with $\lambda \in \{0, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\}$ (using `weight_decay` in Adam). Plot validation MSE vs $\lambda$. Is the optimal $\lambda$ the same for biases and embeddings?

9. **Interpretation**: After training, find the users with the highest and lowest learned biases $b_u$. What do their rating histories look like? Do the biases match intuition?
