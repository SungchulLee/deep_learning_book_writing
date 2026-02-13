# Cross-Entropy Loss

Cross-entropy loss is the natural cost function for classification, arising from maximum likelihood estimation under a categorical distribution. This section derives cross-entropy from the softmax model, establishes the information-theoretic interpretation, covers the three PyTorch interfaces, and demonstrates the training pipeline through an N-gram language model.

!!! note "See Also"
    This section provides the full derivation of cross-entropy as the MLE-optimal loss for softmax classification. For cross-entropy in the broader context of loss function selection—alongside MSE, focal loss, hinge loss, and others—see **Section 3.5 Loss Functions**.

## Data and Model

Given labeled observations $\{(x^{(i)}, y^{(i)}): i=1,\ldots,m\}$ where each $y^{(i)}$ is a one-hot encoded vector over $K$ classes, the softmax model computes class probabilities:

$$p^{(i)} = \operatorname{softmax}(x^{(i)}W + b)$$

Explicitly, for class $k$:

$$p^{(i)}[k] = \frac{\exp\!\left((x^{(i)}W + b)_k\right)}{\sum_{j=0}^{K-1} \exp\!\left((x^{(i)}W + b)_j\right)}$$

The softmax function ensures that $p^{(i)}[k] \geq 0$ for all $k$ and $\sum_{k=0}^{K-1} p^{(i)}[k] = 1$, so the output forms a valid probability distribution over classes.

## Likelihood Function

Each observation $y^{(i)}$ is modeled as a draw from a categorical distribution with probabilities $p^{(i)}$. The probability of observing label $y^{(i)}$ given parameters $(W, b)$ is:

$$P(y^{(i)} \mid x^{(i)};\, W, b) = \prod_{k=0}^{K-1} \left(p^{(i)}[k]\right)^{y^{(i)}[k]}$$

Since $y^{(i)}$ is one-hot, only the term corresponding to the true class $c_i$ survives: the expression simplifies to $p^{(i)}[c_i]$.

Assuming independence across samples, the full likelihood is:

$$L(W,b) = \prod_{i=1}^m \prod_{k=0}^{K-1} \left(p^{(i)}[k]\right)^{y^{(i)}[k]}$$

## Log-Likelihood Function

Taking the logarithm:

$$\ell(W,b) = \sum_{i=1}^m \sum_{k=0}^{K-1} y^{(i)}[k] \log p^{(i)}[k]$$

Since $y^{(i)}$ is one-hot with $y^{(i)}[c_i] = 1$, this reduces to:

$$\ell(W,b) = \sum_{i=1}^m \log p^{(i)}[c_i]$$

The log-likelihood is simply the sum of log-probabilities assigned to the correct classes.

## Cost Function

Negating and averaging the log-likelihood yields the cross-entropy cost:

$$J(W,b) = -\frac{1}{m}\sum_{i=1}^m \sum_{k=0}^{K-1} y^{(i)}[k] \log p^{(i)}[k]$$

Or equivalently using class indices:

$$J(W,b) = -\frac{1}{m}\sum_{i=1}^m \log p^{(i)}[c_i]$$

This is the **negative log-likelihood** (NLL) of the data under the softmax model, scaled by $\frac{1}{m}$.

## Maximum Likelihood Principle

The equivalence chain holds:

$$\underset{W,b}{\operatorname{argmax}}\ L \quad\Leftrightarrow\quad \underset{W,b}{\operatorname{argmax}}\ \ell \quad\Leftrightarrow\quad \underset{W,b}{\operatorname{argmin}}\ J$$

Minimizing cross-entropy loss is equivalent to finding the parameters that maximize the probability of the observed labels under the softmax model.

## Gradient of Cross-Entropy with Softmax

A remarkable simplification occurs when computing the gradient of the cross-entropy loss with respect to the logits $z^{(i)} = x^{(i)}W + b$. For a single sample with true class $c$:

$$\frac{\partial J}{\partial z_k} = p_k - y_k$$

where $y_k = \mathbb{1}[k = c]$. The gradient at each logit is simply the difference between the predicted probability and the target. This clean form is one reason the softmax–cross-entropy combination is so widely used: the gradient computation is both simple and numerically stable.

### Derivation

For a single sample, the loss is $J = -\log p_c$ where $p_c = e^{z_c} / \sum_j e^{z_j}$. Consider two cases.

**When $k = c$ (the true class):**

$$\frac{\partial J}{\partial z_c} = -\frac{\partial}{\partial z_c}\!\left(z_c - \log\sum_j e^{z_j}\right) = -1 + \frac{e^{z_c}}{\sum_j e^{z_j}} = p_c - 1 = p_c - y_c$$

**When $k \neq c$:**

$$\frac{\partial J}{\partial z_k} = -\frac{\partial}{\partial z_k}\!\left(z_c - \log\sum_j e^{z_j}\right) = \frac{e^{z_k}}{\sum_j e^{z_j}} = p_k = p_k - y_k$$

Both cases yield $p_k - y_k$, giving the unified gradient formula.

## Information-Theoretic Interpretation

Cross-entropy has a natural interpretation from information theory. For discrete distributions $P$ (true) and $Q$ (predicted):

$$H(P, Q) = -\sum_k P(k) \log Q(k)$$

This decomposes as:

$$H(P, Q) = H(P) + D_{\text{KL}}(P \| Q)$$

where $H(P) = -\sum_k P(k) \log P(k)$ is the entropy of the true distribution and $D_{\text{KL}}(P \| Q)$ is the KL divergence. Since $H(P)$ is fixed for a given dataset, minimizing cross-entropy is equivalent to minimizing KL divergence between the true and predicted distributions.

## Binary Cross-Entropy as Special Case

For $K = 2$ classes with a single output probability $p$, the cross-entropy cost reduces to the binary cross-entropy:

$$J = -\frac{1}{m}\sum_{i=1}^m \left[y^{(i)} \log p^{(i)} + (1 - y^{(i)}) \log(1 - p^{(i)})\right]$$

This is the loss used in logistic regression and binary classification. The connection is exact: `nn.BCELoss` implements this formula, while `nn.CrossEntropyLoss` implements the general multi-class version. See [Binary Cross-Entropy](bce.md) for the dedicated treatment.

## PyTorch Interfaces

PyTorch provides three routes to compute cross-entropy, differing in what preprocessing they expect.

### `nn.CrossEntropyLoss` (Recommended)

This is the most common choice. It accepts **raw logits** (pre-softmax scores) and **integer class indices**:

```python
import torch
import torch.nn as nn

logits = torch.tensor([[2.0, 0.5, 0.1],
                        [0.1, 2.5, 0.3]])  # (batch, K)
targets = torch.tensor([0, 1])              # class indices

criterion = nn.CrossEntropyLoss()
loss = criterion(logits, targets)
```

Internally, `CrossEntropyLoss` applies `log_softmax` followed by negative log-likelihood in a single numerically stable operation. This is equivalent to:

```python
import torch.nn.functional as F

log_probs = F.log_softmax(logits, dim=1)
loss_manual = F.nll_loss(log_probs, targets)
# loss_manual ≈ loss
```

!!! warning "Common Mistake"
    Never apply `softmax` or `log_softmax` before `nn.CrossEntropyLoss`. The function handles this internally; applying it twice produces incorrect gradients.

### `F.cross_entropy`

The functional equivalent:

```python
loss = F.cross_entropy(logits, targets)
```

Identical behaviour, but as a stateless function rather than a module.

### `nn.NLLLoss`

Negative Log-Likelihood Loss expects **log-probabilities** (output of `log_softmax`), not raw logits:

```python
log_probs = F.log_softmax(logits, dim=1)
nll_criterion = nn.NLLLoss()
loss = nll_criterion(log_probs, targets)
```

This two-step approach is useful when you need access to the log-probabilities for other purposes (e.g., beam search in sequence models).

### Interface Comparison

| Interface | Input | Internal Operation | Use Case |
|-----------|-------|--------------------|----------|
| `nn.CrossEntropyLoss` | Raw logits | log_softmax + NLL | Standard classification |
| `F.cross_entropy` | Raw logits | log_softmax + NLL | Functional style |
| `nn.NLLLoss` | Log-probabilities | NLL only | When log-probs are needed elsewhere |

All three produce identical loss values when used correctly.

## N-gram Language Model Example

Cross-entropy loss is the standard training objective for language models. The following example trains an N-gram neural language model on Shakespeare's Sonnet 2 using `nn.CrossEntropyLoss`.

### Data Preparation

```python
import torch

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

text = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,...""".split()

vocab = set(text)
word_to_ix = {word: i for i, word in enumerate(vocab)}

# Build n-gram context-target pairs
ngrams = [
    ([text[i - j - 1] for j in range(CONTEXT_SIZE)], text[i])
    for i in range(CONTEXT_SIZE, len(text))
]
```

### Model Architecture

```python
import torch.nn as nn
import torch.nn.functional as F

class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view(1, -1)
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)      # raw logits — no softmax!
        return out
```

### Training Loop

```python
import torch.optim as optim

model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

losses = []
for epoch in range(1000):
    total_loss = 0
    for context, target in ngrams:
        context_idxs = torch.tensor(
            [word_to_ix[w] for w in context], dtype=torch.long
        )

        model.zero_grad()
        logits = model(context_idxs)

        loss = loss_function(
            logits,
            torch.tensor([word_to_ix[target]], dtype=torch.long)
        )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    losses.append(total_loss)
```

## Numerical Stability

The reason PyTorch fuses `log_softmax` with NLL rather than computing `softmax` followed by `log` is numerical stability. Consider:

$$\log\text{softmax}(z_k) = \log\frac{e^{z_k}}{\sum_j e^{z_j}} = z_k - \log\sum_j e^{z_j}$$

Computing this directly avoids the intermediate step of exponentiating large logits (which can overflow to `inf`) and then taking their log (which can underflow to `-inf`). PyTorch's `log_softmax` uses the **log-sum-exp trick**:

$$\log\sum_j e^{z_j} = c + \log\sum_j e^{z_j - c}, \qquad c = \max_j z_j$$

Subtracting the maximum ensures the largest exponent is $e^0 = 1$, preventing overflow. This is why `nn.CrossEntropyLoss` (which fuses both operations) is more numerically stable than manually computing `softmax` followed by `log`.

## Key Takeaways

Cross-entropy loss is the negative log-likelihood under a categorical distribution, making it the principled loss for classification via the maximum likelihood principle. The softmax activation ensures valid probability outputs, and its combination with cross-entropy produces a gradient of the simple form $p - y$ (predicted minus target). From an information-theoretic perspective, minimizing cross-entropy minimizes the KL divergence between the true label distribution and the model's predicted distribution. PyTorch provides three equivalent interfaces (`nn.CrossEntropyLoss`, `F.cross_entropy`, `nn.NLLLoss`) that differ only in what preprocessing they expect—always feed raw logits to `CrossEntropyLoss` and `F.cross_entropy`, and the fused implementation provides numerical stability that separate `softmax` + `log` cannot match.
