# Reduction

Reduction transforms one problem into another that we already know how to solve. This is a central idea in deep learning: transfer learning, feature extraction, and fine-tuning all reduce new problems to previously solved ones.

## Definition

A reduction from problem $A$ to problem $B$ is a transformation that converts any instance of $A$ into an instance of $B$, such that solving $B$ yields a solution to $A$:

$$
A \leq_p B
$$

If the transformation runs in polynomial time, then $B$ is at least as hard as $A$.

## Explanation

Reductions eliminate redundant work by reusing existing solutions:

- **Transfer learning**: Reduces a new classification task (e.g., medical imaging) to feature extraction from a pretrained ImageNet model. The pretrained model acts as the "known solver" for feature extraction.
- **Fine-tuning**: Reduces a domain-specific task to a general language modeling task by starting from pretrained weights and adjusting them with domain data.
- **Embedding-based retrieval**: Reduces nearest-neighbor search over complex objects (images, text) to Euclidean nearest-neighbor search in an embedding space.
- **Feature engineering**: Reduces a complex learning task to a simpler one by transforming raw inputs into informative features.

The key question when applying a reduction is whether the transformation preserves enough structure. A reduction that discards essential information about $A$ will produce a poor solution, even if $B$ is solved perfectly.

## Examples

```python
import torch
import torch.nn as nn

# Reduction: use a pretrained feature extractor for a new task
# Instead of training from scratch, we reduce to feature extraction

# Simulated "pretrained" feature extractor (frozen)
torch.manual_seed(42)
feature_extractor = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 32))
for p in feature_extractor.parameters():
    p.requires_grad = False

# New task: classify into 5 classes using extracted features
classifier = nn.Linear(32, 5)

# Generate data
X = torch.randn(100, 20)
y = torch.randint(0, 5, (100,))

# Extract features (the reduction step)
with torch.no_grad():
    features = feature_extractor(X)
print(f"Reduced {X.shape} inputs to {features.shape} features")

# Train only the classifier on the reduced problem
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
for _ in range(100):
    loss = nn.functional.cross_entropy(classifier(features), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(f"Loss after training classifier: {loss.item():.4f}")
```
