# Algorithm Definition

An algorithm is a finite, well-defined computational procedure that maps inputs to outputs. Neural networks are parameterized algorithms: the forward pass is a fixed procedure, but the learned weights determine the specific input-output mapping.

## Definition

An algorithm is a procedure satisfying five properties:

$$
\text{Algorithm}: \mathcal{X} \rightarrow \mathcal{Y}
$$

1. **Input**: Zero or more quantities are externally supplied
2. **Output**: At least one quantity is produced
3. **Definiteness**: Each instruction is clear and unambiguous
4. **Finiteness**: The procedure terminates after a finite number of steps
5. **Effectiveness**: Every instruction can be carried out in finite time

## Explanation

A neural network's forward pass is an algorithm: given input tensor $\mathbf{x}$, it applies a sequence of matrix multiplications, nonlinearities, and normalizations to produce output $\hat{\mathbf{y}}$. Training is also an algorithm: given a dataset and loss function, gradient descent iteratively updates parameters until a stopping criterion is met.

The distinction between an algorithm and a model is that an algorithm has fixed behavior, while a model has learned behavior. However, once training is complete, inference is a deterministic algorithm (assuming no dropout or stochastic components at test time).

Key algorithmic properties of neural networks:

- **Definiteness**: Every operation (matmul, ReLU, softmax) is precisely defined
- **Finiteness**: A forward pass terminates in $O(L)$ layer computations
- **Effectiveness**: Each operation is a basic tensor operation executable on hardware

## Examples

```python
import torch
import torch.nn as nn

# A neural network forward pass is an algorithm
class SimpleNet(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.layer1 = nn.Linear(d_in, d_hidden)  # step 1: affine transform
        self.relu = nn.ReLU()                      # step 2: nonlinearity
        self.layer2 = nn.Linear(d_hidden, d_out)   # step 3: affine transform

    def forward(self, x):
        # Each step is definite, finite, and effective
        h = self.layer1(x)
        h = self.relu(h)
        return self.layer2(h)

model = SimpleNet(d_in=5, d_hidden=16, d_out=3)
x = torch.randn(4, 5)
y = model(x)
print(f"Input:  {x.shape}")
print(f"Output: {y.shape}")
print(f"Steps:  Linear -> ReLU -> Linear (finite, 3 steps)")
print(f"Deterministic: {torch.equal(model(x), model(x))}")
```
