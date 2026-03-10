# Proof Techniques

Proof techniques provide the logical tools for establishing correctness of algorithms, convergence of optimization methods, and bounds on generalization error. Every theoretical result in deep learning rests on one or more of these techniques.

## Definition

A mathematical proof is a logically rigorous argument that a statement is true, starting from axioms and previously established results. The main proof strategies are:

$$
\begin{array}{ll}
\text{Direct proof} & \text{Assume premises, derive conclusion} \\
\text{Contradiction} & \text{Assume negation, derive impossibility} \\
\text{Induction} & \text{Base case + inductive step} \\
\text{Construction} & \text{Exhibit a concrete example} \\
\text{Contrapositive} & \text{Prove } \neg Q \Rightarrow \neg P \text{ instead of } P \Rightarrow Q
\end{array}
$$

## Explanation

Each technique suits different situations:

- **Direct proof**: The default approach. Given hypotheses, apply definitions and known results to reach the conclusion. Example: proving that the gradient of MSE loss is a linear function of the residual.
- **Contradiction**: Useful for impossibility results. Assume the claim is false and derive a logical conflict. Example: proving that no deterministic algorithm can minimize a non-convex function in polynomial time under certain complexity assumptions.
- **Induction**: Essential for statements indexed by natural numbers. Example: proving that backpropagation correctly computes gradients through $L$ layers by induction on $L$.
- **Construction**: Proves existence by building an explicit example. Example: constructing a neural network that approximates any continuous function (universal approximation theorem proofs).
- **Contrapositive**: Instead of proving $P \Rightarrow Q$, prove the logically equivalent $\neg Q \Rightarrow \neg P$. Useful when the negation of the conclusion gives a stronger starting point.

## Examples

```python
import torch

# Direct proof verification: gradient of L2 loss is 2*(pred - target)/n
torch.manual_seed(0)
pred = torch.randn(5, requires_grad=True)
target = torch.randn(5)

loss = ((pred - target) ** 2).mean()
loss.backward()

# Analytical gradient
analytical_grad = 2 * (pred.detach() - target) / pred.numel()
print(f"Autograd gradient:    {pred.grad.tolist()}")
print(f"Analytical gradient:  {analytical_grad.tolist()}")
print(f"Match: {torch.allclose(pred.grad, analytical_grad)}")

# Constructive proof: build a network that maps input to output
# Construct weights that implement f(x) = 2x + 1
W = torch.tensor([[2.0]])
b = torch.tensor([1.0])
x = torch.tensor([3.0])
y = W @ x + b
print(f"\nConstructed linear map: f({x.item()}) = {y.item()}")
```
