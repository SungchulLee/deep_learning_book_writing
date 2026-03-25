# Potential Method

The aggregate method computes total cost and divides by $n$. The accounting method assigns per-operation charges and tracks credit locally. The potential method takes a more global approach: it defines a single function of the entire data structure state, analogous to potential energy in physics. When the data structure moves to a "higher energy" state, the potential increases and stores the difference for later use. When an expensive operation brings the state back to "low energy," the stored potential pays for the cost. This makes the potential method the most powerful and flexible of the three amortized analysis techniques.

## Definition

Let $D_0, D_1, \ldots, D_n$ denote the states of the data structure after operations $0, 1, \ldots, n$, where $D_0$ is the initial state. A **potential function** $\Phi$ maps each state to a real number:

$$
\Phi: \{D_0, D_1, \ldots, D_n\} \to \mathbb{R}
$$

The **amortized cost** of the $i$-th operation is defined as:

$$
\hat{c}_i = c_i + \Phi(D_i) - \Phi(D_{i-1})
$$

where $c_i$ is the actual cost. The change in potential $\Delta\Phi_i = \Phi(D_i) - \Phi(D_{i-1})$ acts as a correction term: if $\Delta\Phi_i > 0$, the operation increases the stored energy (making the amortized cost higher than actual), and if $\Delta\Phi_i < 0$, the operation releases stored energy (making the amortized cost lower than actual).

## Key Theorem

The total amortized cost is an upper bound on the total actual cost, provided $\Phi(D_n) \geq \Phi(D_0)$:

$$
\sum_{i=1}^{n} \hat{c}_i = \sum_{i=1}^{n} c_i + \Phi(D_n) - \Phi(D_0) \geq \sum_{i=1}^{n} c_i
$$

This follows by telescoping: the intermediate potential terms cancel. To guarantee the bound, it suffices to require:

$$
\Phi(D_i) \geq \Phi(D_0) \quad \text{for all } i = 1, 2, \ldots, n
$$

A common convention is to set $\Phi(D_0) = 0$ and require $\Phi(D_i) \geq 0$ for all $i$.

!!! tip "Choosing a Good Potential Function"
    The art of the potential method lies in choosing $\Phi$. A good potential function should:

    - Be 0 (or small) in the initial state
    - Increase during cheap operations (storing energy)
    - Decrease during expensive operations (releasing energy to pay for the cost)
    - Make the amortized cost of every operation type a simple expression, ideally a constant

## Example: Multipop Stack

Define $\Phi(D) = |S|$, the number of elements on the stack.

**PUSH:** Actual cost $c = 1$, potential increases by 1.

$$
\hat{c} = 1 + 1 = 2
$$

**POP:** Actual cost $c = 1$, potential decreases by 1.

$$
\hat{c} = 1 + (-1) = 0
$$

**MULTIPOP(k):** Actual cost $c = k' = \min(k, s)$, potential decreases by $k'$.

$$
\hat{c} = k' + (-k') = 0
$$

The amortized cost of any operation is at most 2, giving $O(1)$ amortized per operation.

## Example: Binary Counter

Let $b_i$ denote the number of 1-bits after the $i$-th increment. Define $\Phi(D_i) = b_i$.

If the $i$-th increment resets $t_i$ bits from 1 to 0 and sets one bit from 0 to 1, then $c_i = t_i + 1$ and $\Delta\Phi_i = 1 - t_i$:

$$
\hat{c}_i = (t_i + 1) + (1 - t_i) = 2
$$

The costly $t_i$ flips are exactly cancelled by the potential decrease, yielding a constant amortized cost.

## Example: Dynamic Array

For a dynamic array with size $s$ and capacity $C$, define:

$$
\Phi(D) = 2s - C
$$

**Append without resize:** $c = 1$, size increases by 1, capacity unchanged.

$$
\hat{c} = 1 + [2(s+1) - C] - [2s - C] = 1 + 2 = 3
$$

**Append with resize** (when $s = C$): The array doubles to capacity $2C$, copies $s$ elements, then inserts. Actual cost $c = s + 1$. After the resize, size is $s + 1$ and capacity is $2s$.

$$
\hat{c} = (s + 1) + [2(s+1) - 2s] - [2s - s] = (s+1) + 2 - s = 3
$$

Both cases give an amortized cost of exactly 3.

## General Framework

The potential method is particularly useful for:

1. **Multi-operation data structures** where different operations interact (e.g., insertions create work that deletions must clean up).
2. **Self-adjusting data structures** like splay trees, where the potential captures the "disorder" of the tree.
3. **Complex state transitions** where per-element credit (accounting method) is hard to track.

The potential method relates to the accounting method through the identity:

$$
\text{credit after operation } i = \Phi(D_i) - \Phi(D_0)
$$

The accounting method's credit invariant (credit $\geq 0$) is equivalent to the potential method's requirement ($\Phi(D_i) \geq \Phi(D_0)$).

## Python Example

```python
"""
Potential method demonstration.

Applies the potential method to a multipop stack and a dynamic array,
verifying that amortized costs match the theoretical predictions.
"""


# ===================================================================
# Multipop Stack with Potential Tracking
# ===================================================================
class MultipopStackPotential:
    """Stack where potential = number of elements."""

    def __init__(self):
        self.items = []
        self.total_actual = 0
        self.total_amortized = 0
        self.num_ops = 0

    def potential(self):
        """Phi(D) = stack size."""
        return len(self.items)

    def push(self, x):
        """Push with potential tracking."""
        actual = 1
        phi_before = self.potential()
        self.items.append(x)
        phi_after = self.potential()
        amortized = actual + (phi_after - phi_before)

        self.total_actual += actual
        self.total_amortized += amortized
        self.num_ops += 1
        return actual, amortized

    def multipop(self, k):
        """Multipop with potential tracking."""
        k_prime = min(k, len(self.items))
        actual = k_prime
        phi_before = self.potential()
        for _ in range(k_prime):
            self.items.pop()
        phi_after = self.potential()
        amortized = actual + (phi_after - phi_before)

        self.total_actual += actual
        self.total_amortized += amortized
        self.num_ops += 1
        return actual, amortized


# ===================================================================
# Dynamic Array with Potential Tracking
# ===================================================================
class DynamicArrayPotential:
    """Dynamic array where potential = 2*size - capacity."""

    def __init__(self):
        self.capacity = 1
        self.size = 0
        self.data = [None] * self.capacity
        self.total_actual = 0
        self.total_amortized = 0
        self.num_ops = 0

    def potential(self):
        """Phi(D) = 2*size - capacity."""
        return 2 * self.size - self.capacity

    def append(self, value):
        """Append with potential tracking."""
        phi_before = self.potential()

        actual = 1
        if self.size == self.capacity:
            actual += self.size  # copying cost
            new_data = [None] * (2 * self.capacity)
            for i in range(self.size):
                new_data[i] = self.data[i]
            self.data = new_data
            self.capacity *= 2

        self.data[self.size] = value
        self.size += 1

        phi_after = self.potential()
        amortized = actual + (phi_after - phi_before)

        self.total_actual += actual
        self.total_amortized += amortized
        self.num_ops += 1
        return actual, amortized


# ===================================================================
# Demonstration
# ===================================================================
if __name__ == "__main__":
    # --- Multipop Stack ---
    print("=== Multipop Stack (Potential Method) ===")
    stack = MultipopStackPotential()
    for i in range(10):
        actual, amortized = stack.push(i)
    print(f"After 10 pushes: Phi={stack.potential()}, "
          f"total_actual={stack.total_actual}, "
          f"total_amortized={stack.total_amortized}")

    actual, amortized = stack.multipop(10)
    print(f"Multipop(10): actual={actual}, amortized={amortized}")
    print(f"Final: Phi={stack.potential()}, "
          f"total_actual={stack.total_actual}, "
          f"total_amortized={stack.total_amortized}")
    print(f"Amortized >= Actual: "
          f"{stack.total_amortized >= stack.total_actual}")

    # --- Dynamic Array ---
    print("\n=== Dynamic Array (Potential Method) ===")
    arr = DynamicArrayPotential()
    print(f"{'Op':>3} {'Actual':>7} {'Amort':>7} {'Phi':>5} "
          f"{'Size':>5} {'Cap':>5}")
    print("-" * 38)
    for i in range(1, 18):
        actual, amortized = arr.append(i)
        print(f"{i:>3} {actual:>7} {amortized:>7} "
              f"{arr.potential():>5} {arr.size:>5} {arr.capacity:>5}")

    print(f"\nTotal actual:    {arr.total_actual}")
    print(f"Total amortized: {arr.total_amortized}")
    print(f"Amortized >= Actual: "
          f"{arr.total_amortized >= arr.total_actual}")
    print(f"Avg amortized/op: "
          f"{arr.total_amortized / arr.num_ops:.2f}")
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 16: Amortized Analysis. MIT Press.
