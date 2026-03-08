# Fractional Knapsack

Fractional knapsack: take items by value-to-weight ratio (greedy works, unlike 0-1 knapsack).

$$dp[i][w] = \max(dp[i-1][w],\; dp[i-1][w-w_i] + v_i)$$

```python
def knapsack(W, weights, values):
    n = len(weights)
    dp = [[0]*(W+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for w in range(W+1):
            dp[i][w] = dp[i-1][w]
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w], dp[i-1][w-weights[i-1]]+values[i-1])
    return dp[n][W]

print(f"Max value: {knapsack(50, [10,20,30], [60,100,120])}")
```

**Output:**
```
Max value: 220
```

# Reference

[Introduction to Algorithms (CLRS), Chapter 16](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
