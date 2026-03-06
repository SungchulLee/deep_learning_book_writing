# Knapsack FPTAS

PTAS: for any $\epsilon > 0$, achieves $(1+\epsilon)$-approximation in poly time. FPTAS: poly in $n$ and $1/\epsilon$.

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

[Approximation Algorithms (Vazirani)](https://www.springer.com/gp/book/9783540653677)
