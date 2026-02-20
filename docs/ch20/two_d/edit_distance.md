# Edit Distance

**Edit Distance** is an important concept in algorithm design and analysis.

$$dp[i][j] = \begin{cases} dp[i-1][j-1] & \text{if } s_1[i]=s_2[j] \\ 1+\min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) & \text{otherwise} \end{cases}$$

```python
def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]: dp[i][j] = dp[i-1][j-1]
            else: dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]

s1, s2 = "kitten", "sitting"
print(f"edit_distance({s1},{s2}) = {edit_distance(s1,s2)}")
```

**Output:**
```
edit_distance(kitten,sitting) = 3
```


# Reference

[Introduction to Algorithms (CLRS), Chapter 15](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
