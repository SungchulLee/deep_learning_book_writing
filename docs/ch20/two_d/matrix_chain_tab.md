# Matrix Chain - Tabulation

$$

\text{shape of $A_i$ : ($p_{i-1},p_i$)}

$$

<img src="img/Screen Shot 2022-06-21 at 1.20.36 AM.png" width=60%>

[[알고리즘] 제18-4강 동적계획법 (Dynamic Programming)](https://www.youtube.com/watch?v=n_3E2-UhLeU&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=55)

<img src="img/Screen Shot 2022-06-21 at 1.24.34 AM.png" width=60%>

[[알고리즘] 제18-4강 동적계획법 (Dynamic Programming)](https://www.youtube.com/watch?v=n_3E2-UhLeU&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=55)

```python
import numpy as np

def MatrixChainMultiplication(p):
    """
    We are interested in the matrix multiplication $A[i]A[i+1]...A[j]$, not $A[1]A[2]...A[n]$
    Matrix A[k] has dimension p[k-1] x p[k]
    p = [p[0], p[1], ..., p[n]]
    """
    n = len(p) - 1
    
    m = np.zeros( (n + 1, n + 1) )
    for off_diagonal in range(1, n):
        for i in range(1, n + 1 - off_diagonal):
            j = i + off_diagonal
            m[i, j] = np.infty
            for k in range(i, j):
                m[i, j] = min( m[i, j], m[i, k] + m[k + 1, j] + p[i - 1] * p[k] * p[j] ) 
    return m[1, -1]

def main():
    p = [1, 2, 3, 4, 3] # four matrix multiplications
    print(f"Minimum number of multiplications : {MatrixChainMultiplication(p)}")

    
if __name__ == "__main__":
    main()
```

**Output:**
```
Minimum number of multiplications : 30.0
```

# Reference

[[알고리즘] 제18-4강 동적계획법 (Dynamic Programming)](https://www.youtube.com/watch?v=n_3E2-UhLeU&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=55)

[Matrix Chain Multiplication | DP-8](https://www.geeksforgeeks.org/matrix-chain-multiplication-dp-8/)
