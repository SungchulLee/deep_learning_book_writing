"""
Matrix Chain - Memoization
"""

import numpy as np
 

def MatrixChainMultiplication(p, i, j, memo):
    """
    We are interested in the matrix multiplication $A[i]A[i+1]...A[j]$, not $A[1]A[2]...A[n]$
    Matrix A[k] has dimension p[k-1] x p[k]
    p = [p[0], p[1], ..., p[n]]
    """
    # Base Case
    if i == j:
        return 0 
    if memo[i, j] != -1 :
        return memo[i, j]
    
    # Recursive Case
    memo[i, j] = np.infty 
    for k in range(i, j):
        memo[i, j] = min( memo[i, j],
                         MatrixChainMultiplication(p, i, k, memo) 
                         + MatrixChainMultiplication(p, k + 1, j, memo)
                         + p[i - 1] * p[k] * p[j] )  
    return memo[i,j]


def main():
    p = [1, 2, 3, 4, 3] # four matrix multiplications
    n = len(p) # 5
    memo = np.ones((n,n)) * (-1)
    print(f"Minimum number of multiplications : {MatrixChainMultiplication(p, 1, n-1, memo)}")

    
if __name__ == "__main__":
    main()
