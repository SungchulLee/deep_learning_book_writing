"""
Matrix Chain - Tabulation
"""

import numpy as np


# === Functions ===

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

    

# === Main ===
if __name__ == "__main__":
    main()
