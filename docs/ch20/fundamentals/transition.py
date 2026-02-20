"""
Transition Function
"""

import numpy as np
from copy import deepcopy


def main():
    A_original = np.array([[6,7,12,5],[5,3,11,18],[7,17,3,3],[8,10,14,9]])
    A = np.full( ( A_original.shape[0] + 1, A_original.shape[1] + 1 ), np.Infinity )   
    A[1:,1:] = A_original
    
    T = deepcopy(A) # record shortest time to travel from top left
    D = np.zeros_like(A, dtype=object) # record shortest path direction "<" or "^"
    
    n, m = A.shape
    for i in range(1,n):
        for j in range(1,m):
            if not ( (i==1) and (j==1) ):
                if T[i,j-1] < T[i-1,j]:
                    T[i,j] = T[i,j-1] + A[i,j]
                    D[i,j] = "<" 
                else:
                    T[i,j] = T[i-1,j] + A[i,j]
                    D[i,j] = "^" 
                    
    print(T[1:,1:])
    print(D[1:,1:])
    
    def print_path(i, j):
        if D[i,j] == 0:
            print(f"({i-1},{j-1})--->", end="")
            return    
        elif D[i,j] == "<":
            print_path(i, j-1)
        else:
            print_path(i-1, j)
        print(f"({i-1},{j-1})--->", end="")
            
    print_path(4,4)
            
        
if __name__ == "__main__":
    main()
