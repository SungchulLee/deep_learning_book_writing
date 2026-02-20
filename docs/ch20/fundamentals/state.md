# State Definition


<img src='img/Screen Shot 2022-06-16 at 1.05.37 AM.png' width=50%>


```python
import numpy as np
from copy import deepcopy


def main():
    A_original = np.array([[6,7,12,5],[5,3,11,18],[7,17,3,3],[8,10,14,9]])
    A = np.full( ( A_original.shape[0] + 1, A_original.shape[1] + 1 ), np.Infinity )   
    A[1:,1:] = A_original
    
    M = np.ones_like(A, dtype=np.uint8) * (-1) # memoization
    D = np.zeros_like(A, dtype=object) # record shortest path direction "<" or "^"
    
    def find_shortest(i, j):
        if M[i,j] != -1:
            return M[i,j]
        if (i==1) and (j==1):
            M[i,j] = A[i,j]
            D[i,j] = "^" 
        elif (i==1):
            M[i,j] = find_shortest(i, j-1) + A[i,j]
            D[i,j] = "<" 
        elif (j==1):
            M[i,j] = find_shortest(i-1, j) + A[i,j]
            D[i,j] = "^"  
        elif find_shortest(i-1, j) < find_shortest(i, j-1):
            M[i,j] = find_shortest(i-1, j) + A[i,j]
            D[i,j] = "^" 
        else:
            M[i,j] = find_shortest(i, j-1) + A[i,j]
            D[i,j] = "<" 
        return M[i,j] 
    
    n, m = A.shape
    find_shortest(i=n-1, j=m-1)
                    
    print(M[1:,1:])
    print(D[1:,1:])
    
    def print_path(i, j):
        if (i==1) and (j==1):
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
```

**Output:**
```
[[ 6 13 25 30]
 [11 14 25 43]
 [18 31 28 31]
 [26 36 42 40]]
[['^' '<' '<' '<']
 ['^' '<' '<' '<']
 ['^' '^' '^' '<']
 ['^' '<' '^' '^']]
(0,0)--->(1,0)--->(1,1)--->(1,2)--->(2,2)--->(2,3)--->(3,3)--->
```


# Reference

[[알고리즘] 제18-2강 동적계획법 (Dynamic Programming)](https://www.youtube.com/watch?v=j_sdjivoPs8&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=53)
