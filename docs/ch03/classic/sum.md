# Sum


$$
\sum_{k=1}^n k
$$


```python
import numpy as np


def compute_sum_using_recursion(n):
    if n == 0:
        return 0
    return n + compute_sum_using_recursion(n-1)


def compute_sum_using_package(n):
    return np.arange(n+1).sum()


def main():
    n = 10
    for i in range(n):
        result_0 = compute_sum_using_recursion(i)
        result_1 = compute_sum_using_package(i)
        print(f"Computation of sum from 0 to {i} using recursion : {result_0}")
        print(f"Computation of sum from 0 to {i} using package   : {result_1}")
        print()
        
        
if __name__ == "__main__":
    main()
```

**Output:**
```
Computation of sum from 0 to 0 using recursion : 0
Computation of sum from 0 to 0 using package   : 0

Computation of sum from 0 to 1 using recursion : 1
Computation of sum from 0 to 1 using package   : 1

Computation of sum from 0 to 2 using recursion : 3
Computation of sum from 0 to 2 using package   : 3

Computation of sum from 0 to 3 using recursion : 6
Computation of sum from 0 to 3 using package   : 6

Computation of sum from 0 to 4 using recursion : 10
Computation of sum from 0 to 4 using package   : 10

Computation of sum from 0 to 5 using recursion : 15
Computation of sum from 0 to 5 using package   : 15

Computation of sum from 0 to 6 using recursion : 21
Computation of sum from 0 to 6 using package   : 21

Computation of sum from 0 to 7 using recursion : 28
Computation of sum from 0 to 7 using package   : 28

Computation of sum from 0 to 8 using recursion : 36
Computation of sum from 0 to 8 using package   : 36

Computation of sum from 0 to 9 using recursion : 45
Computation of sum from 0 to 9 using package   : 45
```


<img src="img/Screen Shot 2022-06-16 at 11.11.35 AM.png" width=50%>


# Reference

[[알고리즘] 제1-1강 Recursion의 개념과 기본 예제들 (1/3)](https://www.youtube.com/watch?v=ln7AfppN7mY&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=1)
