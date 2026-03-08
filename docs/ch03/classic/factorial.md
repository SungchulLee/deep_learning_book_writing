# Factorial

<img src="img/Screen Shot 2022-05-01 at 7.13.46 PM.png" width=50%>

```python
import math

def compute_factorial_using_recursion(n):
    if n == 0:
        return 1
    return n * compute_factorial_using_recursion(n-1)

def compute_factorial_using_package(n):
    return math.factorial(n)

def main():
    n = 10
    for i in range(n):
        result_0 = compute_factorial_using_recursion(i)
        result_1 = compute_factorial_using_package(i)
        print(f"Computation of {i}! using recursion : {result_0}")
        print(f"Computation of {i}! using package   : {result_1}")
        print()
        
        
if __name__ == "__main__":
    main()
```

**Output:**
```
Computation of 0! using recursion : 1
Computation of 0! using package   : 1

Computation of 1! using recursion : 1
Computation of 1! using package   : 1

Computation of 2! using recursion : 2
Computation of 2! using package   : 2

Computation of 3! using recursion : 6
Computation of 3! using package   : 6

Computation of 4! using recursion : 24
Computation of 4! using package   : 24

Computation of 5! using recursion : 120
Computation of 5! using package   : 120

Computation of 6! using recursion : 720
Computation of 6! using package   : 720

Computation of 7! using recursion : 5040
Computation of 7! using package   : 5040

Computation of 8! using recursion : 40320
Computation of 8! using package   : 40320

Computation of 9! using recursion : 362880
Computation of 9! using package   : 362880
```

<img src="img/Screen Shot 2022-06-08 at 4.22.22 PM.png" width=50%>

# Reference

[[알고리즘] 제1-1강 Recursion의 개념과 기본 예제들 (1/3)](https://www.youtube.com/watch?v=ln7AfppN7mY&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=1)
