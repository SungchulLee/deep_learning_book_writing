# Greatest Common Divisor


<img src="img/Screen Shot 2022-05-01 at 7.32.44 PM.png" width=50%>
<img src="img/Screen Shot 2022-05-01 at 7.33.55 PM.png" width=50%>


```python
import math


def compute_gcd_using_recursion(i, j):
    if j == 0:
        return i
    return compute_gcd_using_recursion(j, i%j)


def compute_gcd_using_while_loop(i, j):
    while j:
        i, j = j, i%j
    return i


def compute_gcd_using_package(i, j):
    return math.gcd(i, j)


def main():
    i_ = (6, 10, 25, 30, 364, 45246, 465344)
    j_ = (9, 10, 70, 12, 162, 41312, 413116)
    for i, j in zip(i_, j_):
        result_0 = compute_gcd_using_recursion(i, j)
        result_1 = compute_gcd_using_while_loop(i, j)
        result_2 = compute_gcd_using_package(i, j)
        print(f"Computation of gcd({i},{j}) using recursion  : {result_0}")
        print(f"Computation of gcd({i},{j}) using while loop : {result_1}")
        print(f"Computation of gcd({i},{j}) using package    : {result_2}")
        print()

    
if __name__ == "__main__":
    main()
```

**Output:**
```
Computation of gcd(6,9) using recursion  : 3
Computation of gcd(6,9) using while loop : 3
Computation of gcd(6,9) using package    : 3

Computation of gcd(10,10) using recursion  : 10
Computation of gcd(10,10) using while loop : 10
Computation of gcd(10,10) using package    : 10

Computation of gcd(25,70) using recursion  : 5
Computation of gcd(25,70) using while loop : 5
Computation of gcd(25,70) using package    : 5

Computation of gcd(30,12) using recursion  : 6
Computation of gcd(30,12) using while loop : 6
Computation of gcd(30,12) using package    : 6

Computation of gcd(364,162) using recursion  : 2
Computation of gcd(364,162) using while loop : 2
Computation of gcd(364,162) using package    : 2

Computation of gcd(45246,41312) using recursion  : 2
Computation of gcd(45246,41312) using while loop : 2
Computation of gcd(45246,41312) using package    : 2

Computation of gcd(465344,413116) using recursion  : 44
Computation of gcd(465344,413116) using while loop : 44
Computation of gcd(465344,413116) using package    : 44
```


# Reference

[[알고리즘] 제1-1강 Recursion의 개념과 기본 예제들 (1/3)](https://www.youtube.com/watch?v=ln7AfppN7mY&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=1)
