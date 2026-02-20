# Power Computation


<img src="img/Screen Shot 2022-05-01 at 7.18.21 PM.png" width=50%>


```python
def compute_power_using_recursion(x, n):
    if n == 0:
        return 1
    return x * compute_power_using_recursion(x, n-1)


def compute_power_using_python(x, n):
    return x ** n


def main():
    x = 2
    n = 10
    for i in range(n):
        result_0 = compute_power_using_recursion(x, i)
        result_1 = compute_power_using_python(x, i)
        print(f"Computation of {x}^{i} using recursion : {result_0}")
        print(f"Computation of {x}^{i} using python    : {result_1}")
        print()
        
        
if __name__ == "__main__":
    main()
```

**Output:**
```
Computation of 2^0 using recursion : 1
Computation of 2^0 using python    : 1

Computation of 2^1 using recursion : 2
Computation of 2^1 using python    : 2

Computation of 2^2 using recursion : 4
Computation of 2^2 using python    : 4

Computation of 2^3 using recursion : 8
Computation of 2^3 using python    : 8

Computation of 2^4 using recursion : 16
Computation of 2^4 using python    : 16

Computation of 2^5 using recursion : 32
Computation of 2^5 using python    : 32

Computation of 2^6 using recursion : 64
Computation of 2^6 using python    : 64

Computation of 2^7 using recursion : 128
Computation of 2^7 using python    : 128

Computation of 2^8 using recursion : 256
Computation of 2^8 using python    : 256

Computation of 2^9 using recursion : 512
Computation of 2^9 using python    : 512
```


# Reference

[[알고리즘] 제1-1강 Recursion의 개념과 기본 예제들 (1/3)](https://www.youtube.com/watch?v=ln7AfppN7mY&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=1)
