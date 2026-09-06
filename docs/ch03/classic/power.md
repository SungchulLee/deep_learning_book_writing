# 거듭제곱 계산
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

**출력:**
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

# 참고 자료

[[알고리즘] 제1-1강 Recursion의 개념과 기본 예제들 (1/3)](https://www.youtube.com/watch?v=ln7AfppN7mY&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=1)

## 연습문제

**연습문제 1.**
$x^n$을 $O(\log n)$ 시간에 계산하는 빠른 거듭제곱(제곱을 이용한 거듭제곱)을 구현하라.

??? success "연습문제 1 풀이"
    ```python
    def fast_power(x, n):
        if n == 0:
            return 1
        if n % 2 == 0:
            half = fast_power(x, n // 2)
            return half * half
        return x * fast_power(x, n - 1)
    ```

---

**연습문제 2.**
거듭제곱 함수를 음의 지수도 다루도록 확장하라.

??? success "연습문제 2 풀이"
    ```python
    def power(x, n):
        if n < 0:
            return 1 / power(x, -n)
        if n == 0:
            return 1
        if n % 2 == 0:
            half = power(x, n // 2)
            return half * half
        return x * power(x, n - 1)
    ```

---

**연습문제 3.**
$2^{16}$을 계산할 때 소박한 재귀와 제곱을 이용한 거듭제곱의 곱셈 횟수를 비교하라.

??? success "연습문제 3 풀이"
    소박한 방법: $T(n) = T(n-1) + 1$이므로 $T(16) = 15$번의 곱셈이다.

    빠른 방법: 짝수 $n$에 대해 $T(n) = T(n/2) + 1$이므로 $T(16) = T(8) + 1 = T(4) + 2 = T(2) + 3 = T(1) + 4 = 4$번의 곱셈이다. 속도 향상은 $15/4 \approx 3.75\times$이다.

---

**연습문제 4.**
같은 제곱 기법을 사용하여 모듈러 거듭제곱 $x^n \bmod m$을 구현하라. 암호학에서 유용하다.

??? success "연습문제 4 풀이"
    ```python
    def mod_power(x, n, m):
        if n == 0:
            return 1
        if n % 2 == 0:
            half = mod_power(x, n // 2, m)
            return (half * half) % m
        return (x * mod_power(x, n - 1, m)) % m
    ```
    각 단계에서 나머지를 취하면 정수 넘침을 막고 중간 결과를 $m^2$ 이하로 유지할 수 있다.
