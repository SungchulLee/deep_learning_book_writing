# 합
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

**출력:**
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

# 참고 자료

[[알고리즘] 제1-1강 Recursion의 개념과 기본 예제들 (1/3)](https://www.youtube.com/watch?v=ln7AfppN7mY&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=1)

## 연습문제

**연습문제 1.**
합 $1 + 2 + \cdots + n$을 계산하는 재귀 함수를 작성하고 공식 $n(n+1)/2$와 일치함을 확인하라.

??? success "연습문제 1 풀이"
    ```python
    def recursive_sum(n):
        if n == 0:
            return 0
        return n + recursive_sum(n - 1)

    assert recursive_sum(100) == 100 * 101 // 2  # 5050
    ```

---

**연습문제 2.**
재귀적 합을 꼬리 재귀 버전으로 바꾸어라.

??? success "연습문제 2 풀이"
    ```python
    def sum_tail(n, acc=0):
        if n == 0:
            return acc
        return sum_tail(n - 1, acc + n)
    ```
    누적자 `acc`가 부분합을 나르므로 재귀 호출 뒤에 남는 일이 없다.

---

**연습문제 3.**
재귀적 합과 꼬리 재귀 버전의 공간 복잡도는 각각 얼마인가? 파이썬은 꼬리 호출을 최적화하는가?

??? success "연습문제 3 풀이"
    CPython이 꼬리 호출 최적화(TCO)를 **하지 않으므로** 파이썬에서는 둘 다 $O(n)$의 스택 공간을 쓴다. TCO가 있는 언어(Scheme, Haskell)에서는 꼬리 재귀 버전이 $O(1)$ 공간을 쓸 것이다. 파이썬에서 $O(1)$ 공간을 얻으려면 반복문으로 바꾼다. `sum(range(n+1))`.

---

**연습문제 4.**
임의의 함수 $f$에 대해 $\sum_{i=1}^n f(i)$를 재귀적으로 계산하도록 일반화하라.

??? success "연습문제 4 풀이"
    ```python
    def sum_function(f, n):
        if n == 0:
            return 0
        return f(n) + sum_function(f, n - 1)

    # 제곱의 합: 1^2 + 2^2 + ... + 10^2
    result = sum_function(lambda x: x**2, 10)
    assert result == 385  # n(n+1)(2n+1)/6 = 10*11*21/6
    ```
