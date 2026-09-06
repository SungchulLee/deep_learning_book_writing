# 계승
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

**출력:**
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

# 참고 자료

[[알고리즘] 제1-1강 Recursion의 개념과 기본 예제들 (1/3)](https://www.youtube.com/watch?v=ln7AfppN7mY&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=1)

## 연습문제

**연습문제 1.**
이중 계승 $n!! = n \cdot (n-2) \cdot (n-4) \cdots$을 계산하는 재귀 함수를 작성하라.

??? success "연습문제 1 풀이"
    ```python
    def double_factorial(n):
        if n <= 1:
            return 1
        return n * double_factorial(n - 2)
    ```

---

**연습문제 2.**
파이썬에서 `factorial(1000)`의 최대 재귀 깊이는 얼마인가? 한계에 부딪히지 않으려면 어떻게 해야 하는가?

??? success "연습문제 2 풀이"
    파이썬의 기본 재귀 한계는 1000이므로 `factorial(1000)`은 `RecursionError`를 일으킨다. 해결책은 (1) `sys.setrecursionlimit(2000)`을 쓰거나, (2) 반복문으로 바꾸거나, (3) C로 구현된 `math.factorial`을 쓰는 것이다.

---

**연습문제 3.**
재귀적 계승이 모든 $n \geq 0$에 대해 $n!$을 올바르게 계산함을 귀납법으로 증명하라.

??? success "연습문제 3 풀이"
    **기저 사례**: $n=0$일 때 1을 반환하고 $0! = 1$이다. 올바르다.

    **귀납 단계**: 어떤 $k \geq 0$에 대해 `factorial(k)`가 $k!$을 반환한다고 가정하자. 그러면 `factorial(k+1)`은 $(k+1) \cdot$ `factorial(k)` $= (k+1) \cdot k! = (k+1)!$을 반환한다. $\square$

---

**연습문제 4.**
재귀적 계승 함수의 시간 복잡도와 공간 복잡도를 분석하라.

??? success "연습문제 4 풀이"
    시간: $T(n) = T(n-1) + O(1)$이므로 $T(n) = O(n)$이다. 호출마다 곱셈이 한 번이다.

    공간: 스택 프레임이 $n$개이므로 $O(n)$이다. 각 프레임은 정수 하나와 반환 주소를 저장한다. 반복 버전은 $O(1)$의 보조 공간을 쓴다.
