# 최대공약수
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

**출력:**
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

# 참고 자료

[[알고리즘] 제1-1강 Recursion의 개념과 기본 예제들 (1/3)](https://www.youtube.com/watch?v=ln7AfppN7mY&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=1)

## 연습문제

**연습문제 1.**
유클리드 호제법으로 `gcd(48, 18)`의 실행을 단계별로 따라가라.

??? success "연습문제 1 풀이"
    `gcd(48, 18)` -> `gcd(18, 48 % 18)` = `gcd(18, 12)` -> `gcd(12, 18 % 12)` = `gcd(12, 6)` -> `gcd(6, 12 % 6)` = `gcd(6, 0)` -> 6을 반환한다.

    실제로 $\gcd(48, 18) = 6$이다.

---

**연습문제 2.**
유클리드 호제법이 많아야 $O(\log(\min(a, b)))$ 단계 안에 종료됨을 증명하라.

??? success "연습문제 2 풀이"
    연속된 두 단계를 거치면 나머지가 적어도 절반으로 줄어든다. $r = a \bmod b$라 하면 $r \leq b/2$이거나(즉시 절반) $b \bmod r \leq b/2$이다(다음 단계에서 절반). $\min(a,b)$가 두 단계마다 절반이 되므로 많아야 $2\log_2(\min(a,b))$ 단계 안에 종료된다. $\square$

---

**연습문제 3.**
$\gcd(a,b)$와 함께 $ax + by = \gcd(a,b)$를 만족하는 계수 $x, y$를 반환하는 확장 유클리드 호제법을 구현하라.

??? success "연습문제 3 풀이"
    ```python
    def extended_gcd(a, b):
        if b == 0:
            return a, 1, 0
        g, x, y = extended_gcd(b, a % b)
        return g, y, x - (a // b) * y
    ```
    확인: $48(-1) + 18(3) = -48 + 54 = 6$이므로 `extended_gcd(48, 18)`은 `(6, -1, 3)`을 반환한다.

---

**연습문제 4.**
최대공약수를 사용해 $\text{lcm}(a, b)$를 계산하고 $\text{lcm}(a,b) = \frac{ab}{\gcd(a,b)}$인 이유를 설명하라.

??? success "연습문제 4 풀이"
    $a$와 $b$의 모든 공배수는 $\text{lcm}(a,b)$의 배수이다. 산술의 기본정리에 의해 $a = \prod p_i^{a_i}$이고 $b = \prod p_i^{b_i}$이면 $\gcd = \prod p_i^{\min(a_i,b_i)}$이고 $\text{lcm} = \prod p_i^{\max(a_i,b_i)}$이다. $\min + \max = a_i + b_i$이므로 $\gcd \cdot \text{lcm} = a \cdot b$이며 $\text{lcm} = ab / \gcd$를 얻는다.

    ```python
    def lcm(a, b):
        return a * b // gcd(a, b)
    ```
