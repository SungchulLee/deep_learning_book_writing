# 재귀로 구하는 피보나치 수

피보나치 수열은 컴퓨터 과학에서 재귀의 가장 자연스러운 예 중 하나이다. 각 항이 앞의 두 항의 합으로 정의되므로 재귀적 정의가 수학 공식을 거의 그대로 옮긴 것이 된다. 그러나 소박한 재귀 구현은 근본적인 효율성 문제를 드러내며, 이것이 메모화와 동적 계획법 같은 핵심 알고리즘 설계 기법의 동기가 된다.

## 정의

피보나치 수열은 다음과 같이 재귀적으로 정의된다.

$$
F(n) = \begin{cases} 0 & \text{if } n = 0 \\ 1 & \text{if } n = 1 \\ F(n-1) + F(n-2) & \text{if } n \geq 2 \end{cases}
$$

처음 몇 항은 다음과 같다: $0, 1, 1, 2, 3, 5, 8, 13, 21, 34, \ldots$

## 소박한 재귀 구현

재귀적 정의는 파이썬 코드로 그대로 옮겨진다.

```python
def compute_fibonacci_using_recursion(n):
    """Compute the n-th Fibonacci number using naive recursion.

    Time complexity: O(2^n) -- exponential due to repeated subproblems.
    Space complexity: O(n) -- maximum recursion depth.
    """
    if n <= 1:
        return n
    return compute_fibonacci_using_recursion(n - 1) + compute_fibonacci_using_recursion(n - 2)


# === 메인 ===
if __name__ == "__main__":
    n = 10
    for i in range(n):
        result = compute_fibonacci_using_recursion(i)
        print(f"fibonacci({i}) = {result}")
```

**출력:**
```
fibonacci(0) = 0
fibonacci(1) = 1
fibonacci(2) = 1
fibonacci(3) = 2
fibonacci(4) = 3
fibonacci(5) = 5
fibonacci(6) = 8
fibonacci(7) = 13
fibonacci(8) = 21
fibonacci(9) = 34
```

## 복잡도 분석

소박한 재귀 접근은 **지수** 시간 복잡도를 가진다. `compute_fibonacci_using_recursion(n)`을 호출할 때마다 재귀 호출이 두 개 생기고 많은 부분문제가 거듭 풀린다. 호출 횟수는 점화식 $T(n) = T(n-1) + T(n-2) + O(1)$을 만족하며 다음과 같이 증가한다.

$$
T(n) = O(\phi^n) \quad \text{where } \phi = \frac{1 + \sqrt{5}}{2} \approx 1.618
$$

$n = 40$이면 함수 호출이 10억 번을 넘는다. 이러한 중복 계산 때문에 소박한 재귀는 아주 작은 입력을 빼면 실용적이지 않다.

!!! warning "지수적 폭발"
    소박한 재귀 피보나치는 올바른 알고리즘이 쓸 수 없을 만큼 느릴 수 있음을 보여주는 고전적인 예이다. 해결책은 이전에 계산한 결과를 저장하거나(메모화) 상향식으로 해를 쌓는 것(동적 계획법)이며, 시간 복잡도를 $O(\phi^n)$에서 $O(n)$으로 줄인다.

## 참고 자료s

[Introduction to Algorithms (CLRS), Section 15.1](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)


## 연습문제

**연습문제 1.**
소박한 재귀 구현으로 $F(10)$을 계산할 때의 정확한 재귀 호출 횟수를 구하라. 값 $F(10) = 55$와 비교하라.

??? success "연습문제 1 풀이"
    호출 횟수 $C(n)$은 $C(0) = C(1) = 1$인 $C(n) = C(n-1) + C(n-2) + 1$을 만족한다. $C(10) = 177$번으로 $F(10) = 55$보다 훨씬 크다. $F(k)$가 $F(n-k)$번 거듭 계산되기 때문에 이런 중복이 생긴다.

---

**연습문제 2.**
$\phi = (1+\sqrt{5})/2$일 때 소박한 재귀 피보나치가 $\Omega(\phi^n)$ 시간에 실행됨을 귀납법으로 증명하라.

??? success "연습문제 2 풀이"
    **기저 단계**: $T(0) = T(1) = 1 \geq \phi^0 = 1$. $\checkmark$ **귀납 단계**: $T(n) = T(n-1) + T(n-2) \geq \phi^{n-1} + \phi^{n-2} = \phi^{n-2}(\phi + 1) = \phi^{n-2} \cdot \phi^2 = \phi^n$($\phi^2 = \phi + 1$을 사용). $\square$

---

**연습문제 3.**
메모화가 시간을 $O(\phi^n)$에서 $O(n)$으로 줄임을 보여라. 서로 다른 부분문제는 몇 개인가?

??? success "연습문제 3 풀이"
    서로 다른 부분문제는 정확히 $n + 1$개, 즉 $F(0), F(1), \ldots, F(n)$이다. 각각 $O(1)$ 비용으로 한 번씩 계산된다. 총합은 $O(n)$이다. 메모화 표는 항목이 $n + 1$개이며 각각 정확히 한 번씩 채워진다.

---

**연습문제 4.**
딥러닝에서 피보나치의 재귀 패턴은 이진 트리를 처리하는 재귀 신경망에 나타난다. 노드 하나를 처리하는 비용이 $O(d^2)$일 때 잎이 $n$개인 균형 이진 트리의 전체 비용은 얼마인가?

??? success "연습문제 4 풀이"
    잎이 $n$개인 균형 이진 트리는 내부 노드가 $n - 1$개이고 전체 노드가 $2n - 1$개이다. 각각을 처리하는 비용이 $O(d^2)$이므로 총합은 $O(nd^2)$이다. 피보나치와 달리 중복되는 부분문제가 없으므로(각 부분 트리가 서로 다르다) 메모화가 도움이 되지 않는다. 점화식은 $T(n) = 2T(n/2) + O(d^2)$이며 $T(n) = O(nd^2)$을 준다.