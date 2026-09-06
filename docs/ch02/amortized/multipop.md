# Multipop 스택

Multipop 스택은 분할 상환 분석의 도입부에서 가장 많이 쓰이는 예이다. 표준 스택은 `PUSH`와 `POP`을 각각 $O(1)$에 지원한다. 여기에 한 번에 최대 $k$개의 원소를 제거하는 `MULTIPOP(k)` 연산을 더하면 최악의 경우 비용이 $O(n)$인 연산이 하나 생긴다. $n$개의 혼합 연산을 소박하게 최악의 경우로 분석하면 $O(n^2)$이 나오지만, 분할 상환 분석은 진짜 비용이 전체 $O(n)$, 즉 연산당 $O(1)$임을 보여준다.

## 연산

Multipop 스택은 현재 크기가 $s$인 스택에 대해 세 가지 연산을 지원한다.

| 연산 | 설명 | 실제 비용 |
|-----------|-------------|-------------|
| `PUSH(x)` | 원소 $x$를 스택에 밀어 넣는다 | $1$ |
| `POP()` | 맨 위 원소를 제거하고 반환한다 | $1$ |
| `MULTIPOP(k)` | 맨 위 $\min(k, s)$개의 원소를 제거한다 | $\min(k, s)$ |

`MULTIPOP`의 의사코드는 다음과 같다.

```
MULTIPOP(S, k):
    while S is not empty and k > 0:
        POP(S)
        k = k - 1
```

## 소박한 최악의 경우 분석

크기 $n$인 스택에 대한 `MULTIPOP(k)` 한 번은 $O(n)$이 든다. $n$개의 연산에 걸쳐 최악의 경우는 $O(n) \times n = O(n^2)$처럼 보인다. 그러나 이 분석은 각 원소가 꺼내지려면 먼저 밀어 넣어져야 한다는 제약을 무시하므로 지나치게 비관적이다.

## 총계 분석

핵심 관찰은 각 원소가 밀어 넣어진 횟수마다 많아야 한 번 꺼내진다는 것이다. 빈 스택에서 시작하는 임의의 $n$개 연산 수열에 대해 다음이 성립한다.

- 밀어 넣기의 총 횟수는 많아야 $n$이다.
- (모든 `POP`과 `MULTIPOP` 호출을 통틀어) 꺼내기의 총 횟수도 많아야 $n$이다. 밀어 넣은 것보다 많이 꺼낼 수는 없기 때문이다.

따라서 전체 비용은 다음과 같다.

$$
T(n) \leq n + n = 2n
$$

연산당 분할 상환 비용은 다음과 같다.

$$
\hat{c} = \frac{T(n)}{n} \leq 2 = O(1)
$$

## 회계 분석

다음과 같이 분할 상환 비용을 배정한다.

- `PUSH`: $\hat{c} = 2$(밀어 넣기 자체에 1, 원소에 신용으로 예치하는 데 1)
- `POP`: $\hat{c} = 0$(꺼내진 원소의 신용으로 지불한다)
- `MULTIPOP(k)`: $\hat{c} = 0$($\min(k, s)$개의 꺼내진 원소가 각각 스스로 지불한다)

**신용 불변식:** 현재 스택에 있는 모든 원소는 밀어 넣어질 때 예치된 정확히 1단위의 신용을 지니고 있다. (`POP`이든 `MULTIPOP`이든) 모든 꺼내기가 이전에 밀어 넣어진 원소를 제거하므로 신용이 항상 충분하다. 전체 신용은 스택 크기와 같으며 언제나 음이 아니다.

**결과:** $n$개 연산의 전체 분할 상환 비용은 (밀어 넣기가 많아야 $n$번이고 각각 $\hat{c} = 2$이므로) 많아야 $2n$이며, 연산당 $O(1)$ 분할 상환이다.

## 퍼텐셜 분석

퍼텐셜 함수를 스택 크기로 정의한다.

$$
\Phi(D) = |S|
$$

여기서 $|S|$는 스택 위의 원소 개수이다. 이는 $\Phi(D_0) = 0$을 만족하고 모든 $i$에 대해 $\Phi(D_i) \geq 0$이다.

**PUSH의 분할 상환 비용:**

$$
\hat{c}_i = c_i + \Phi(D_i) - \Phi(D_{i-1}) = 1 + 1 = 2
$$

**POP의 분할 상환 비용:**

$$
\hat{c}_i = c_i + \Phi(D_i) - \Phi(D_{i-1}) = 1 + (-1) = 0
$$

**MULTIPOP(k)의 분할 상환 비용**($k' = \min(k, s)$개를 꺼낼 때):

$$
\hat{c}_i = c_i + \Phi(D_i) - \Phi(D_{i-1}) = k' + (-k') = 0
$$

전체 분할 상환 비용은 다음과 같다.

$$
\sum_{i=1}^{n} \hat{c}_i \leq 2n = O(n)
$$

($\Phi(D_n) \geq \Phi(D_0) = 0$이므로) $\sum \hat{c}_i \geq \sum c_i$이며, 이로써 연산당 $O(1)$의 분할 상환 비용이 확인된다.

## 파이썬 예제

```python
"""
Multipop 스택의 분할 상환 분석 시연.

MULTIPOP이 있는 스택을 구현하고 실제 비용을 추적하여
O(1) 분할 상환 경계를 확인한다.
"""


# ===================================================================
# Multipop 스택 구현
# ===================================================================
class MultipopStack:
    """비용을 추적하는 push, pop, multipop 스택."""

    def __init__(self):
        self.items = []
        self.total_cost = 0
        self.num_ops = 0

    def push(self, x):
        """x를 스택에 밀어 넣는다. 비용 = 1."""
        self.items.append(x)
        self.total_cost += 1
        self.num_ops += 1
        return 1

    def pop(self):
        """맨 위 원소를 꺼내 반환한다. 비용 = 1."""
        if not self.items:
            raise IndexError("pop from empty stack")
        val = self.items.pop()
        self.total_cost += 1
        self.num_ops += 1
        return val

    def multipop(self, k):
        """min(k, 크기)개의 원소를 꺼낸다. 비용 = 꺼낸 개수."""
        num_popped = min(k, len(self.items))
        for _ in range(num_popped):
            self.items.pop()
        self.total_cost += num_popped
        self.num_ops += 1
        return num_popped

    def size(self):
        """현재 스택 크기(퍼텐셜이기도 하다)를 반환한다."""
        return len(self.items)

    def amortized_cost(self):
        """지금까지의 연산당 평균 비용을 반환한다."""
        if self.num_ops == 0:
            return 0.0
        return self.total_cost / self.num_ops


# ===================================================================
# 시연: 최악의 경우는 나빠 보이지만 분할 상환은 괜찮다
# ===================================================================
def demo_multipop():
    """multipop의 비용이 분할 상환으로 O(1)임을 보인다."""
    stack = MultipopStack()

    # n개를 밀어 넣은 뒤 한 번에 모두 꺼낸다
    n = 1000
    for i in range(n):
        stack.push(i)
    print(f"After {n} pushes: size={stack.size()}, "
          f"total_cost={stack.total_cost}, ops={stack.num_ops}")

    # 비싼 multipop 한 번
    popped = stack.multipop(n)
    print(f"Multipop({n}): popped={popped}, "
          f"total_cost={stack.total_cost}, ops={stack.num_ops}")
    print(f"Amortized cost/op: {stack.amortized_cost():.4f}")
    print(f"Bound (2.0): {stack.amortized_cost() <= 2.0}")


# ===================================================================
# 시연: 혼합 연산
# ===================================================================
def demo_mixed():
    """push/pop/multipop을 섞은 연산."""
    stack = MultipopStack()
    operations = []

    # 혼합 연산 수열을 흉내 낸다
    import random
    random.seed(42)
    for _ in range(500):
        r = random.random()
        if r < 0.6 or stack.size() == 0:
            stack.push(random.randint(1, 100))
            operations.append("PUSH")
        elif r < 0.8:
            stack.pop()
            operations.append("POP")
        else:
            k = random.randint(1, max(1, stack.size()))
            stack.multipop(k)
            operations.append(f"MULTIPOP")

    push_count = operations.count("PUSH")
    pop_count = operations.count("POP")
    mpop_count = sum(1 for op in operations if op == "MULTIPOP")

    print(f"\nMixed operations: {push_count} PUSH, "
          f"{pop_count} POP, {mpop_count} MULTIPOP")
    print(f"Total ops: {stack.num_ops}, total cost: {stack.total_cost}")
    print(f"Amortized cost/op: {stack.amortized_cost():.4f}")
    print(f"Bound (2.0): {stack.amortized_cost() <= 2.0}")


# ===================================================================
# 메인
# ===================================================================
if __name__ == "__main__":
    print("=== Worst-Case Multipop ===")
    demo_multipop()

    print("\n=== Mixed Operations ===")
    demo_mixed()
```

## 이 예제가 중요한 이유

Multipop 스택은 알고리즘 설계 전반에 나타나는 패턴을 보여준다. (`MULTIPOP`처럼) 이따금 비싼 연산이 (`PUSH`처럼) 수많은 싼 연산으로 상쇄되는 패턴이다. 같은 패턴이 동적 배열(이따금 일어나는 크기 조정), 해시 테이블 재해싱(이따금 일어나는 재구축), 스플레이 트리 연산(이따금 일어나는 깊은 회전 수열)에도 나타난다. Multipop 스택을 이해하면 이런 더 복잡한 분석에 필요한 직관을 얻을 수 있다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 16: Amortized Analysis. MIT Press.


## 연습문제

**연습문제 1.**
Multipop 스택에서 설명한 방법을 사용하여 $n$개 연산의 수열을 분석하고 연산당 분할 상환 비용을 구하라.

??? success "연습문제 1 풀이"
    이 절의 구체적인 기법(총계, 회계, 퍼텐셜)을 적용하여 $n$개 연산의 전체 비용에 상계를 준다. 이를 $n$으로 나누면 연산당 분할 상환 비용을 얻는다. 핵심 통찰은 비싼 연산이 충분히 드물어서 그 비용이 수많은 싼 연산에 흩어진다는 것이다.

---

**연습문제 2.**
Multipop 스택의 분할 상환 패턴, 즉 싼 연산들이 이따금 비싼 연산 하나를 촉발하는 패턴에 대응하는 딥러닝 상황을 찾아라.

??? success "연습문제 2 풀이"
    예로는 경사 누적(싼 마이크로배치 순전파, 비싼 매개변수 갱신), 모델 체크포인팅(싼 학습 단계, 비싼 저장), 동적 배치에서의 해시 테이블 크기 조정(싼 삽입, 비싼 재해싱)이 있다. 이따금 비싼 연산이 있어도 단계당 분할 상환 비용은 일정하게 유지된다.

---

**연습문제 3.**
Multipop 스택으로 유도한 분할 상환 경계가 꽉 조여 있음을, 그 경계를 달성하는 연산 수열을 구성하여 증명하라.

??? success "연습문제 3 풀이"
    전체 실제 비용을 연산 횟수로 나눈 비를 최대화하는 수열을 구성한다. 보통 퍼텐셜을 쌓는 연산과 그것을 방출하는 연산을 번갈아 수행하는 형태가 된다. 이 구성으로 분할 상환 경계를 더 개선할 수 없음이 확인된다. $\square$

---

**연습문제 4.**
Multipop 스택의 분할 상환 분석을 최악의 경우 분석과 비교하라. 분할 상환 경계는 최악의 경우보다 몇 배나 개선되는가?

??? success "연습문제 4 풀이"
    최악의 경우 분석은 비싼 연산 앞에 싼 연산이 많이 온다는 사실을 무시하고 각 연산에 가능한 최대 비용을 부과한다. 개선 배수는 최악의 경우 비용을 분할 상환 비용으로 나눈 값이다. 전형적인 자료구조에서는 $O(n)$ 대 $O(1)$로 $n$배 개선된다.
