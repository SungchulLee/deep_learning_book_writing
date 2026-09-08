# 퍼텐셜 방법

총계 방법은 전체 비용을 계산해 $n$으로 나눈다. 회계 방법은 연산별로 비용을 부과하고 신용을 지역적으로 추적한다. 퍼텐셜 방법(potential method)은 더 전역적인 접근을 취한다. 물리학의 위치 에너지와 비슷하게, 자료구조 전체 상태의 함수 하나를 정의한다. 자료구조가 "에너지가 높은" 상태로 옮겨 가면 퍼텐셜이 증가하며 그 차이를 나중에 쓰도록 저장한다. 비싼 연산이 상태를 "낮은 에너지"로 되돌리면 저장된 퍼텐셜이 그 비용을 지불한다. 그래서 퍼텐셜 방법은 세 가지 분할 상환 분석 기법 중 가장 강력하고 유연하다.

---

## 1. 정의

연산 $0, 1, \ldots, n$ 후의 자료구조 상태를 $D_0, D_1, \ldots, D_n$이라 하고 $D_0$을 초기 상태라 하자. **퍼텐셜 함수** $\Phi$는 각 상태를 실수로 대응시킨다.

$$
\Phi: \{D_0, D_1, \ldots, D_n\} \to \mathbb{R}
$$

$i$번째 연산의 **분할 상환 비용** 은 다음과 같이 정의된다.

$$
\hat{c}_i = c_i + \Phi(D_i) - \Phi(D_{i-1})
$$

여기서 $c_i$는 실제 비용이다. 퍼텐셜의 변화 $\Delta\Phi_i = \Phi(D_i) - \Phi(D_{i-1})$이 보정항 역할을 한다. $\Delta\Phi_i > 0$이면 그 연산이 저장된 에너지를 늘리고(분할 상환 비용이 실제보다 커진다), $\Delta\Phi_i < 0$이면 저장된 에너지를 방출한다(분할 상환 비용이 실제보다 작아진다).

---

## 2. 핵심 정리

$\Phi(D_n) \geq \Phi(D_0)$이면 전체 분할 상환 비용은 전체 실제 비용의 상계이다.

$$
\sum_{i=1}^{n} \hat{c}_i = \sum_{i=1}^{n} c_i + \Phi(D_n) - \Phi(D_0) \geq \sum_{i=1}^{n} c_i
$$

중간의 퍼텐셜 항들이 상쇄되는 망원경 합에서 이것이 따라 나온다. 이 경계를 보장하려면 다음을 요구하면 충분하다.

$$
\Phi(D_i) \geq \Phi(D_0) \quad \text{for all } i = 1, 2, \ldots, n
$$

흔한 관례는 $\Phi(D_0) = 0$으로 두고 모든 $i$에 대해 $\Phi(D_i) \geq 0$을 요구하는 것이다.

!!! tip "좋은 퍼텐셜 함수 고르기"
    퍼텐셜 방법의 기예는 $\Phi$를 고르는 데 있다. 좋은 퍼텐셜 함수는 다음 조건을 만족해야 한다.

    - 초기 상태에서 0(또는 작은 값)이다
    - 싼 연산에서 증가한다(에너지를 저장한다)
    - 비싼 연산에서 감소한다(비용을 지불할 에너지를 방출한다)
    - 모든 종류의 연산에 대해 분할 상환 비용을 이상적으로는 상수인, 간단한 식으로 만든다

---

## 3. 예: Multipop 스택

$\Phi(D) = |S|$, 즉 스택 위의 원소 개수로 정의하자.

**PUSH:** 실제 비용 $c = 1$이고 퍼텐셜이 1만큼 증가한다.

$$
\hat{c} = 1 + 1 = 2
$$

**POP:** 실제 비용 $c = 1$이고 퍼텐셜이 1만큼 감소한다.

$$
\hat{c} = 1 + (-1) = 0
$$

**MULTIPOP(k):** 실제 비용 $c = k' = \min(k, s)$이고 퍼텐셜이 $k'$만큼 감소한다.

$$
\hat{c} = k' + (-k') = 0
$$

어떤 연산이든 분할 상환 비용이 많아야 2이므로 연산당 $O(1)$ 분할 상환이다.

---

## 4. 예: 이진 계수기

$i$번째 증가 후의 1비트 개수를 $b_i$라 하고 $\Phi(D_i) = b_i$로 정의하자.

$i$번째 증가가 $t_i$개의 비트를 1에서 0으로 되돌리고 한 비트를 0에서 1로 세운다면 $c_i = t_i + 1$이고 $\Delta\Phi_i = 1 - t_i$이다.

$$
\hat{c}_i = (t_i + 1) + (1 - t_i) = 2
$$

비용이 큰 $t_i$번의 뒤집기가 퍼텐셜 감소와 정확히 상쇄되어 상수 분할 상환 비용이 나온다.

---

## 5. 예: 동적 배열

크기가 $s$이고 용량이 $C$인 동적 배열에 대해 다음과 같이 정의하자.

$$
\Phi(D) = 2s - C
$$

**크기 조정 없는 추가:** $c = 1$이고 크기가 1 증가하며 용량은 그대로이다.

$$
\hat{c} = 1 + [2(s+1) - C] - [2s - C] = 1 + 2 = 3
$$

**크기 조정이 있는 추가**($s = C$일 때): 배열이 용량 $2C$로 두 배가 되고 $s$개의 원소를 복사한 뒤 삽입한다. 실제 비용은 $c = s + 1$이다. 크기 조정 후 크기는 $s + 1$이고 용량은 $2s$이다.

$$
\hat{c} = (s + 1) + [2(s+1) - 2s] - [2s - s] = (s+1) + 2 - s = 3
$$

두 경우 모두 분할 상환 비용이 정확히 3이다.

---

## 6. 일반적인 틀

퍼텐셜 방법은 다음의 경우에 특히 유용하다.

1. 서로 다른 연산이 상호작용하는 **다중 연산 자료구조**(예: 삽입이 만들어 낸 일을 삭제가 치워야 하는 경우).
2. 퍼텐셜이 트리의 "무질서도"를 포착하는 스플레이 트리 같은 **자기 조정 자료구조.**
3. (회계 방법의) 원소별 신용을 추적하기 어려운 **복잡한 상태 전이.**

퍼텐셜 방법은 다음 항등식을 통해 회계 방법과 연결된다.

$$
\text{credit after operation } i = \Phi(D_i) - \Phi(D_0)
$$

회계 방법의 신용 불변식(신용 $\geq 0$)은 퍼텐셜 방법의 요구조건($\Phi(D_i) \geq \Phi(D_0)$)과 동치이다.

---

## 7. 파이썬 예제

```python
"""
퍼텐셜 방법 시연.

multipop 스택과 동적 배열에 퍼텐셜 방법을 적용하여
분할 상환 비용이 이론적 예측과 일치함을 확인한다.
"""

# ===================================================================
# 퍼텐셜을 추적하는 Multipop 스택
# ===================================================================
class MultipopStackPotential:
    """퍼텐셜 = 원소 개수인 스택."""

    def __init__(self):
        self.items = []
        self.total_actual = 0
        self.total_amortized = 0
        self.num_ops = 0

    def potential(self):
        """Phi(D) = 스택 크기."""
        return len(self.items)

    def push(self, x):
        """퍼텐셜을 추적하며 밀어 넣는다."""
        actual = 1
        phi_before = self.potential()
        self.items.append(x)
        phi_after = self.potential()
        amortized = actual + (phi_after - phi_before)

        self.total_actual += actual
        self.total_amortized += amortized
        self.num_ops += 1
        return actual, amortized

    def multipop(self, k):
        """퍼텐셜을 추적하며 여러 개를 꺼낸다."""
        k_prime = min(k, len(self.items))
        actual = k_prime
        phi_before = self.potential()
        for _ in range(k_prime):
            self.items.pop()
        phi_after = self.potential()
        amortized = actual + (phi_after - phi_before)

        self.total_actual += actual
        self.total_amortized += amortized
        self.num_ops += 1
        return actual, amortized

# ===================================================================
# 퍼텐셜을 추적하는 동적 배열
# ===================================================================
class DynamicArrayPotential:
    """퍼텐셜 = 2*크기 - 용량인 동적 배열."""

    def __init__(self):
        self.capacity = 1
        self.size = 0
        self.data = [None] * self.capacity
        self.total_actual = 0
        self.total_amortized = 0
        self.num_ops = 0

    def potential(self):
        """Phi(D) = 2*크기 - 용량."""
        return 2 * self.size - self.capacity

    def append(self, value):
        """퍼텐셜을 추적하며 추가한다."""
        phi_before = self.potential()

        actual = 1
        if self.size == self.capacity:
            actual += self.size  # 복사 비용
            new_data = [None] * (2 * self.capacity)
            for i in range(self.size):
                new_data[i] = self.data[i]
            self.data = new_data
            self.capacity *= 2

        self.data[self.size] = value
        self.size += 1

        phi_after = self.potential()
        amortized = actual + (phi_after - phi_before)

        self.total_actual += actual
        self.total_amortized += amortized
        self.num_ops += 1
        return actual, amortized

# ===================================================================
# 시연
# ===================================================================
if __name__ == "__main__":
    # --- Multipop 스택 ---
    print("=== Multipop Stack (Potential Method) ===")
    stack = MultipopStackPotential()
    for i in range(10):
        actual, amortized = stack.push(i)
    print(f"After 10 pushes: Phi={stack.potential()}, "
          f"total_actual={stack.total_actual}, "
          f"total_amortized={stack.total_amortized}")

    actual, amortized = stack.multipop(10)
    print(f"Multipop(10): actual={actual}, amortized={amortized}")
    print(f"Final: Phi={stack.potential()}, "
          f"total_actual={stack.total_actual}, "
          f"total_amortized={stack.total_amortized}")
    print(f"Amortized >= Actual: "
          f"{stack.total_amortized >= stack.total_actual}")

    # --- 동적 배열 ---
    print("\n=== Dynamic Array (Potential Method) ===")
    arr = DynamicArrayPotential()
    print(f"{'Op':>3} {'Actual':>7} {'Amort':>7} {'Phi':>5} "
          f"{'Size':>5} {'Cap':>5}")
    print("-" * 38)
    for i in range(1, 18):
        actual, amortized = arr.append(i)
        print(f"{i:>3} {actual:>7} {amortized:>7} "
              f"{arr.potential():>5} {arr.size:>5} {arr.capacity:>5}")

    print(f"\nTotal actual:    {arr.total_actual}")
    print(f"Total amortized: {arr.total_amortized}")
    print(f"Amortized >= Actual: "
          f"{arr.total_amortized >= arr.total_actual}")
    print(f"Avg amortized/op: "
          f"{arr.total_amortized / arr.num_ops:.2f}")
```

---

## 연습문제

**연습문제 1.**
퍼텐셜 방법에서 설명한 방법을 사용하여 $n$개 연산의 수열을 분석하고 연산당 분할 상환 비용을 구하라.

??? success "연습문제 1 풀이"
    이 절의 구체적인 기법(총계, 회계, 퍼텐셜)을 적용하여 $n$개 연산의 전체 비용에 상계를 준다. 이를 $n$으로 나누면 연산당 분할 상환 비용을 얻는다. 핵심 통찰은 비싼 연산이 충분히 드물어서 그 비용이 수많은 싼 연산에 흩어진다는 것이다.

---

**연습문제 2.**
퍼텐셜 방법의 분할 상환 패턴, 즉 싼 연산들이 이따금 비싼 연산 하나를 촉발하는 패턴에 대응하는 딥러닝 상황을 찾아라.

??? success "연습문제 2 풀이"
    예로는 경사 누적(싼 마이크로배치 순전파, 비싼 매개변수 갱신), 모델 체크포인팅(싼 학습 단계, 비싼 저장), 동적 배치에서의 해시 테이블 크기 조정(싼 삽입, 비싼 재해싱)이 있다. 이따금 비싼 연산이 있어도 단계당 분할 상환 비용은 일정하게 유지된다.

---

**연습문제 3.**
퍼텐셜 방법으로 유도한 분할 상환 경계가 꽉 조여 있음을, 그 경계를 달성하는 연산 수열을 구성하여 증명하라.

??? success "연습문제 3 풀이"
    전체 실제 비용을 연산 횟수로 나눈 비를 최대화하는 수열을 구성한다. 보통 퍼텐셜을 쌓는 연산과 그것을 방출하는 연산을 번갈아 수행하는 형태가 된다. 이 구성으로 분할 상환 경계를 더 개선할 수 없음이 확인된다. $\square$

---

**연습문제 4.**
퍼텐셜 방법의 분할 상환 분석을 최악의 경우 분석과 비교하라. 분할 상환 경계는 최악의 경우보다 몇 배나 개선되는가?

??? success "연습문제 4 풀이"
    최악의 경우 분석은 비싼 연산 앞에 싼 연산이 많이 온다는 사실을 무시하고 각 연산에 가능한 최대 비용을 부과한다. 개선 배수는 최악의 경우 비용을 분할 상환 비용으로 나눈 값이다. 전형적인 자료구조에서는 $O(n)$ 대 $O(1)$로 $n$배 개선된다.

## 정리하며

이 마당은 정의、핵심 정리、예: Multipop 스택、예: 이진 계수기을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 16: Amortized Analysis. MIT Press.
