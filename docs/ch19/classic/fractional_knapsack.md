# 쪼갤 수 있는 배낭

The knapsack problem asks: given a set of items, each with a weight and a value, which items should a thief put in a knapsack of limited capacity to maximize total value? In the **fractional** variant, the thief may take fractions of items --- for example, pouring half a bag of gold dust into the sack. This seemingly small relaxation changes the problem fundamentally: while the 0-1 knapsack requires dynamic programming (and is NP-hard), the fractional knapsack admits an elegant $O(n \log n)$ greedy solution.

## 문제 서술

**들임.**

- 무게가 $w_i > 0$이고 값이 $v_i > 0$인 물건 $n$개.
- 배낭의 담이 $W > 0$.

**Decision variable.** For each item $i$, choose a fraction $x_i \in [0, 1]$ to take.

**목표.** 전체 값을 가장 크게 한다:

$$
\max \sum_{i=1}^{n} v_i \cdot x_i
$$

**제약.** 전체 무게가 담이를 넘어서는 안 된다:

$$
\sum_{i=1}^{n} w_i \cdot x_i \leq W
$$

## 여기서 욕심쟁이가 통하는 까닭

핵심 눈썰미는 **값 대 무게 비** $r_i = v_i / w_i$이다. 물건 $i$의 무게 한 단위가 전체 값에 $r_i$만큼 보탠다. 쪼갤 수 있으므로 도둑은 비가 가장 큰 물건을 앞세워야 한다. 곧 값의 "밀도"가 가장 높은 것부터 배낭을 채운다.

이것이 통하는 까닭은:

1. 비가 큰 물건을 조금 담는 것이, 비가 작은 물건을 같은 무게만큼 담는 것보다 늘 낫다.
2. 일부만 채우는 것을 나쁘게 만들 쪼갤 수 없음의 제약이 없다.

## 욕심쟁이 알고리즘

!!! note "쪼갤 수 있는 배낭 알고리즘"

    1. 물건마다 값 대 무게 비 $r_i = v_i / w_i$을 셈한다.
    2. 물건을 $r_i$의 내림차순으로 정렬한다.
    3. (정렬된 차례로) 물건마다:
        - 물건 전체가 들어가면 다 담는다($x_i = 1$).
        - Otherwise, take the fraction that fills the remaining capacity ($x_i = (W_{\text{remaining}}) / w_i$) and stop.

## 풀이 예제

**담이:** $W = 50$.

| 물건 | $w_i$ | $v_i$ | $r_i = v_i/w_i$ |
|------|--------|--------|------------------|
| A    | 10     | 60     | 6.0              |
| B    | 20     | 100    | 5.0              |
| C    | 30     | 120    | 4.0              |

**비로 정렬:** A (6.0), B (5.0), C (4.0).

**욕심쟁이 실행:**

1. A를 다 담는다: 쓴 무게 = 10, 값 = 60, 남은 담이 = 40.
2. B를 다 담는다: 쓴 무게 = 30, 값 = 160, 남은 담이 = 20.
3. C을 $20/30 = 2/3$만큼 담는다: 쓴 무게 = 50, 값 = $160 + 80 = 240$.

$$
\text{Total value} = 60 + 100 + \frac{2}{3} \cdot 120 = 240
$$

**0-1 배낭과의 견줌:** 0-1의 최적은 $v_B + v_C = 220$이다(B와 C을 통째로 담는다). 쪼갤 수 있는 풀이는 물건 C을 쪼개어 240이라는 더 높은 값을 얻는다.

## 옳음의 증명

**정리.** 욕심쟁이 알고리즘은 쪼갤 수 있는 배낭 문제의 가장 좋은 풀이를 낸다.

**Proof.** Without loss of generality, assume items are sorted so that $r_1 \geq r_2 \geq \cdots \geq r_n$. Let $G = (x_1^G, \ldots, x_n^G)$ be the greedy solution and $S^* = (x_1^*, \ldots, x_n^*)$ be any optimal solution.

Suppose $G \neq S^*$. Let $j$ be the first index where they differ: $x_j^G \neq x_j^*$.

**Case 1:** $x_j^G > x_j^*$ (greedy takes more of item $j$). Since the greedy algorithm takes as much of item $j$ as possible before moving to item $j+1$, the remaining capacity in $S^*$ allocated to items $j, j+1, \ldots, n$ differs from $G$.

Construct $S'$ by increasing $x_j$ from $x_j^*$ toward $x_j^G$ by some amount $\delta$, and decreasing later items to maintain the capacity constraint. The change in value is:

$$
\Delta = \delta \cdot w_j \cdot r_j - \sum_{k > j} \delta_k \cdot w_k \cdot r_k
$$

Since $r_j \geq r_k$ for all $k > j$ and $\delta \cdot w_j = \sum_{k>j} \delta_k \cdot w_k$ (weight balance), we have $\Delta \geq 0$. So $S'$ is at least as good as $S^*$ and agrees with $G$ on one more item.

Repeating this process transforms $S^*$ into $G$ without decreasing value. $\square$

## 파이썬 구현

```python
"""
값어치 대 무게 비의 욕심쟁이 전략으로 푸는 쪼갤 수 있는 배낭.

동적 짜기가 필요한 0-1 배낭과 달리, 쪼갤 수 있는 판은
물건을 쪼개어 담을 수 있어 O(n log n) 욕심쟁이 풀이를 허락한다.
"""


# === 욕심쟁이 쪼갤 수 있는 배낭 ===

def fractional_knapsack(capacity, items):
    """쪼갤 수 있는 배낭 문제를 푼다.

    인수:
        capacity: 배낭이 담을 수 있는 최대 무게
        items: (무게, 값어치) 짝의 목록

    반환값:
        (최대 값어치, 비율)의 짝. 여기서 fractions[i]는
        물건 i를 담은 비율이다
    """
    n = len(items)
    # 비를 셈하고 비가 큰 차례로 정렬한다
    indexed = [(v / w, w, v, i) for i, (w, v) in enumerate(items)]
    indexed.sort(reverse=True)

    fractions = [0.0] * n
    total_value = 0.0
    remaining = capacity

    for ratio, weight, value, idx in indexed:
        if remaining <= 0:
            break
        if weight <= remaining:
            # 물건 전체를 담는다
            fractions[idx] = 1.0
            total_value += value
            remaining -= weight
        else:
            # 일부만 담는다
            fraction = remaining / weight
            fractions[idx] = fraction
            total_value += value * fraction
            remaining = 0

    return total_value, fractions


if __name__ == "__main__":
    # 보기: (무게, 값어치)
    items = [(10, 60), (20, 100), (30, 120)]
    capacity = 50

    max_val, fracs = fractional_knapsack(capacity, items)

    print("Fractional Knapsack Solution:")
    print(f"Capacity: {capacity}")
    print(f"{'Item':>5} {'Weight':>7} {'Value':>7} {'Ratio':>7} {'Taken':>7}")
    print("-" * 36)
    for i, (w, v) in enumerate(items):
        print(f"{i+1:>5} {w:>7} {v:>7} {v/w:>7.1f} {fracs[i]:>7.3f}")
    print(f"\nMaximum value: {max_val}")
```

**출력:**
```
Fractional Knapsack Solution:
Capacity: 50
 Item  Weight   Value   Ratio   Taken
------------------------------------
    1      10      60     6.0   1.000
    2      20     100     5.0   1.000
    3      30     120     4.0   0.667

Maximum value: 240.0
```

## 복잡도 분석

- **비 셈하기:** $O(n)$.
- **Sorting:** $O(n \log n)$.
- **배낭 채우기:** $O(n)$.
- **Total:** $O(n \log n)$.

**공간:** 몫 배열에 $O(n)$.

## 대비: 쪼갤 수 있는 배낭과 0-1 배낭

| 성질 | 쪼갤 수 있는 배낭 | 0-1 배낭 |
|----------|---------------------|--------------|
| 물건 쪼개기 | 된다 | 안 된다 |
| 알고리즘 | 욕심쟁이(비로) | 동적 계획 |
| Time complexity | $O(n \log n)$ | $O(nW)$ (pseudo-polynomial) |
| 욕심쟁이 고름 성질 | 성립한다 | 성립하지 않는다 |
| NP-어려움 | 아니다 | 그렇다 |

모든 0-1 풀이가 쪼갤 수 있는 문제에서도 될 수 있는 풀이이므로, 쪼갤 수 있는 변종은 늘 0-1 변종만큼은 높은 값을 얻는다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 16.2절. MIT Press.
- Kleinberg, J. & Tardos, E. (2006). *Algorithm Design*, 4장. Pearson.

## 연습문제

**연습문제 1.**
쪼갤 수 있는 배낭에서 욕심쟁이 고름이 무엇인지 가려내고 왜 가장 좋은 풀이로 이어지는지 밝혀라.

??? success "연습문제 1 풀이"
    The greedy choice selects the locally optimal option at each step. For Fractional Knapsack, this choice satisfies the greedy choice property: there exists an optimal solution that includes this greedy selection. Combined with optimal substructure (the remaining subproblem after the greedy choice is also optimally solvable by the same strategy), the greedy algorithm produces a globally optimal solution. $\square$

---

**연습문제 2.**
쪼갤 수 있는 배낭이 가장 좋은 아래 짜임을 갖는지 증명하거나 반증하여라.

??? success "연습문제 2 풀이"
    Optimal substructure means that an optimal solution to the problem contains optimal solutions to its subproblems. For Fractional Knapsack, after making the greedy choice, the remaining problem is a smaller instance of the same type. If the subproblem solution were not optimal, we could improve the overall solution by replacing it — contradicting overall optimality. Therefore optimal substructure holds. $\square$

---

**연습문제 3.**
쪼갤 수 있는 배낭의 시간 복잡도는 무엇인가? 가장 값비싼 단계를 가려내어라.

??? success "연습문제 3 풀이"
    The time complexity depends on the sorting step (if required) and the greedy selection loop. Sorting typically dominates at $O(n \log n)$. The greedy loop processes each element once in $O(n)$. Total: $O(n \log n)$. If the input is pre-sorted, the algorithm runs in $O(n)$. $\square$

---

**연습문제 4.**
(쪼갤 수 있는 배낭에서 쓴 것이 아닌) 다른 욕심쟁이 전략은 가장 좋은 풀이를 내지 못함을 보이는 반례를 들어라.

??? success "연습문제 4 풀이"
    Consider an alternative greedy criterion that does not align with the problem's structure. This alternative may select an element that blocks better future choices. The counterexample demonstrates that the wrong greedy criterion can produce a suboptimal result, highlighting why the specific greedy choice property must be proven for each problem. $\square$
