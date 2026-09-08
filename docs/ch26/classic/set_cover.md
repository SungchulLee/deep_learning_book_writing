# 모임 덮기 어림

**모임 덮기** 문제는 가장 바탕이 되는 NP-어려운 가장 좋게 하기 문제 가운데 하나이다. 어렵지만 단순한 욕심쟁이 알고리즘이 $O(\ln n)$ 어림 비율을 이루며, 이는 사실상 어떤 다항식 시간 알고리즘도 넘지 못하는 수준이다.

---

## 1. 문제의 정의

온 모임 $U = \{1, 2, \dots, n\}$과 모임 묶음 $\mathcal{S} = \{S_1, S_2, \dots, S_m\}$이 주어지고 $S_j \subseteq U$이며 모임마다 비용 $c_j > 0$이 있을 때 다음을 채우는 아래 묶음 $\mathcal{C} \subseteq \mathcal{S}$을 찾아라.

$$
\bigcup_{S_j \in \mathcal{C}} S_j = U
$$

그리고 온 비용 $\sum_{S_j \in \mathcal{C}} c_j$이 가장 작다.

---

## 2. 욕심쟁이 알고리즘

**직관.** 걸음마다 *새로 덮는 낱개마다 비용*이 가장 낮은 모임을 고른다. 이 비용 대비 효과 잣대는 보태는 것에 견주어 헤픈 모임이 골리지 않게 한다.

**알고리즘:**

1. $R \leftarrow U$으로 첫자리매김한다(덮이지 않은 낱개)
2. $R \neq \emptyset$인 동안:
    - $S_j$마다 비용 대비 효과 $\frac{c_j}{|S_j \cap R|}$을 셈한다
    - 이 비율을 가장 작게 하는 $S_j$을 고른다
    - $S_j$을 $\mathcal{C}$에 더하고 $R \leftarrow R \setminus S_j$으로 고친다
3. $\mathcal{C}$을 돌려준다

---

## 3. 풀이 예제

비용이 1인 모임 $S_1 = \{1, 2, 3\}$, $S_2 = \{2, 4, 5\}$, $S_3 = \{3, 5, 6\}$, $S_4 = \{1, 4, 6\}$과 함께 $U = \{1, 2, 3, 4, 5, 6\}$을 보자.

| 걸음 | $R$ | 가장 좋은 모임 | 비용 대비 효과 | 고른 것 |
|------|-----|----------|-------------------|--------|
| 1 | $\{1,2,3,4,5,6\}$ | 모두 비용 1으로 낱개 3개를 덮는다 | $1/3$ | $S_1$ |
| 2 | $\{4,5,6\}$ | $S_2$: 새 2개, $S_3$: 새 2개, $S_4$: 새 2개 | $1/2$ | $S_2$ |
| 3 | $\{6\}$ | $S_3$: 새 1개, $S_4$: 새 1개 | $1/1$ | $S_3$ |

욕심쟁이는 온 비용 $3$인 $\{S_1, S_2, S_3\}$을 고른다. 가장 좋은 풀이는 $\{S_1, S_2, S_3\}$이나 $\{S_2, S_4, S_1\}$이며 비용도 $3$이다. 이 경우 욕심쟁이 풀이가 가장 좋다.

---

## 4. 어림 보장

!!! tip "정리"
    욕심쟁이 알고리즘은 어림 비율 $H_n$을 이루며 여기서 $H_n = \sum_{k=1}^n \frac{1}{k} \le \ln n + 1$은 $n$번째 조화수이다.

**밝힘.** 낱개 $e$이 덮일 때 *값*을 매긴다. 곧 모임 $S_j$이 골려 새 낱개 $k$개를 덮으면 새 낱개마다 $c_j / k$을 낸다.

$\text{OPT}$을 가장 좋은 비용이라 하자. 걸음마다 남은 낱개를 온 비용 많아야 $\text{OPT}$으로 덮을 수 있다(가장 좋은 풀이가 모두 덮으므로). 낱개가 $|R|$개 남았다면 가장 좋은 풀이의 어떤 모임이 그 가운데 적어도 $|R| / |\mathcal{S}^*|$개를 비용 대비 효과 많아야 $\text{OPT} / |R|$으로 덮는다. 욕심쟁이 고르기는 적어도 그만큼 좋다.

더 자세히는 $n_t$을 걸음 $t$ 뒤 덮이지 않은 낱개의 수라 하자. $n_0 = n$으로 두면 걸음 $t + 1$에서 욕심쟁이 고르기의 비용 대비 효과는 많아야 $\text{OPT} / n_t$이다. 온 욕심쟁이 비용은 다음과 같다.

$$
\sum_{t=0}^{T-1} \frac{\text{OPT}}{n_t} \cdot (n_t - n_{t+1})
\le \text{OPT} \sum_{k=1}^{n} \frac{1}{k} = H_n \cdot \text{OPT}
$$

이 부등식은 낱개 $n_t - n_{t+1}$개를 저마다 값 $\text{OPT}/n_t$으로 덮는 것이 많아야 $\text{OPT} \cdot \sum_{j=n_{t+1}+1}^{n_t} 1/j$이므로 따라 나온다. 망원경처럼 더하면 $H_n$이 된다. $\square$

---

## 5. 어림할 수 없음

Dinur와 Steurer(2014)는 P = NP가 아니라면 모임 덮기를 어떤 $\epsilon > 0$에 대해서도 $(1 - \epsilon) \ln n$ 안으로 어림할 수 없음을 보였다. 따라서 욕심쟁이 알고리즘은 사실상 가장 좋다.

---

## 6. 구현

```python
"""
모임 덮기: 욕심쟁이 H_n 어림 알고리즘.
"""

# === 욕심쟁이 모임 덮기 =======================================================

def greedy_set_cover(universe, sets, costs):
    """
    비용 대비 효과 잣대를 쓴 욕심쟁이 모임 덮기.

    인수:
        universe: 덮을 낱개의 모임.
        sets: 모임의 목록.
        costs: 모임마다 비용의 목록.

    반환값:
        (온 비용, 고른 어깨수).
    Approximation ratio: H_n = O(ln n).
    """
    remaining = set(universe)
    selected = []
    total_cost = 0.0

    while remaining:
        # 비용 대비 효과가 가장 좋은 모임을 찾는다
        best_idx = -1
        best_ratio = float("inf")
        for j, s in enumerate(sets):
            new_covered = len(s & remaining)
            if new_covered > 0:
                ratio = costs[j] / new_covered
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_idx = j

        if best_idx == -1:
            break  # 남은 낱개를 덮을 수 없다

        selected.append(best_idx)
        total_cost += costs[best_idx]
        remaining -= sets[best_idx]

    return total_cost, selected

# === 조화수 ==================================================================

def harmonic(n):
    """Compute H_n = 1 + 1/2 + ... + 1/n."""
    return sum(1.0 / k for k in range(1, n + 1))

# === 보여 주기 ===============================================================

if __name__ == "__main__":
    universe = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    sets = [
        {1, 2, 3, 4},
        {3, 4, 5, 6},
        {5, 6, 7, 8},
        {7, 8, 9, 10},
        {1, 5, 9},
        {2, 6, 10},
    ]
    costs = [4, 4, 4, 4, 3, 3]

    cost, selected = greedy_set_cover(universe, sets, costs)
    n = len(universe)
    print(f"Selected sets: {selected}")
    print(f"Total cost:    {cost}")
    print(f"H_{n} = {harmonic(n):.3f}")
    print(f"Guarantee:     <= {harmonic(n) * 8:.1f} (if OPT=8)")
```

---

## 연습문제

**연습문제 1.**
모임 덮기 어림의 어림 알고리즘을 설명하고 그 어림 보장을 밝혀라.

??? success "연습문제 1 풀이"
    이 알고리즘은 다항식 시간에 돌며 가장 좋은 값의 밝힐 수 있는 갑절 안에 드는 풀이를 낸다. 어림 비율은 알고리즘이 내놓은 것을 가장 좋은 값의 아래 한계(가장 작게 하기)나 위 한계(가장 크게 하기), 곧 선형 계획 느슨하게 하기 값이나 조합 한계, 문제의 짜임 성질과 이어 밝힌다. $\square$

---

**연습문제 2.**
모임 덮기 어림의 어림 비율을 밝히는 데 어떤 아래 한계 재주를 쓰는가?

??? success "연습문제 2 풀이"
    밝힘은 흔히 알고리즘의 풀이를 느슨하게 한 한계(선형 계획 느슨하게 하기, 분수 풀이, 조합 아래 한계)와 견준다. 가장 작게 하기에서는 $ALG \leq \rho \cdot LP^* \leq \rho \cdot OPT$이다. 가장 크게 하기에서는 $ALG \geq OPT / \rho$이다. 아래 한계는 효율 좋게 셈할 수 있고 쓸모 있는 비율을 줄 만큼 빡빡해야 한다. $\square$

---

**연습문제 3.**
모임 덮기 어림의 어림 비율을 더 좋게 할 수 있는가? 알려진 어려움 결과는 무엇인가?

??? success "연습문제 3 풀이"
    어림 비율이 얼마나 빡빡한지는 복잡도 이론의 가정(P $\neq$ NP, 하나뿐인 놀이 추측 등)에 달렸다. 어떤 문제에서는 단순한 욕심쟁이나 반올림 알고리즘이 여느 가정 아래 이미 가장 좋다. 다른 문제에서는 가장 좋은 알고리즘과 가장 센 어려움 결과 사이에 틈이 있어 아직 풀리지 않은 연구 문제로 남아 있다. $\square$

---

**연습문제 4.**
모임 덮기 어림을 구체적인 보기에 써서 어림 비율이 참임을 확인하라.

??? success "연습문제 4 풀이"
    작은 보기(예컨대 꼭짓점이나 물건 5~6개)를 고른다. 어림 알고리즘을 한 걸음씩 돌린다. 알고리즘이 내놓은 것을 (작은 보기에서 막무가내로 찾은) 가장 좋은 풀이와 견준다. 비율 $ALG/OPT$(또는 $OPT/ALG$)이 밝힌 한계 안에 드는지 확인한다. 그러면 구체적인 보기에서 이론이 굳어진다. $\square$

## 정리하며

| 성질 | 값 |
|---|---|
| 어림 비율 | $H_n \le \ln n + 1$ |
| 시간 복잡도 | 되풀이마다 $O(n \cdot m)$, 모두 $O(n^2 m)$ |
| 빡빡한가? | 그렇다 — $(1 - \epsilon)\ln n$보다 잘할 수 없다 |

**참고 문헌**

- Chvatal, V. "A Greedy Heuristic for the Set-Covering Problem." *Math. of Operations Research*, 1979.
- Dinur, I. and Steurer, D. "Analytical Approach to Parallel Repetition." *STOC*, 2014.
- Vazirani, V. V. *Approximation Algorithms*. Springer, 2001. Chapter 2.
