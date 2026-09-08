# 0/1 배낭

0/1 배낭 문제는 조합 가장 좋게 하기에서 가장 중요한 문제 가운데 하나이다. 무게와 값어치를 가진 물건 모임이 주어질 때 무게 담이를 넘지 않으면서 전체 값어치를 가장 크게 하는 부분 모임을 고르는 것이 일이다. "0/1"은 물건마다 전부 담거나 아예 두고 가야 한다는, 곧 쪼개어 고를 수 없다는 제약을 뜻한다. 이 문제는 한 차원이 물건, 다른 차원이 무게 담이인 2차원 동적 짜기를 보여 준다.

---

## 1. 문제 서술

물건 $i$의 무게가 $w_i$, 값어치가 $v_i$인 물건 $n$개와 무게 담이 $W$인 배낭이 주어질 때 다음을 찾아라

$$
\max \sum_{i=1}^{n} v_i x_i \quad \text{subject to} \quad \sum_{i=1}^{n} w_i x_i \le W, \quad x_i \in \{0, 1\}
$$

---

## 2. 점화식

$dp[i][w]$을 무게 담이 $w$으로 물건 $1$부터 $i$까지 써서 얻을 수 있는 최대 값어치라 하자. 물건 $i$마다 고름이 둘이다:

1. 물건 $i$을 **뺀다**: 값어치는 $dp[i-1][w]$이다.
2. 물건 $i$을 **넣는다**($w_i \le w$일 때만): 값어치는 $dp[i-1][w - w_i] + v_i$이다.

더 나은 쪽을 취하면:

$$
dp[i][w] = \begin{cases} dp[i-1][w] & \text{if } w_i > w \\ \max\bigl(dp[i-1][w],\; dp[i-1][w - w_i] + v_i\bigr) & \text{if } w_i \le w \end{cases}
$$

바탕 경우는 모든 $w$에 대해 $dp[0][w] = 0$이다(물건이 없으면 값어치도 0).

---

## 3. 풀이 예제

무게가 $[2, 3, 4, 5]$, 값어치가 $[3, 4, 5, 6]$이고 담이가 $W = 8$인 경우를 보자.

동적 짜기 표를 가로줄마다 채운다. 칸 $dp[i][w]$은 담이 $w$으로 물건 $1 \ldots i$을 써서 얻는 가장 좋은 값어치를 뜻한다:

| $dp[i][w]$ | $w=0$ | $w=1$ | $w=2$ | $w=3$ | $w=4$ | $w=5$ | $w=6$ | $w=7$ | $w=8$ |
|---|---|---|---|---|---|---|---|---|---|
| $i=0$ | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| $i=1$ ($w_1\!=\!2, v_1\!=\!3$) | 0 | 0 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| $i=2$ ($w_2\!=\!3, v_2\!=\!4$) | 0 | 0 | 3 | 4 | 4 | 7 | 7 | 7 | 7 |
| $i=3$ ($w_3\!=\!4, v_3\!=\!5$) | 0 | 0 | 3 | 4 | 5 | 7 | 8 | 9 | 9 |
| $i=4$ ($w_4\!=\!5, v_4\!=\!6$) | 0 | 0 | 3 | 4 | 5 | 7 | 8 | 9 | **10** |

가장 좋은 값어치는 $dp[4][8] = 10$이다. 거슬러 좇으면 $dp[4][8] \ne dp[3][8]$이므로 물건 4를 골랐고($w = 8 - 5 = 3$), 이어 $dp[2][3] \ne dp[1][3]$이므로 물건 2를 골랐다. 가장 좋은 부분 모임은 물건 $\{2, 4\}$이며 전체 무게는 $3 + 5 = 8$, 전체 값어치는 $4 + 6 = 10$이다.

---

## 4. 표 채우기

아래 짜기는 온전한 2차원 표를 세우고, 1차원 배열로 공간을 줄이며, 풀이 다시 세우기를 준다.

```python
"""
0/1 배낭 — 동적 짜기.

세 방식: 2차원 표 채우기, 1차원 공간 줄임, 다시 세우기.
"""

# === 2차원 표 채우기 ===

def knapsack_2d(weights: list[int], values: list[int], capacity: int) -> int:
    """2차원 표를 쓴 0/1 배낭. 시간: O(nW), 공간: O(nW)."""
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i - 1][w]
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])

    return dp[n][capacity]

# === 1차원 공간 줄이기 ===

def knapsack_1d(weights: list[int], values: list[int], capacity: int) -> int:
    """1차원 배열을 쓴 0/1 배낭. 시간: O(nW), 공간: O(W)."""
    dp = [0] * (capacity + 1)

    for i in range(len(weights)):
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

    return dp[capacity]

# === 다시 세우기 ===

def knapsack_with_items(
    weights: list[int], values: list[int], capacity: int
) -> tuple[int, list[int]]:
    """최대 값어치와 고른 물건의 번호를 돌려준다."""
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i - 1][w]
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])

    # 표를 거슬러 좇아 고른 물건을 찾는다
    selected = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected.append(i - 1)
            w -= weights[i - 1]

    return dp[n][capacity], list(reversed(selected))

# === 메인 ===

if __name__ == "__main__":
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 8

    max_val = knapsack_2d(weights, values, capacity)
    max_val_1d = knapsack_1d(weights, values, capacity)
    max_val_items, items = knapsack_with_items(weights, values, capacity)

    print(f"Weights: {weights}")
    print(f"Values:  {values}")
    print(f"Capacity: {capacity}")
    print(f"Max value (2D): {max_val}")
    print(f"Max value (1D): {max_val_1d}")
    print(f"Max value: {max_val_items}, items: {items}")
    # 내임:
    # 무게: [2, 3, 4, 5]
    # 값어치:  [3, 4, 5, 6]
    # 담이: 8
    # 최대 값어치(2차원): 10
    # 최대 값어치(1차원): 10
    # 최대 값어치: 10, 물건: [1, 3]
```

---

## 5. 공간 줄이기

가로줄 $i$이 가로줄 $i-1$에만 기대므로 1차원 배열 하나면 넉넉하다. 핵심 눈썰미는 무게를 **거꾸로**($W$에서 $w_i$까지) 처리하는 것이다. 그러면 물건을 두 번 쓰지 않는다. 곧 물건 $i$에 대해 $dp[w]$을 셈할 때 $dp[w - w_i]$이 여전히 물건 $i$을 *쓰지 않은* 상태를 담고 있다.

무게를 앞으로 가는 차례로 처리하면 $dp[w - w_i]$이 이미 물건 $i$을 담고 있을 수 있어 사실상 얼마든지 담을 수 있게 된다(그러면 한정 없는 배낭 변형을 푸는 셈이다).

---

## 6. 복잡도

| 갈래 | 값 |
|---|---|
| 시간 | $O(nW)$ — 유사 다항 |
| 공간(2차원) | $O(nW)$ |
| 공간(1차원) | $O(W)$ |
| 아래 문제 | $(n+1)(W+1)$ |

!!! warning "유사 다항 복잡도"
    $O(nW)$이 다항처럼 보이지만 $W$의 들임 크기는 $\log W$ 비트이다. 이 알고리즘은 $W$의 들임 크기에 대해 지수이며 그래서 배낭 문제가 NP 어려움으로 남는다. $W$이 크면 어림 알고리즘이나 가지 뻗어 묶기가 더 현실적일 수 있다.

---

## 연습문제

**연습문제 1.**
담이가 $W = 7$이고 물건이 $(w, v) = \{(1,1), (3,4), (4,5), (5,7)\}$인 0/1 배낭 문제를 풀어라.

??? success "연습문제 1 풀이"
    담이 $w$으로 앞선 $i$개 물건을 써서 얻는 최대 값어치인 동적 짜기 표 $dp[i][w]$을 세운다. 물건 4 $(5,7)$에 대해 $dp[4][7] = \max(dp[3][7], dp[3][2] + 7) = \max(9, 1+7) = 9$이다. 거슬러 좇으면 물건 2와 3이다(무게 3+4=7, 값어치 4+5=9). 가장 좋은 값어치는 9이다. $\square$

---

**연습문제 2.**
0/1 배낭 문제가 가장 좋은 아래 짜임을 갖춤을 밝혀라.

??? success "연습문제 2 풀이"
    물건 $n$개와 담이 $W$에 대한 가장 좋은 풀이 $S^*$을 보자. 물건 $n \in S^*$이면 $S^* \setminus \{n\}$은 물건 $1, \ldots, n-1$과 담이 $W - w_n$에 대한 가장 좋은 풀이여야 한다(아니라면 더 나은 아래 풀이가 $S^*$을 낫게 할 것이다). 물건 $n \notin S^*$이면 $S^*$은 물건 $1, \ldots, n-1$과 담이 $W$에 대해 가장 좋다. 이로써 되돌이 관계식 $dp[i][w] = \max(dp[i-1][w], dp[i-1][w-w_i] + v_i)$을 얻는다. $\square$

---

**연습문제 3.**
여느 0/1 배낭 동적 짜기의 시간 복잡도와 공간 복잡도는 무엇인가? 공간을 어떻게 줄이는가?

??? success "연습문제 3 풀이"
    여느 것: 2차원 표에 $O(nW)$ 시간과 $O(nW)$ 공간. 공간 줄이기: $dp[i]$이 $dp[i-1]$에만 기대므로 1차원 배열을 쓰고 $w$을 $W$에서 $w_i$까지 거꾸로 되풀이한다(지금 가로줄에서 새로 고친 값을 쓰지 않으려고). 그러면 시간은 $O(nW)$을 지키면서 공간이 $O(W)$으로 준다. $\square$

---

**연습문제 4.**
$O(nW)$ 알고리즘이 있는데도 배낭 문제가 왜 NP 어려움인가? 이것이 왜 유사 다항인지 설명하라.

??? success "연습문제 4 풀이"
    복잡도 $O(nW)$은 $n$과 $W$에 대해 다항이지만 $W$은 들임 크기에 대해 지수만큼 클 수 있다. 들임은 $W$을 $\log W$ 비트로 담으므로 $W = 2^{\log W}$이다. 이 알고리즘은 들임 길이인 $\log W$에 대해 다항이 아니라 지수이다. 알고리즘이 참으로 다항이려면 도는 시간이 들임 길이에 대해 다항이어야 한다. 배낭 문제를 $n + \log W$에 대해 다항 시간에 푸는 알고리즘이 알려져 있지 않으므로 NP 어려움으로 남는다. $\square$

## 정리하며

이 마당은 문제 서술、점화식、풀이 예제、표 채우기을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 15. MIT Press.
