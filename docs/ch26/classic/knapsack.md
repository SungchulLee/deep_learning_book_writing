# 배낭 FPTAS

0/1 배낭 문제는 NP-어려움이므로 P = NP이 아니라면 다항식 시간의 정확한 알고리즘은 없다. 그러나 **온전 다항식 시간 어림 얼개(FPTAS)**를 쓰면 다항식 시간에 가장 좋은 값에 *얼마든지 가까이* 갈 수 있다. 핵심 생각은 아름답도록 단순하다. 곧 물건 값을 반올림해 동적 짜기 표의 크기를 줄이고, 정확함을 조금 잃는 대신 빠르기를 크게 얻는다.

## 문제 설정

무게가 $w_1, \dots, w_n$이고 값이 $v_1, \dots, v_n$인 물건 $n$개와 배낭 담이 $W$이 주어질 때 $\sum_{i \in S} w_i \le W$ 아래서 $\sum_{i \in S} v_i$을 가장 크게 하는 부분 모임 $S \subseteq \{1, \dots, n\}$을 찾아라.

정확한 동적 짜기 풀이는 값의 합을 어깨수로 삼는 표를 쓴다. $v_{\max} = \max_i v_i$이라 하자. 동적 짜기는 잣수를 맞춘 값에 걸쳐 도므로 표의 크기가 물건 값의 크기에 달려 비슷 다항식 시간 $O(n^2 v_{\max})$이 된다.

## FPTAS 알고리즘

**직관.** 모든 값을 어떤 알갱이 $K$의 배수로 내림하면 동적 짜기 표가 줄어든다. $K$을 꼼꼼히 고르면 반올림 어긋남이 가장 좋은 값의 $(1 - \epsilon)$ 갑절 안에 머문다.

**뜻매김.** 주어진 정확도 매개변수 $\epsilon > 0$에 대해 잣수 갑절을 다음과 같이 둔다.

$$
K = \frac{\epsilon \cdot v_{\max}}{n}
$$

그리고 잣수를 맞춘 값

$$
\hat{v}_i = \left\lfloor \frac{v_i}{K} \right\rfloor
$$

FPTAS은 값이 $\hat{v}_i$인 배낭 문제를 정확한 동적 짜기로 풀고 그에 맞물리는 본디 물건의 부분 모임을 돌려준다.

**알고리즘 걸음:**

1. $v_{\max} = \max_i v_i$과 $K = \epsilon \cdot v_{\max} / n$을 셈한다
2. 물건 $i$마다 $\hat{v}_i = \lfloor v_i / K \rfloor$으로 둔다
3. 값 $\hat{v}_i$과 무게 $w_i$으로 정확한 배낭 동적 짜기를 푼다
4. 고른 물건을 돌려준다(본디 값을 쓴다)

## 어림 보장

!!! tip "정리"
    FPTAS은 값이 적어도 $(1 - \epsilon) \cdot \text{OPT}$인 풀이를 돌려준다.

**밝힘.** $S^*$을 값이 $\text{OPT} = \sum_{i \in S^*} v_i$인 가장 좋은 풀이라 하자. $\hat{S}$을 잣수를 맞춘 값으로 FPTAS이 찾은 풀이라 하자. $\hat{S}$은 잣수를 맞춘 문제에서 가장 좋으므로,

$$
\sum_{i \in \hat{S}} \hat{v}_i \ge \sum_{i \in S^*} \hat{v}_i
$$

내림 함수에 따라 물건마다 $\hat{v}_i \ge v_i / K - 1$이므로,

$$
\sum_{i \in S^*} \hat{v}_i \ge \sum_{i \in S^*} \frac{v_i}{K} - |S^*|
\ge \frac{\text{OPT}}{K} - n
$$

$\hat{S}$은 잣수를 맞춘 값에서 가장 좋으므로 본디 값의 합은 다음을 채운다.

$$
\sum_{i \in \hat{S}} v_i \ge K \cdot \sum_{i \in \hat{S}} \hat{v}_i
\ge K \left(\frac{\text{OPT}}{K} - n\right)
= \text{OPT} - nK = \text{OPT} - \epsilon \cdot v_{\max}
$$

$v_{\max} \le \text{OPT}$이므로 다음을 얻는다.

$$
\sum_{i \in \hat{S}} v_i \ge \text{OPT} - \epsilon \cdot \text{OPT}
= (1 - \epsilon) \cdot \text{OPT} \qquad \square
$$

## 도는 시간

잣수를 맞춘 값마다 $\hat{v}_i \le v_i / K \le v_{\max} / K = n / \epsilon$이다. 동적 짜기 표는 가로줄이 $n$개이고 세로줄이 많아야 $n \cdot (n / \epsilon) = n^2 / \epsilon$개이므로 온 도는 시간은 다음과 같다.

$$
O\!\left(\frac{n^3}{\epsilon}\right)
$$

이는 $n$과 $1/\epsilon$ 모두에 대해 다항식이며 이것이 바로 FPTAS의 뜻매김이다.

## 구현

```python
"""
배낭 FPTAS: O(n^3 / epsilon) 시간의 (1-epsilon) 어림.
"""


# === 정확한 동적 짜기(값을 어깨수로) ========================================

def knapsack_exact(W, weights, values):
    """무게를 어깨수로 삼은 동적 짜기로 푸는 정확한 0/1 배낭."""
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(W + 1):
            dp[i][w] = dp[i - 1][w]
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
    return dp[n][W]


# === FPTAS ===================================================================

def knapsack_fptas(W, weights, values, epsilon):
    """
    0/1 배낭의 FPTAS.

    값이 (1 - epsilon) * OPT 이상인 풀이를 돌려준다.
    O(n^3 / epsilon) 시간에 돈다.
    """
    n = len(weights)
    if n == 0:
        return 0, []

    v_max = max(values)
    K = (epsilon * v_max) / n

    # 값의 잣수를 낮춘다
    scaled = [int(v // K) for v in values]
    V = sum(scaled)

    # 동적 짜기: 잣수를 맞춘 값 합마다 이루는 최소 무게
    INF = float("inf")
    dp = [INF] * (V + 1)
    dp[0] = 0
    parent = [[] for _ in range(V + 1)]

    for i in range(n):
        for v in range(V, scaled[i] - 1, -1):
            if dp[v - scaled[i]] + weights[i] <= W:
                new_w = dp[v - scaled[i]] + weights[i]
                if new_w < dp[v]:
                    dp[v] = new_w
                    parent[v] = parent[v - scaled[i]] + [i]

    # 이룰 수 있는 가장 좋은 잣수 값을 찾는다
    best_v = 0
    for v in range(V + 1):
        if dp[v] < INF:
            best_v = v

    selected = parent[best_v]
    total = sum(values[i] for i in selected)
    return total, selected


# === 보여 주기 ===============================================================

if __name__ == "__main__":
    W = 50
    weights = [10, 20, 30]
    values = [60, 100, 120]

    exact = knapsack_exact(W, weights, values)
    print(f"Exact DP:  {exact}")

    epsilon = 0.1
    approx, items = knapsack_fptas(W, weights, values, epsilon)
    print(f"FPTAS (ε={epsilon}): value={approx}, items={items}")
    print(f"Guarantee: >= {(1 - epsilon) * exact:.1f}")
```

**출력:**
```
정확한 동적 짜기:  220
FPTAS (ε=0.1): value=220, items=[1, 2]
Guarantee: >= 198.0
```

## FPTAS이 중요한 까닭

| 성질 | 정확한 동적 짜기 | FPTAS |
|---|---|---|
| 시간 | $O(nW)$이나 $O(n \cdot v_{\max} \cdot n)$ | $O(n^3 / \epsilon)$ |
| 품질 | 가장 좋음 | $\ge (1 - \epsilon) \cdot \text{OPT}$ |
| 갈래 | 비슷 다항식 | 온전 다항식 |

FPTAS은 어림 이론에서 가장 센 긍정 결과 가운데 하나이다. 곧 0/1 배낭이 NP-어려움이지만 어림하기는 어렵지 않음을 보인다. 모든 NP-어려운 문제에 FPTAS이 있는 것은 아니다. 예컨대 일반 떠돌이 장수 문제는 P = NP이 아니라면 없다.

## 참고 문헌

- Vazirani, V. V. *Approximation Algorithms*. Springer, 2001. Chapter 8.
- Ibarra, O. H. and Kim, C. E. "Fast Approximation Algorithms for the Knapsack and Sum of Subset Problems." *JACM*, 1975.

## 연습문제

**연습문제 1.**
배낭 FPTAS의 어림 알고리즘을 설명하고 그 어림 보장을 밝혀라.

??? success "연습문제 1 풀이"
    이 알고리즘은 다항식 시간에 돌며 가장 좋은 값의 밝힐 수 있는 갑절 안에 드는 풀이를 낸다. 어림 비율은 알고리즘이 내놓은 것을 가장 좋은 값의 아래 한계(가장 작게 하기)나 위 한계(가장 크게 하기), 곧 선형 계획 느슨하게 하기 값이나 조합 한계, 문제의 짜임 성질과 이어 밝힌다. $\square$

---

**연습문제 2.**
배낭 FPTAS의 어림 비율을 밝히는 데 어떤 아래 한계 재주를 쓰는가?

??? success "연습문제 2 풀이"
    밝힘은 흔히 알고리즘의 풀이를 느슨하게 한 한계(선형 계획 느슨하게 하기, 분수 풀이, 조합 아래 한계)와 견준다. 가장 작게 하기에서는 $ALG \leq \rho \cdot LP^* \leq \rho \cdot OPT$이다. 가장 크게 하기에서는 $ALG \geq OPT / \rho$이다. 아래 한계는 효율 좋게 셈할 수 있고 쓸모 있는 비율을 줄 만큼 빡빡해야 한다. $\square$

---

**연습문제 3.**
배낭 FPTAS의 어림 비율을 더 좋게 할 수 있는가? 알려진 어려움 결과는 무엇인가?

??? success "연습문제 3 풀이"
    어림 비율이 얼마나 빡빡한지는 복잡도 이론의 가정(P $\neq$ NP, 하나뿐인 놀이 추측 등)에 달렸다. 어떤 문제에서는 단순한 욕심쟁이나 반올림 알고리즘이 여느 가정 아래 이미 가장 좋다. 다른 문제에서는 가장 좋은 알고리즘과 가장 센 어려움 결과 사이에 틈이 있어 아직 풀리지 않은 연구 문제로 남아 있다. $\square$

---

**연습문제 4.**
배낭 FPTAS을 구체적인 보기에 써서 어림 비율이 참임을 확인하라.

??? success "연습문제 4 풀이"
    작은 보기(예컨대 꼭짓점이나 물건 5~6개)를 고른다. 어림 알고리즘을 한 걸음씩 돌린다. 알고리즘이 내놓은 것을 (작은 보기에서 막무가내로 찾은) 가장 좋은 풀이와 견준다. 비율 $ALG/OPT$(또는 $OPT/ALG$)이 밝힌 한계 안에 드는지 확인한다. 그러면 구체적인 보기에서 이론이 굳어진다. $\square$
