# PTAS과 FPTAS

어떤 NP-어려운 가장 좋게 하기 문제에서는 도는 시간이 늘어나는 대가로 가장 좋은 값에 얼마든지 가까이 갈 수 있다. **다항식 시간 어림 얼개(PTAS)**는 바라는 정확도 $\epsilon > 0$을 마음대로 고르게 해 주고, **온전 다항식 시간 어림 얼개(FPTAS)**는 도는 시간이 $1/\epsilon$에 대해서도 다항식으로 커짐을 보장한다. 이 쪽은 이 개념을 뜻매김하고 어림의 층층 짜임과 이어 보며 배낭 FPTAS으로 보인다.

## PTAS 뜻매김

!!! tip "뜻매김: PTAS"
    가장 작게 하기 문제 $\Pi$의 **다항식 시간 어림 얼개**는 모든 $\epsilon > 0$과 모든 보기 $I$에 대해 다음을 채우는 알고리즘 무리 $\{A_\epsilon\}_{\epsilon > 0}$이다.

    $$
    A_\epsilon(I) \leq (1 + \epsilon) \cdot \text{OPT}(I)
    $$

    그리고 $A_\epsilon$은 $|I|$에 대해 다항식 시간에 돈다(다만 $1/\epsilon$에 대해서는 그렇지 않을 수 있다).

가장 크게 하기 문제에서는 보장이 $A_\epsilon(I) \geq (1 - \epsilon) \cdot \text{OPT}(I)$이 된다.

도는 시간은 $O(n^{1/\epsilon})$이나 $O(n^{2^{1/\epsilon}})$일 수 있다. 곧 붙박인 $\epsilon$마다 $n$에 대해 다항식이지만 $\epsilon$이 작으면 쓸 수 없을 수 있다.

## FPTAS 뜻매김

!!! tip "뜻매김: FPTAS"
    **온전 다항식 시간 어림 얼개**는 도는 시간이 $|I|$과 $1/\epsilon$ 모두에 대해 다항식인 PTAS이다.

흔한 FPTAS의 도는 시간은 $O(n^2 / \epsilon)$이나 $O(n^3 / \epsilon^2)$ 같다. FPTAS은 가장 센 갈래의 어림 결과이다. 곧 바라는 어떤 정확도에서도 알고리즘이 쓸 만한 다항식 시간에 돈다.

## 어림의 층층 짜임

센 것부터 여린 것까지의 담김 짜임:

$$
\text{FPTAS} \subset \text{PTAS} \subset \text{APX} \subset \text{NPO}
$$

- **FPTAS:** $n$과 $1/\epsilon$에 대해 다항식이다.
- **PTAS:** 붙박인 $\epsilon$마다 $n$에 대해 다항식이지만 $1/\epsilon$에 대해서는 지수일 수 있다.
- **APX:** 어떤 붙박인 비율의 상수 갑절 어림을 받아들인다.
- **NPO:** 모든 NP 가장 좋게 하기 문제의 갈래이다.

| Class | Example Problems |
|-------|-----------------|
| FPTAS | Knapsack, Scheduling on identical machines |
| PTAS (not FPTAS) | Euclidean TSP, Bin Packing |
| APX (not PTAS) | MAX-3SAT, Vertex Cover, Metric TSP |
| NPO (not APX) | Clique, Chromatic Number |

## FPTAS for Knapsack

The 0/1 Knapsack problem is NP-hard, yet it admits an FPTAS via a **scaling and rounding** technique applied to the exact dynamic programming solution.

### 준비

Given $n$ items with values $v_1, \ldots, v_n$ and weights $w_1, \ldots, w_n$, and a capacity $W$, maximize $\sum_{i \in S} v_i$ subject to $\sum_{i \in S} w_i \leq W$.

The exact DP runs in $O(n \cdot V)$ time where $V = \sum_i v_i$ --- pseudo-polynomial, not polynomial in the input size.

### Scaling Strategy

The idea is to round down the values to reduce $V$ while controlling the error.

1. Let $v_{\max} = \max_i v_i$.
2. Set the scaling factor $K = \frac{\epsilon \cdot v_{\max}}{n}$.
3. Define scaled values $\hat{v}_i = \lfloor v_i / K \rfloor$ for each item $i$.
4. 잣수를 맞춘 보기 $(\hat{v}_1, \ldots, \hat{v}_n, w_1, \ldots, w_n, W)$에 정확한 동적 짜기를 돌린다.
5. 찾은 풀이를 돌려준다.

### 살피기

**도는 시간.** 잣수를 맞춘 최대 값은 다음과 같다.

$$
\hat{v}_{\max} = \left\lfloor \frac{v_{\max}}{K} \right\rfloor = \left\lfloor \frac{n}{\epsilon} \right\rfloor
$$

동적 짜기는 $O(n \cdot n \cdot \hat{v}_{\max}) = O(n^3 / \epsilon)$에 돌며 이는 $n$과 $1/\epsilon$ 모두에 대해 다항식이다.

**어림 비율.** $S^*$을 가장 좋은 풀이, $\hat{S}$을 잣수를 맞춘 동적 짜기가 찾은 풀이라 하자.

물건 $i$마다 반올림 어긋남은 $v_i - K \hat{v}_i < K$을 채운다. $S^*$의 물건에 걸쳐 더하면:

$$
\sum_{i \in S^*} v_i - K \sum_{i \in S^*} \hat{v}_i < n \cdot K = \epsilon \cdot v_{\max}
$$

$\hat{S}$은 잣수를 맞춘 보기에서 가장 좋으므로 $\sum_{i \in \hat{S}} \hat{v}_i \geq \sum_{i \in S^*} \hat{v}_i$이며, 따라서:

$$
\sum_{i \in \hat{S}} v_i \geq K \sum_{i \in \hat{S}} \hat{v}_i \geq K \sum_{i \in S^*} \hat{v}_i > \sum_{i \in S^*} v_i - \epsilon \cdot v_{\max} \geq (1 - \epsilon) \cdot \text{OPT}
$$

마지막 부등식은 $\text{OPT} \geq v_{\max}$을 쓴다. $\square$

## 유클리드 떠돌이 장수 문제의 PTAS

Arora(1998)와 Mitchell(1999)은 서로 매이지 않게 유클리드 공간의 떠돌이 장수 문제가 PTAS을 받아들임을 보였다. 유클리드 거리를 지닌 $\mathbb{R}^2$의 점 $n$개가 주어지면 아무 $\epsilon > 0$에 대해 길이가 많아야 $(1 + \epsilon) \cdot \text{OPT}$인 나들이를 $n \cdot (\log n)^{O(1/\epsilon)}$ 시간에 내는 알고리즘이 있다.

핵심 생각은 **아무렇게나 옮긴 네 갈래 나무** 나누기를 쓴다. 이 알고리즘은:

1. 모든 점을 감싸는 정사각형에 담는다.
2. 아무렇게나 옮기고 되돌이로 사분면으로 나눈다.
3. 층마다 가로지름을 가둔 수의 "문"으로 제한한다.
4. 그렇게 나온 짜임 있는 아래 문제를 동적 짜기로 푼다.

이는 PTAS이지만 FPTAS은 아니다. 도는 시간이 $1/\epsilon$에 대해 지수이다.

## FPTAS이 있을 수 없을 때

PTAS이 있는 모든 문제에 FPTAS이 있는 것은 아니다. 여느 복잡도 가정 아래서는:

!!! warning "정리"
    **강하게 NP-어려운** 문제에 FPTAS이 있으면 P = NP이다.

들임의 모든 수가 $n$의 다항식으로 가둬져도 여전히 NP-어려우면 그 문제는 강하게 NP-어렵다. 보기로 통 담기와 3-나누기가 있다. FPTAS이 있으면 비슷 다항식 동적 짜기를 참으로 다항식 시간에 풀게 되어 강하게 NP-어려운 보기를 풀어 버린다.

??? example "풀어 본 보기: 배낭 FPTAS"
    **보기:** 물건 4개, 담이 $W = 10$.

    | 물건 | 값 | 무게 |
    |------|-------|--------|
    | 1    | 100   | 5      |
    | 2    | 60    | 3      |
    | 3    | 120   | 7      |
    | 4    | 80    | 4      |

    $\epsilon = 0.2$으로 두자. 그러면 $v_{\max} = 120$, $K = \frac{0.2 \times 120}{4} = 6$이다.

    **잣수를 맞춘 값:** $\hat{v}_1 = \lfloor 100/6 \rfloor = 16$, $\hat{v}_2 = \lfloor 60/6 \rfloor = 10$, $\hat{v}_3 = \lfloor 120/6 \rfloor = 20$, $\hat{v}_4 = \lfloor 80/6 \rfloor = 13$.

    **잣수를 맞춘 동적 짜기의 가장 좋은 값:** 물건 1과 4을 고른다(무게 $5+4=9 \leq 10$, 잣수를 맞춘 값 $16+13=29$).

    **참 값:** $100 + 80 = 180$. **OPT:** 물건 1과 3은 무게가 $12 > 10$이고, 물건 1과 4은 $180$을 주며, 물건 2와 3은 무게 $10$에 값 $180$이다. 따라서 OPT $= 180$이다.

    **비율:** $180 / 180 = 1.0 \geq 1 - \epsilon = 0.8$이다. 보장이 참이다.

## 참고 문헌

- Vazirani, V. V. (2001). *Approximation Algorithms*. Springer.
- Arora, S. (1998). Polynomial time approximation schemes for Euclidean traveling salesman and other geometric problems. *JACM*, 45(5), 753--782.
- Williamson, D. P., & Shmoys, D. B. (2011). *The Design of Approximation Algorithms*. Cambridge University Press.

## 연습문제

**연습문제 1.**
PTAS과 FPTAS을 뜻매김하라. 둘의 핵심 차이는 무엇인가?

??? success "연습문제 1 풀이"
    **PTAS**(다항식 시간 어림 얼개)는 아무 $\epsilon > 0$에 대해 $(1+\epsilon)$ 어림을 이루며 도는 시간이 $n$에 대해 다항식이지만 $1/\epsilon$에 대해서는 지수일 수 있다(예컨대 $O(n^{1/\epsilon})$). **FPTAS**(온전 다항식 시간 어림 얼개)는 같은 보장을 이루되 $n$과 $1/\epsilon$ 모두에 대해 다항식 시간에 돈다(예컨대 $O(n^2/\epsilon)$). FPTAS이 엄격히 더 세다. 곧 바라는 어떤 정확도에서도 쓸 만하다. $\square$

---

**연습문제 2.**
PTAS이 있는 모든 NP-어려운 문제에 FPTAS이 있지는 않은 까닭은 무엇인가?

??? success "연습문제 2 풀이"
    FPTAS이 있으면 ($\epsilon$을 넉넉히 작게 두어) 비슷 다항식 시간 알고리즘이 있다는 뜻이 된다. 강하게 NP-어려운 문제(수가 다항식으로 가둬져도 NP-어려운 문제)에는 P = NP이 아니라면 비슷 다항식 알고리즘이 없다. 따라서 강하게 NP-어려운 문제에는 FPTAS이 있을 수 없다. 보기: 통 담기와 나란한 기계의 차례 잡기는 PTAS은 있으나 (P = NP이 아니라면) FPTAS은 없다. $\square$

---

**연습문제 3.**
배낭 문제에는 FPTAS이 있다. 그 핵심 생각을 설명하라.

??? success "연습문제 3 풀이"
    물건의 이익을 잣수 맞추고 반올림한다. 곧 $K = \epsilon \cdot v_{\max} / n$일 때 이익마다 $K$으로 나누고 내림한다. 그러면 이익의 범위가 $O(n/\epsilon)$으로 줄어 동적 짜기 표 크기가 $n$과 $1/\epsilon$ 모두에 대해 다항식이 된다. 물건마다 반올림 어긋남은 많아야 $K$이므로 온 어긋남은 많아야 $nK = \epsilon \cdot v_{\max} \leq \epsilon \cdot OPT$이다. 시간: $O(n^2/\epsilon)$. 이는 (가장 크게 하기에서) $(1-\epsilon)$ 어림을 준다. $\square$

---

**연습문제 4.**
NP-어려운 문제에 FPTAS이 있으면 (P $\neq$ NP을 가정할 때) 그 문제가 강하게 NP-어렵지 않음을 밝혀라.

??? success "연습문제 4 풀이"
    FPTAS에서 $\epsilon = 1/(2 \cdot OPT)$으로 둔다. FPTAS은 $n$과 $1/\epsilon = 2 \cdot OPT$에 대해 다항식 시간에 돈다. (강하게 NP-어려운 자리처럼) $OPT$이 $n$의 다항식으로 가둬져 있으면 이는 다항식 시간의 정확한 알고리즘이 된다(정수 값 최적해의 $(1 + 1/(2 \cdot OPT))$ 갑절 안 어림은 정확하기 때문이다). 그러면 강하게 NP-어려운 문제가 P에 들어 P $\neq$ NP에 어긋난다. $\square$
