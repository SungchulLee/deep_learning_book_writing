# 커누스 가장 좋게 하기

가장 좋은 두 갈래 찾기 나무나 값이 가장 적은 합치기 같은 구간 동적 짜기 문제는 상태 $dp[i][j]$마다 $i$부터 $j$까지 가르는 점 $k$을 모두 훑으므로 흔히 $O(n^3)$ 시간이 든다. 커누스 가장 좋게 하기는 값 함수의 짜임 성질인 **사각 부등식**을 써서 가르는 점 찾기를 더 좁은 범위로 제한한다. 가장 좋은 가르는 점을 적어 두고 한쪽으로만 가는 성질 $\text{opt}(i, j-1) \leq \text{opt}(i, j) \leq \text{opt}(i+1, j)$을 쓰면 모든 상태에 걸친 일감이 $O(n^3)$에서 $O(n^2)$으로 떨어진다.

---

## 1. 문제 설정

다음 꼴의 구간 동적 짜기 되돌이 관계식을 보자:

$$
dp[i][j] = \min_{i \leq k < j} \bigl( dp[i][k] + dp[k+1][j] + C(i, j) \bigr)
$$

여기서 $C(i, j)$은 구간 $[i, j]$을 합치는 값이며 가르는 점 $k$과는 무관하다. $\text{opt}(i, j)$을 가장 작은 값을 주는 가장 작은 $k$이라 하자.

---

## 2. 사각 부등식

함수 $C$이 **사각 부등식**을 채운다는 것은 다음을 뜻한다:

$$
C(a, c) + C(b, d) \leq C(a, d) + C(b, c) \quad \text{for all } a \leq b \leq c \leq d
$$

직관으로 보면 "품은" 두 구간의 값이 "엇갈린" 두 구간의 값보다 크지 않다는 말이다.

!!! note "넉넉한 조건"
    $C(i, j)$이 사각 부등식을 채우고 **한쪽으로만 간다면**(곧 $[i', j'] \subseteq [i, j]$일 때마다 $C(i', j') \leq C(i, j)$), 동적 짜기 값 함수 $dp[i][j]$도 사각 부등식을 채우고 가장 좋은 가르는 점도 한쪽으로만 간다.

---

## 3. 가르는 점이 한쪽으로만 감

사각 부등식이 성립하면:

$$
\text{opt}(i, j-1) \leq \text{opt}(i, j) \leq \text{opt}(i+1, j)
$$

곧 주어진 구간의 가장 좋은 가르는 점이 "이웃한" 두 아래 구간의 가장 좋은 가르는 점 사이에 낀다는 뜻이다. 이 제약이 찾는 범위를 크게 좁힌다.

---

## 4. 알고리즘

여느 구간 동적 짜기에 가하는 핵심 고침:

1. 가장 좋은 가르는 점을 적어 두는 표 $\text{opt}[i][j]$을 둔다
2. $dp[i][j]$을 셈할 때 $k$을 $\text{opt}[i][j-1]$부터 $\text{opt}[i+1][j]$까지만 훑는다

**고르게 나눈 분석**: 구간 길이 $\ell = j - i$을 고정할 때 길이 $\ell$인 모든 구간에 걸쳐 살피는 가르는 점의 총수는 다음과 같다:

$$
\sum_{i} \bigl(\text{opt}(i+1, i+\ell) - \text{opt}(i, i+\ell-1)\bigr) + n = O(n)
$$

이는 망원경 합이다. $O(n)$개 길이에 걸쳐 더하면 전체 일감이 $O(n^2)$이다.

---

## 5. 복잡도 비교

| 방법 | 시간 | 공간 |
|--------|------|-------|
| 막무가내 구간 동적 짜기 | $O(n^3)$ | $O(n^2)$ |
| 커누스 가장 좋게 하기 | $O(n^2)$ | $O(n^2)$ |

---

## 6. 구현

```python
"""
커누스 가장 좋게 하기: 구간 동적 짜기를 O(n^3)에서 O(n^2)으로 줄인다.

값 함수가 사각 부등식을 채워 가장 좋은 가르는 점이
한쪽으로만 감이 보장될 때 쓴다.
"""

import math

# ===================================================================
# 커누스 가장 좋게 하기를 쓴 가장 좋은 두 갈래 찾기 나무
# ===================================================================
def optimal_bst(freq: list[int]) -> int:
    """가장 좋은 두 갈래 찾기 나무의 가장 적은 기대 찾기 값을 찾는다.

    찾는 잦기가 붙은 열쇠가 주어질 때 무게 붙은 전체 찾기 값을
    가장 적게 하는 두 갈래 찾기 나무 짜임을 찾는다.

    매개변수
    ----------
    freq : list[int]
        열쇠 0, 1, ..., n-1의 찾는 잦기.

    반환값
    -------
    int
        무게 붙은 가장 적은 전체 찾기 값.
    """
    n = len(freq)
    INF = math.inf

    # O(1)에 구간 잦기를 묻기 위한 앞합
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + freq[i]

    def range_freq(i: int, j: int) -> int:
        return prefix[j + 1] - prefix[i]

    dp = [[0] * n for _ in range(n)]
    opt = [[0] * n for _ in range(n)]

    # 바탕 경우: 열쇠 하나
    for i in range(n):
        dp[i][i] = freq[i]
        opt[i][i] = i

    # 길이가 늘어나는 차례로 채운다
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = INF

            lo = opt[i][j - 1]
            hi = opt[i + 1][j] if i + 1 <= j else j

            for k in range(lo, min(hi, j) + 1):
                left = dp[i][k - 1] if k > i else 0
                right = dp[k + 1][j] if k < j else 0
                cost = left + right + range_freq(i, j)
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    opt[i][j] = k

    return dp[0][n - 1]

# ===================================================================
# 두루 쓰는 커누스 가장 좋게 한 구간 동적 짜기
# ===================================================================
def knuth_interval_dp(n: int, cost_fn) -> int:
    """두루 쓰는 커누스 가장 좋게 한 구간 동적 짜기.

    매개변수
    ----------
    n : int
        원소의 수.
    cost_fn : callable
        cost_fn(i, j)은 구간 [i, j]의 합치기 값을 돌려준다.

    반환값
    -------
    int
        가장 적은 전체 값.
    """
    INF = math.inf
    dp = [[0] * n for _ in range(n)]
    opt = [[0] * n for _ in range(n)]

    for i in range(n):
        opt[i][i] = i

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = INF

            lo = opt[i][j - 1]
            hi = opt[i + 1][j] if i + 1 <= j else j

            for k in range(lo, min(hi, j) + 1):
                left = dp[i][k - 1] if k > i else 0
                right = dp[k + 1][j] if k < j else 0
                val = left + right + cost_fn(i, j)
                if val < dp[i][j]:
                    dp[i][j] = val
                    opt[i][j] = k

    return dp[0][n - 1]

# ===================================================================
# 메인
# ===================================================================
if __name__ == "__main__":
    # 가장 좋은 두 갈래 찾기 나무 보기
    freq = [25, 20, 5, 20, 30]
    result = optimal_bst(freq)
    print(f"Frequencies: {freq}")
    print(f"Optimal BST cost: {result}")

    # 돌 합치기에서 막무가내 O(n^3)과 커누스 O(n^2)을 견준다
    piles = [3, 5, 1, 2, 6]
    prefix = [0]
    for p in piles:
        prefix.append(prefix[-1] + p)
    cost_fn = lambda i, j: prefix[j + 1] - prefix[i]

    result = knuth_interval_dp(len(piles), cost_fn)
    print(f"\nPiles: {piles}")
    print(f"Merge cost (Knuth): {result}")
```

**출력:**
```
Frequencies: [25, 20, 5, 20, 30]
Optimal BST cost: 210
Merge cost (Knuth): 38
```

??? example "앞합에 대해 사각 부등식 확인하기"
    $C(i, j) = \sum_{t=i}^{j} a_t$이라 하자. $a \leq b \leq c \leq d$에 대해 살피자:

    $$
    C(a,c) + C(b,d) = \sum_{t=a}^{c} a_t + \sum_{t=b}^{d} a_t
    $$

    $$
    C(a,d) + C(b,c) = \sum_{t=a}^{d} a_t + \sum_{t=b}^{c} a_t
    $$

    차는 $C(a,c) + C(b,d) - C(a,d) - C(b,c) = -\sum_{t=c+1}^{d} a_t + \sum_{t=c+1}^{d} a_t = 0 \leq 0$이므로 앞합 값에서는 사각 부등식이 등호로 성립한다.

---

## 연습문제

**연습문제 1.**
커누스 가장 좋게 하기의 상태, 옮아감, 바탕 경우를 가려내어라.

??? success "연습문제 1 풀이"
    **상태**는 아래 문제를 적는 데 필요한 앎을 담는다. **옮아감**(되돌이 관계식)은 어떤 상태의 가장 좋은 값을 더 작은 상태로 나타낸다. **바탕 경우**는 곧바로 풀 수 있는 가장 작은 아래 문제의 값을 준다. 이 셋이 함께 동적 짜기 풀이를 온전히 정한다. $\square$

---

**연습문제 2.**
커누스 가장 좋게 하기의 위에서 아래로(적어 두기) 짜기와 아래에서 위로(표 채우기) 짜기를 견주어라. 어느 쪽이 나으며 왜 그런가?

??? success "연습문제 2 풀이"
    **위에서 아래로**: 곳간을 곁들인 되돌이. 정말 필요한 아래 문제만 셈한다(게으른 값매김). 되돌이 관계식에서 옮겨 적기 쉽다. 되돌이 깊이 문제가 생길 수 있다. **아래에서 위로**: 되풀이로 기댐 차례에 따라 표를 채운다. 필요 없는 것까지 모든 아래 문제를 셈한다. 되돌이 군더더기가 없다. 공간을 줄이기 쉽다. 이 문제에서는 아래 문제가 모두 필요하면 아래에서 위로가 흔히 낫고, 닿지 않는 아래 문제가 많으면 위에서 아래로가 낫다. $\square$

---

**연습문제 3.**
커누스 가장 좋게 하기의 시간 복잡도와 공간 복잡도는 무엇인가? 공간을 더 줄일 수 있는가?

??? success "연습문제 3 풀이"
    시간 복잡도는 상태의 수에 상태마다의 옮아감 값을 곱한 것으로 정해진다. 공간은 담아 두는 상태의 수와 같다. 옮아감이 앞선 상태 가운데 한정된 몇 개에만 기대면(예컨대 2차원 표의 바로 앞 가로줄) 그 상태만 기억 공간에 두어 공간을 줄일 수 있으며, 흔히 $O(n^2)$에서 $O(n)$으로 줄어든다. $\square$

---

**연습문제 4.**
커누스 가장 좋게 하기의 알고리즘을 네가 고른 작은 보기에 대해 좇아라. 동적 짜기 표의 값을 보여라.

??? success "연습문제 4 풀이"
    작은 들임(예컨대 $n = 5$이나 짧은 글줄/배열)을 골라라. 동적 짜기 표를 한 걸음씩 채우면서 각 칸이 앞서 셈한 칸에서 어떻게 나오는지 보여라. 마지막 답을 막무가내로 다 세어 본 것과 견주어 확인하라. 이렇게 좇아 보면 되돌이 관계식이 옳음을 확인하고 알고리즘에 대한 직관이 선다. $\square$

## 정리하며

이 마당은 문제 설정、사각 부등식、가르는 점이 한쪽으로만 감、알고리즘을 차례로 짚었다.

**참고 문헌**

- Knuth, D. E. (1971). Optimum binary search trees. *Acta Informatica*, 1(1), 14--25.
- Yao, F. F. (1980). Efficient dynamic programming using quadrangle inequalities. *Proc. STOC*, 429--435.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 15. MIT Press.
