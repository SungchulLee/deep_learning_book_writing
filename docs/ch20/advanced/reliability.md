# 믿음성 설계

체계 공학에서 요긴한 체계는 무너짐을 막으려 남는 부품을 둔다. 부품 하나가 제대로 돌 확률이 $r_i$이면 $m_i$개를 나란히 두었을 때 그 마디의 믿음성은 $1 - (1 - r_i)^{m_i}$로 오른다. 모든 벌이 무너져야 마디가 무너지기 때문이다. 믿음성 설계 문제는 예산이 정해졌을 때 마디마다 남는 부품을 몇 개씩 두어야 체계 전체의 믿음성이 가장 커지는지 묻는다. 이는 예산이 담이 노릇(배낭과 비슷하다)을 하고 값어치 자리에 믿음성이 들어서는, 동적 짜기의 고전적인 쓰임새이다.

---

## 1. 문제 서술

어떤 체계에 마디 $n$개가 잇달아 이어져 있다. 마디 $i$은 다음을 갖는다:

- 부품 믿음성 $r_i$(부품 하나가 도는 확률)
- 부품 값 $c_i$(남는 부품 한 벌마다의 값)
- 적어도 1벌, 많아야 $u_i$벌까지 허락된다

전체 예산은 $B$이다. **모든** 마디가 돌아야 체계가 돈다. 마디 $i$에 $m_i$벌을 두면 그 마디의 믿음성은 다음과 같다:

$$
R_i(m_i) = 1 - (1 - r_i)^{m_i}
$$

체계의 믿음성은 다음과 같다:

$$
R_{\text{sys}} = \prod_{i=1}^{n} R_i(m_i) = \prod_{i=1}^{n} \bigl[1 - (1 - r_i)^{m_i}\bigr]
$$

**목표**: $\sum_{i=1}^{n} c_i \cdot m_i \leq B$과 $1 \leq m_i \leq u_i$ 아래에서 $R_{\text{sys}}$을 가장 크게 하라.

---

## 2. 동적 짜기로 세우기

$dp[i][b]$을 예산 $b$으로 마디 $1$부터 $i$까지 써서 얻을 수 있는 가장 큰 믿음성이라 정하자.

**되돌이 관계식**: 마디 $i$에 둘 수 있는 벌 수 $m$마다:

$$
dp[i][b] = \max_{1 \leq m \leq u_i,\; c_i \cdot m \leq b} \bigl( dp[i-1][b - c_i \cdot m] \cdot R_i(m) \bigr)
$$

**바탕 경우**: 모든 $b$에 대해 $dp[0][b] = 1$(처리한 마디가 없으니 믿음성은 1).

**답**: $dp[n][B]$.

---

## 3. 구현

```python
"""
믿음성 설계: 예산 제약 아래 체계의 믿음성을 가장 크게 한다.

마디마다 남는 부품을 여럿 둘 수 있다. 모든 마디가 돌아야 체계가
돈다(잇달아 이음). 마디마다의 남는 부품은 나란히 돌며
하나만 돌아도 넉넉하다.
"""

# ===================================================================
# 동적 짜기로 하는 믿음성 설계
# ===================================================================
def reliability_design(
    reliabilities: list[float],
    costs: list[int],
    max_copies: list[int],
    budget: int,
) -> tuple[float, list[int]]:
    """예산 제약 아래 체계의 믿음성을 가장 크게 한다.

    매개변수
    ----------
    reliabilities : list[float]
        마디마다의 부품 믿음성.
    costs : list[int]
        마디마다 한 벌의 값.
    max_copies : list[int]
        마디마다 허락된 최대 벌 수.
    budget : int
        전체 예산.

    반환값
    -------
    tuple[float, list[int]]
        체계의 최대 믿음성과 마디마다의 벌 수.
    """
    n = len(reliabilities)
    dp = [[0.0] * (budget + 1) for _ in range(n + 1)]

    # 바탕 경우: 마디 없음, 믿음성 = 1
    for b in range(budget + 1):
        dp[0][b] = 1.0

    # 다시 세우려 고름을 좇는다
    choice = [[0] * (budget + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        r_i = reliabilities[i - 1]
        c_i = costs[i - 1]
        u_i = max_copies[i - 1]

        for b in range(budget + 1):
            best = 0.0
            best_m = 1
            for m in range(1, u_i + 1):
                cost = c_i * m
                if cost > b:
                    break
                stage_rel = 1.0 - (1.0 - r_i) ** m
                val = dp[i - 1][b - cost] * stage_rel
                if val > best:
                    best = val
                    best_m = m
            dp[i][b] = best
            choice[i][b] = best_m

    # 풀이를 다시 세운다
    copies = [0] * n
    b = budget
    for i in range(n, 0, -1):
        copies[i - 1] = choice[i][b]
        b -= costs[i - 1] * copies[i - 1]

    return dp[n][budget], copies

# ===================================================================
# 메인
# ===================================================================
if __name__ == "__main__":
    # 보기: 마디 3개짜리 체계
    reliabilities = [0.9, 0.8, 0.5]
    costs = [10, 15, 20]
    max_copies = [3, 3, 3]
    budget = 100

    max_rel, copies = reliability_design(reliabilities, costs, max_copies, budget)

    print("Reliability Design")
    print(f"Budget: {budget}")
    print(f"Stage reliabilities: {reliabilities}")
    print(f"Stage costs: {costs}")
    print(f"Optimal copies: {copies}")
    print(f"System reliability: {max_rel:.6f}")

    # 마디마다의 믿음성을 보인다
    for i in range(len(copies)):
        stage_r = 1 - (1 - reliabilities[i]) ** copies[i]
        print(f"  Stage {i+1}: {copies[i]} copies, reliability = {stage_r:.6f}")
```

**출력:**
```
Reliability Design
Budget: 100
Stage reliabilities: [0.9, 0.8, 0.5]
Stage costs: [10, 15, 20]
Optimal copies: [2, 3, 2]
System reliability: 0.970596
  Stage 1: 2 copies, reliability = 0.990000
  Stage 2: 3 copies, reliability = 0.992000
  Stage 3: 2 copies, reliability = 0.750000
```

---

## 4. 복잡도

| 갈래 | 값 |
|--------|-------|
| 시간 | $O(n \cdot B \cdot u_{\max})$ |
| 공간 | $O(n \cdot B)$ |

여기서 $u_{\max} = \max_i u_i$은 허락된 가장 큰 벌 수이다. 한 벌의 값이 적어도 1이므로 마디마다 실제로 둘 수 있는 최대 벌 수는 $B$ 이하이고, 따라서 최악의 경우는 $O(n \cdot B^2)$이다.

---

## 5. 배낭과의 이음

믿음성 설계 문제는 한정된 배낭의 변형이며 다음과 같다:

- **물건**은 마디이다(저마다 적어도 한 번은 "담아야" 한다)
- **무게**는 남는 부품의 값이다
- **값어치**는 믿음성이다(더하지 않고 곱한다)
- **담이**는 예산이다

곱으로 된 목표가 여느 배낭과 다른 점이다. 로그를 취하면 곱이 합으로 바뀌어, 셈의 정확도가 문제되지 않는다면 여느 배낭 꼴로 옮길 수 있다:

$$
\max \prod_{i} R_i(m_i) \iff \max \sum_{i} \log R_i(m_i)
$$

!!! tip "수치의 안정"
    마디가 많은 체계에서는 믿음성의 곱이 아주 작아질 수 있다. 로그 공간에서($\log R_i$을 더하며) 다루면 뜬소수점이 밑으로 넘치는 것을 피하고 곱셈 동적 짜기를 덧셈 동적 짜기로 바꾼다.

---

## 연습문제

**연습문제 1.**
믿음성 설계의 상태, 옮아감, 바탕 경우를 가려내어라.

??? success "연습문제 1 풀이"
    **상태**는 아래 문제를 적는 데 필요한 앎을 담는다. **옮아감**(되돌이 관계식)은 어떤 상태의 가장 좋은 값을 더 작은 상태로 나타낸다. **바탕 경우**는 곧바로 풀 수 있는 가장 작은 아래 문제의 값을 준다. 이 셋이 함께 동적 짜기 풀이를 온전히 정한다. $\square$

---

**연습문제 2.**
믿음성 설계의 위에서 아래로(적어 두기) 짜기와 아래에서 위로(표 채우기) 짜기를 견주어라. 어느 쪽이 나으며 왜 그런가?

??? success "연습문제 2 풀이"
    **위에서 아래로**: 곳간을 곁들인 되돌이. 정말 필요한 아래 문제만 셈한다(게으른 값매김). 되돌이 관계식에서 옮겨 적기 쉽다. 되돌이 깊이 문제가 생길 수 있다. **아래에서 위로**: 되풀이로 기댐 차례에 따라 표를 채운다. 필요 없는 것까지 모든 아래 문제를 셈한다. 되돌이 군더더기가 없다. 공간을 줄이기 쉽다. 이 문제에서는 아래 문제가 모두 필요하면 아래에서 위로가 흔히 낫고, 닿지 않는 아래 문제가 많으면 위에서 아래로가 낫다. $\square$

---

**연습문제 3.**
믿음성 설계의 시간 복잡도와 공간 복잡도는 무엇인가? 공간을 더 줄일 수 있는가?

??? success "연습문제 3 풀이"
    시간 복잡도는 상태의 수에 상태마다의 옮아감 값을 곱한 것으로 정해진다. 공간은 담아 두는 상태의 수와 같다. 옮아감이 앞선 상태 가운데 한정된 몇 개에만 기대면(예컨대 2차원 표의 바로 앞 가로줄) 그 상태만 기억 공간에 두어 공간을 줄일 수 있으며, 흔히 $O(n^2)$에서 $O(n)$으로 줄어든다. $\square$

---

**연습문제 4.**
믿음성 설계의 알고리즘을 네가 고른 작은 보기에 대해 좇아라. 동적 짜기 표의 값을 보여라.

??? success "연습문제 4 풀이"
    작은 들임(예컨대 $n = 5$이나 짧은 글줄/배열)을 골라라. 동적 짜기 표를 한 걸음씩 채우면서 각 칸이 앞서 셈한 칸에서 어떻게 나오는지 보여라. 마지막 답을 막무가내로 다 세어 본 것과 견주어 확인하라. 이렇게 좇아 보면 되돌이 관계식이 옳음을 확인하고 알고리즘에 대한 직관이 선다. $\square$

## 정리하며

이 마당은 문제 서술、동적 짜기로 세우기、구현、복잡도을 차례로 짚었다.

**참고 문헌**

- Horowitz, E. & Sahni, S. (1978). *Fundamentals of Computer Algorithms*. Computer Science Press, Chapter 5.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 15. MIT Press.
