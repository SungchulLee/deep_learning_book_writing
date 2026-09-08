# 표 채우기로 하는 최장 공통 부분 차례

되돌이로 푸는 최장 공통 부분 차례는 겹치는 아래 문제를 여러 번 다시 셈한다. 표 채우기는 2차원 표를 아래에서 위로 채워 아래 문제를 꼭 한 번씩만 셈해 이 겹침을 없앤다. 이 쪽은 아래에서 위로 가는 동적 짜기 방식과 공간을 줄인 변형에 힘을 쏟으며, 형제 쪽에서 다룬 되돌이 방식과 적어 두기 방식을 채워 준다.

---

## 1. 아래에서 위로 가는 전략

되돌이 대신 크기 $(m+1) \times (n+1)$인 표 $c$을 잡고 가로줄마다 채운다. 칸 $c[i][j]$은 앞머리 $X_i = x_1 \cdots x_i$과 $Y_j = y_1 \cdots y_j$의 최장 공통 부분 차례 길이를 담는다:

$$
c[i][j] = \begin{cases} 0 & \text{if } i = 0 \text{ or } j = 0 \\ c[i-1][j-1] + 1 & \text{if } x_i = y_j \\ \max(c[i-1][j],\, c[i][j-1]) & \text{if } x_i \ne y_j \end{cases}
$$

답은 $c[m][n]$이다.

---

## 2. 채우는 차례와 기댐

칸 $c[i][j]$은 이웃 셋에 기댄다:

- $c[i-1][j-1]$(대각선, 왼쪽 위)
- $c[i-1][j]$(바로 위)
- $c[i][j-1]$(바로 왼쪽)

가로줄을 위에서 아래로, 세로줄을 왼쪽에서 오른쪽으로 처리하면 칸을 셈하기 앞서 기댐이 모두 채워진다.

---

## 3. 공간 줄이기

가로줄 $i$이 가로줄 $i-1$에만 기대므로 가로줄 둘이면 넉넉하다. "앞선" 가로줄과 "지금" 가로줄을 번갈아 쓰면 공간이 $O(mn)$에서 $O(\min(m, n))$으로 준다.

!!! tip "가로줄 하나 재주"
    꼼꼼히 기록하면 1차원 배열 하나와 (대각선 원소를 담을) 변수 하나면 넉넉하다. 배열 크기를 가장 작게 하려면 짧은 글줄을 세로줄 차원으로 둔다.

---

## 4. 최장 공통 부분 차례 찍기

길이뿐 아니라 실제 부분 차례를 다시 세우려면 채우는 동안 방향 표 $b[i][j]$을 둔다:

- $x_i = y_j$이면 $b[i][j] = \text{DIAG}$(그 글자가 최장 공통 부분 차례에 든다)
- $c[i-1][j] \ge c[i][j-1]$이면 $b[i][j] = \text{UP}$
- 아니면 $b[i][j] = \text{LEFT}$

$b[m][n]$에서 가장자리까지 거슬러 좇아 최장 공통 부분 차례 글자를 거꾸로 모은다.

---

## 5. 파이썬 구현

```python
"""
최장 공통 부분 차례 — 공간을 줄인, 아래에서 위로 가는 표 채우기.

여느 2차원 표 채우기, 공간을 줄인 1차원 변형, 방향 표로 하는
부분 차례 다시 세우기를 보인다.
"""

# === 2차원 표 채우기 ===

def lcs_tabulation(x: str, y: str) -> int:
    """아래에서 위로 가는 동적 짜기로 얻는 길이. 시간: O(mn), 공간: O(mn)."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]

# === 공간 줄임(가로줄 둘) ===

def lcs_space_optimized(x: str, y: str) -> int:
    """가로줄 둘만 써서 얻는 길이. 시간: O(mn), 공간: O(min(m,n))."""
    if len(x) < len(y):
        x, y = y, x
    m, n = len(x), len(y)

    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)

    return prev[n]

# === 방향 표로 다시 세우기 ===

def lcs_with_reconstruction(x: str, y: str) -> tuple[int, str]:
    """최장 공통 부분 차례의 길이와 그 글줄을 돌려준다."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    direction = [[""] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                direction[i][j] = "DIAG"
            elif dp[i - 1][j] >= dp[i][j - 1]:
                dp[i][j] = dp[i - 1][j]
                direction[i][j] = "UP"
            else:
                dp[i][j] = dp[i][j - 1]
                direction[i][j] = "LEFT"

    # 거슬러 좇아 최장 공통 부분 차례를 되찾는다
    i, j = m, n
    result = []
    while i > 0 and j > 0:
        if direction[i][j] == "DIAG":
            result.append(x[i - 1])
            i -= 1
            j -= 1
        elif direction[i][j] == "UP":
            i -= 1
        else:
            j -= 1

    return dp[m][n], "".join(reversed(result))

# === 메인 ===

if __name__ == "__main__":
    x = "AGGTAB"
    y = "GXTXAYB"

    print(f"X = {x}")
    print(f"Y = {y}")
    print(f"LCS length (2D):    {lcs_tabulation(x, y)}")
    print(f"LCS length (space): {lcs_space_optimized(x, y)}")

    length, subseq = lcs_with_reconstruction(x, y)
    print(f"LCS: '{subseq}' (length {length})")
    # 내임:
    # X = AGGTAB
    # Y = GXTXAYB
    # 최장 공통 부분 차례 길이(2차원):    4
    # 최장 공통 부분 차례 길이(공간 줄임): 4
    # 최장 공통 부분 차례: 'GTAB'(길이 4)
```

**출력:**

```
X = AGGTAB
Y = GXTXAYB
LCS length (2D):    4
LCS length (space): 4
LCS: 'GTAB' (length 4)
```

---

## 6. 풀이 예제

$X = \text{AGGTAB}$이고 $Y = \text{GXTXAYB}$일 때:

| | $\varepsilon$ | G | X | T | X | A | Y | B |
|---|---|---|---|---|---|---|---|---|
| $\varepsilon$ | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| A | 0 | 0 | 0 | 0 | 0 | **1** | 1 | 1 |
| G | 0 | **1** | 1 | 1 | 1 | 1 | 1 | 1 |
| G | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| T | 0 | 1 | 1 | **2** | 2 | 2 | 2 | 2 |
| A | 0 | 1 | 1 | 2 | 2 | **3** | 3 | 3 |
| B | 0 | 1 | 1 | 2 | 2 | 3 | 3 | **4** |

최장 공통 부분 차례의 길이는 4이다. 대각선 칸을 좇으면 $\text{GTAB}$이 나온다.

---

## 연습문제

**연습문제 1.**
"ABCBDAB"과 "BDCAB"의 최장 공통 부분 차례를 찾아라.

??? success "연습문제 1 풀이"
    동적 짜기 표를 세운다. 최장 공통 부분 차례의 길이는 4이다. 하나는 "BCAB"이다. 곧 B(1)이 B(0)과, C(2)가 C(2)와, A(5)가 A(3)과, B(6)이 B(4)와 맞는다. "BDAB"도 옳은 답이다. 최대 길이가 같은 최장 공통 부분 차례가 여럿일 수 있다. $\square$

---

**연습문제 2.**
최장 공통 부분 차례 문제가 가장 좋은 아래 짜임을 갖춤을 밝혀라.

??? success "연습문제 2 풀이"
    $Z = z_1, \ldots, z_k$을 $X = x_1, \ldots, x_m$과 $Y = y_1, \ldots, y_n$의 최장 공통 부분 차례라 하자. $x_m = y_n$이면 $z_k = x_m = y_n$이고 $z_1, \ldots, z_{k-1}$은 $X[1..m-1]$과 $Y[1..n-1]$의 최장 공통 부분 차례이다. $x_m \neq y_n$이고 $z_k \neq x_m$이면 $Z$은 $X[1..m-1]$과 $Y$의 최장 공통 부분 차례이다. $z_k \neq y_n$이면 $Z$은 $X$과 $Y[1..n-1]$의 최장 공통 부분 차례이다. 어느 경우든 본디 문제의 최장 공통 부분 차례가 아래 문제의 것을 품는다. $\square$

---

**연습문제 3.**
최장 공통 부분 차례 알고리즘의 시간 복잡도와 공간 복잡도는 무엇인가? 공간을 어떻게 줄이는가?

??? success "연습문제 3 풀이"
    여느 동적 짜기: 표에 $O(mn)$ 시간과 $O(mn)$ 공간. 가로줄마다 바로 앞 가로줄에만 기대므로 가로줄 둘만 지녀 공간을 $O(\min(m,n))$으로 줄일 수 있다. 다만 실제 최장 공통 부분 차례를 다시 세우려면 표 전체나 히르슈베르크의 나누어 이기기 방식이 필요하다. $\square$

---

**연습문제 4.**
최장 공통 부분 글줄(잇닿음)은 최장 공통 부분 차례와 다르다. 두 동적 짜기 꼴은 어떻게 다른가?

??? success "연습문제 4 풀이"
    최장 공통 부분 차례에서 $dp[i][j]$은 앞머리 $X[1..i]$과 $Y[1..j]$의 최장 공통 부분 차례 길이를 뜻하며 잇닿지 않은 자리도 보탠다. 최장 공통 부분 글줄에서 $dp[i][j]$은 $X[1..i]$과 $Y[1..j]$의 가장 긴 공통 뒷가지 길이를 뜻한다(자리 $i$과 $j$에서 끝나야 한다). 되돌이 관계식: $X[i] = Y[j]$이면 $dp[i][j] = dp[i-1][j-1] + 1$, 아니면 $0$. 답은 $\max_{i,j} dp[i][j]$이다. 둘 다 $O(mn)$ 시간이다. $\square$

## 정리하며

이 마당은 아래에서 위로 가는 전략、채우는 차례와 기댐、공간 줄이기、최장 공통 부분 차례 찍기을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 14. MIT Press.
