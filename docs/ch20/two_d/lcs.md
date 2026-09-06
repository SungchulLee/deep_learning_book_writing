# 최장 공통 부분 차례

글줄의 **부분 차례**는 남은 글자의 차례를 바꾸지 않고 글자를 0개 이상 지워 얻는다. 최장 공통 부분 차례(LCS) 문제는 차례 $X$과 $Y$이 주어질 때 둘 모두의 부분 차례로 나타나는 가장 긴 차례를 묻는다. 동적 짜기의 바탕 문제이며 차이 보기 도구, 생물정보학(DNA 차례 맞추기), 판 관리 체계에 쓰인다.

## 문제의 정의

차례 $X = x_1 x_2 \cdots x_m$과 $Y = y_1 y_2 \cdots y_n$이 주어질 때 $X$과 $Y$ 둘 모두의 부분 차례인 가장 긴 차례 $Z$의 길이를 찾아라.

**보기.** $X = \text{ABCBDAB}$이고 $Y = \text{BDCABA}$이면 최장 공통 부분 차례 하나는 $\text{BCBA}$이고 길이는 4이다.

## 가장 좋은 밑짜임

최장 공통 부분 차례 문제는 가장 좋은 아래 짜임을 갖춘다. $X_i = x_1 \cdots x_i$과 $Y_j = y_1 \cdots y_j$을 앞머리라 하자. 그러면:

- $x_i = y_j$이면 이 공통 글자가 최장 공통 부분 차례에 들어야 하고 나머지는 $X_{i-1}$과 $Y_{j-1}$에서 온다.
- $x_i \ne y_j$이면 $x_i$이나 $y_j$ 가운데 적어도 하나는 최장 공통 부분 차례에 들지 않으므로 $\text{LCS}(X_i, Y_{j-1})$과 $\text{LCS}(X_{i-1}, Y_j)$ 가운데 긴 것을 취한다.

## 점화식

$c[i][j]$을 $X_i$과 $Y_j$의 최장 공통 부분 차례 길이라 정하자:

$$
c[i][j] = \begin{cases} 0 & \text{if } i = 0 \text{ or } j = 0 \\ c[i-1][j-1] + 1 & \text{if } i,j > 0 \text{ and } x_i = y_j \\ \max(c[i][j-1],\, c[i-1][j]) & \text{if } i,j > 0 \text{ and } x_i \ne y_j \end{cases}
$$

바탕 경우는 어떤 차례든 빈 차례와의 최장 공통 부분 차례 길이가 0임을 말한다.

## 풀이 예제

$X = \text{ABCB}$이고 $Y = \text{BDCB}$일 때 동적 짜기 표는 다음과 같다:

|  | $\varepsilon$ | B | D | C | B |
|---|---|---|---|---|---|
| $\varepsilon$ | 0 | 0 | 0 | 0 | 0 |
| A | 0 | 0 | 0 | 0 | 0 |
| B | 0 | **1** | 1 | 1 | 1 |
| C | 0 | 1 | 1 | **2** | 2 |
| B | 0 | 1 | 1 | 2 | **3** |

대각선의 맞음을 읽으면 $(2,1)$의 B, $(3,3)$의 C, $(4,4)$의 B이므로 최장 공통 부분 차례는 $\text{BCB}$이고 길이는 3이다.

## 복잡도

| 갈래 | 값 |
|---|---|
| 시간 | $O(mn)$ |
| 공간(2차원) | $O(mn)$ |
| 공간(1차원) | $O(\min(m,n))$ |
| 아래 문제 | $(m+1)(n+1)$ |

## 파이썬 구현

```python
"""
최장 공통 부분 차례 — 동적 짜기.

동적 짜기 표를 되짚어 최장 공통 부분 차례의 길이를 셈하고
실제 부분 차례를 되찾는다.
"""


# === 최장 공통 부분 차례 길이(2차원 표 채우기) ===

def lcs_length(x: str, y: str) -> int:
    """x과 y의 최장 공통 부분 차례 길이를 돌려준다. 시간: O(mn), 공간: O(mn)."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


# === 다시 세우기를 곁들인 최장 공통 부분 차례 ===

def lcs_with_string(x: str, y: str) -> tuple[int, str]:
    """최장 공통 부분 차례의 길이와 그 글줄 하나를 돌려준다."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # 거슬러 좇아 부분 차례를 되찾는다
    i, j = m, n
    result = []
    while i > 0 and j > 0:
        if x[i - 1] == y[j - 1]:
            result.append(x[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return dp[m][n], "".join(reversed(result))


# === 메인 ===

if __name__ == "__main__":
    x = "ABCBDAB"
    y = "BDCABA"

    length = lcs_length(x, y)
    length2, subseq = lcs_with_string(x, y)

    print(f"X = {x}")
    print(f"Y = {y}")
    print(f"LCS length: {length}")
    print(f"LCS string: {subseq} (length {length2})")
    # 내임:
    # X = ABCBDAB
    # Y = BDCABA
    # 최장 공통 부분 차례 길이: 4
    # 최장 공통 부분 차례 글줄: BCBA(길이 4)
```

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 14. MIT Press.

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
