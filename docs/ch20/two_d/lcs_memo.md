# 적어 두기로 하는 최장 공통 부분 차례

최장 공통 부분 차례 문제의 막무가내 되돌이 풀이는 같은 아래 문제를 되풀이해 셈하므로 지수 시간이 든다. 적어 두기는 되돌이 방식에 곳간을 붙인다. 곧 아래 문제를 한 번만 풀고 뒤이은 부름은 담아 둔 결과를 가져온다. 이 위에서 아래로 가는 전략은 자연스러운 되돌이 짜임을 지키면서 아래에서 위로 가는 표 채우기와 같은 $O(mn)$ 시간을 이룬다.

---

## 1. 되돌이에서 적어 두기로

수수한 되돌이 최장 공통 부분 차례는 앞머리 $X_i$과 $Y_j$의 최장 공통 부분 차례 길이인 $c(i, j)$을 다음으로 셈한다:

$$
c(i, j) = \begin{cases} 0 & \text{if } i = 0 \text{ or } j = 0 \\ c(i-1, j-1) + 1 & \text{if } x_i = y_j \\ \max(c(i-1, j),\, c(i, j-1)) & \text{if } x_i \ne y_j \end{cases}
$$

적어 두지 않으면 부름마다 아래 문제 둘로 갈라져 되돌이 나무의 마디가 최대 $2^{m+n}$개가 된다. 그러나 서로 다른 아래 문제 $(i, j)$은 $(m+1)(n+1)$개뿐이다. 적어 두기는 저마다 꼭 한 번씩만 셈되게 한다.

---

## 2. 적어 두기가 도는 법

1. 크기 $(m+1) \times (n+1)$인 2차원 표 `memo`을 파수 값(예컨대 $-1$)으로 첫자리매김한다.
2. $c(i, j)$을 셈하기 앞서 `memo[i][j]`에 이미 결과가 있는지 살핀다.
3. 곳간에 있으면 담아 둔 값을 곧바로 돌려준다.
4. 없으면 되돌이로 결과를 셈해 `memo[i][j]`에 담고 돌려준다.

!!! tip "적어 두기와 표 채우기"
    적어 두기는 본디 부름에서 닿는 아래 문제만 셈하므로 표의 칸이 상당수 필요 없을 때 이롭다. 표 채우기는 그와 무관하게 표 전체를 채우지만 되돌이 군더더기와 쌓임 깊이 한계를 피한다.

---

## 3. 복잡도

| 갈래 | 값 |
|---|---|
| 시간 | $O(mn)$ — 아래 문제 $(m+1)(n+1)$개를 한 번씩 셈한다 |
| 공간 | 적어 두기 표에 $O(mn)$, 되돌이 쌓임에 $O(m+n)$ |
| 쌓임 깊이 | 최악의 경우 $O(m + n)$ |

!!! warning "되돌이 깊이 한계"
    차례가 길면($m + n > 1000$) 되돌이 깊이가 파이썬의 붙박이 한계를 넘을 수 있다. `sys.setrecursionlimit()`을 쓰거나 아래에서 위로 가는 표 채우기로 바꾼다.

---

## 4. 파이썬 구현

```python
"""
최장 공통 부분 차례 — 적어 두기를 곁들인 되돌이 풀이.

최장 공통 부분 차례 문제의 위에서 아래로 가는 동적 짜기를 보이며
막무가내 되돌이 방식과 견준다.
"""

# === 막무가내 되돌이 최장 공통 부분 차례(지수) ===

def lcs_recursive(x: str, y: str, i: int, j: int) -> int:
    """막무가내 되돌이 최장 공통 부분 차례. 시간: O(2^(m+n)), 공간: 쌓임 O(m+n)."""
    if i == 0 or j == 0:
        return 0
    if x[i - 1] == y[j - 1]:
        return lcs_recursive(x, y, i - 1, j - 1) + 1
    return max(lcs_recursive(x, y, i - 1, j), lcs_recursive(x, y, i, j - 1))

# === 적어 둔 최장 공통 부분 차례 ===

def lcs_memo(x: str, y: str) -> int:
    """적어 두기로 얻는 길이. 시간: O(mn), 공간: O(mn)."""
    m, n = len(x), len(y)
    memo = [[-1] * (n + 1) for _ in range(m + 1)]

    def helper(i: int, j: int) -> int:
        if i == 0 or j == 0:
            return 0
        if memo[i][j] != -1:
            return memo[i][j]
        if x[i - 1] == y[j - 1]:
            memo[i][j] = helper(i - 1, j - 1) + 1
        else:
            memo[i][j] = max(helper(i - 1, j), helper(i, j - 1))
        return memo[i][j]

    return helper(m, n)

# === functools로 적어 둔 최장 공통 부분 차례 ===

def lcs_memo_functools(x: str, y: str) -> int:
    """파이썬에 딸린 lru_cache로 적어 두는 최장 공통 부분 차례."""
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def helper(i: int, j: int) -> int:
        if i == 0 or j == 0:
            return 0
        if x[i - 1] == y[j - 1]:
            return helper(i - 1, j - 1) + 1
        return max(helper(i - 1, j), helper(i, j - 1))

    return helper(len(x), len(y))

# === 메인 ===

if __name__ == "__main__":
    x = "ABCBDAB"
    y = "BDCABA"

    print(f"X = {x}")
    print(f"Y = {y}")
    print(f"LCS (naive):     {lcs_recursive(x, y, len(x), len(y))}")
    print(f"LCS (memo):      {lcs_memo(x, y)}")
    print(f"LCS (lru_cache): {lcs_memo_functools(x, y)}")
    # 내임:
    # X = ABCBDAB
    # Y = BDCABA
    # 최장 공통 부분 차례(막무가내):     4
    # 최장 공통 부분 차례(적어 두기):      4
    # 최장 공통 부분 차례(lru_cache): 4
```

**출력:**

```
X = ABCBDAB
Y = BDCABA
LCS (naive):     4
LCS (memo):      4
LCS (lru_cache): 4
```

---

## 5. 아래 문제 다시 쓰기 살피기

$X = \text{AB}$이고 $Y = \text{BA}$일 때 막무가내 되돌이 나무가 적어 두기가 왜 도움이 되는지 보여 준다:

```
c(2,2)
├── c(1,2)             [x2 ≠ y2]
│   ├── c(0,2) = 0
│   └── c(1,1)         [x1 = y1 = 'A'? No: A ≠ B]
│       ├── c(0,1) = 0
│       └── c(1,0) = 0
└── c(2,1)             [x2 ≠ y1]
    ├── c(1,1)         ← RECOMPUTED without memo
    │   ├── c(0,1) = 0
    │   └── c(1,0) = 0
    └── c(2,0) = 0
```

적어 두면 $c(1,1)$을 한 번 셈하고 두 번째 부름에서는 가져온다. 들임이 커질수록 아끼는 몫이 상수에서 지수로 커진다.

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

이 마당은 되돌이에서 적어 두기로、적어 두기가 도는 법、복잡도、파이썬 구현을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 14. MIT Press.
