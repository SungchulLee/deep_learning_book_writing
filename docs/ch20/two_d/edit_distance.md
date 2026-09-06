# 고침 거리

두 글줄은 얼마나 비슷한가? **고침 거리**(레벤슈타인 거리)는 한 글줄을 다른 글줄로 바꾸는 데 드는 글자 하나짜리 연산의 가장 적은 수를 센다. 이 잣대는 맞춤법 살피기, DNA 차례 맞추기, 차이 보기 도구, 자연말 처리의 바탕이다. 가장 좋은 아래 짜임과 겹치는 아래 문제를 갖춘 말끔한 동적 짜기 풀이가 있어 2차원 동적 짜기의 교과서 보기가 된다.

## 문제 서술

길이 $m$인 글줄 $s_1$과 길이 $n$인 글줄 $s_2$이 주어질 때 $s_1$을 $s_2$으로 바꾸는 데 드는 가장 적은 연산 수를 찾아라. 값이 저마다 1인 허락된 연산은 다음과 같다:

- $s_1$에 글자를 **끼운다**.
- $s_1$에서 글자를 **지운다**.
- $s_1$의 글자를 다른 글자로 **갈음한다**.

예컨대 "kitten"을 "sitting"으로 바꾸려면 연산 3번이 든다. 곧 'k'을 's'로 갈음하고, 'e'을 'i'로 갈음하고, 끝에 'g'을 끼운다.

## 가장 좋은 밑짜임

$d(i, j)$을 $s_1$의 앞선 $i$개 글자와 $s_2$의 앞선 $j$개 글자 사이 고침 거리라 하자. 마지막 글자 $s_1[i]$과 $s_2[j]$을 보자:

- $s_1[i] = s_2[j]$이면 글자가 맞으므로 연산이 필요 없다: $d(i, j) = d(i-1, j-1)$.
- $s_1[i] \neq s_2[j]$이면 세 가지 고름 가운데 가장 작은 것을 취한다:
    - $s_1[i]$을 $s_2[j]$으로 **갈음한다**: 값은 $1 + d(i-1, j-1)$.
    - $s_1[i]$을 **지운다**: 값은 $1 + d(i-1, j)$.
    - $s_1[i]$ 뒤에 $s_2[j]$을 **끼운다**: 값은 $1 + d(i, j-1)$.

## 점화식

$$
d(i, j) = \begin{cases} j & \text{if } i = 0 \\ i & \text{if } j = 0 \\ d(i-1, j-1) & \text{if } s_1[i] = s_2[j] \\ 1 + \min\bigl(d(i-1, j),\; d(i, j-1),\; d(i-1, j-1)\bigr) & \text{otherwise} \end{cases}
$$

**바탕 경우.** 길이 $i$인 글줄을 빈 글줄로 바꾸려면 $i$번 지워야 하므로 $d(i, 0) = i$이다. 빈 글줄을 길이 $j$인 글줄로 바꾸려면 $j$번 끼워야 하므로 $d(0, j) = j$이다.

## 표 채우기

동적 짜기 표를 가로줄마다 왼쪽에서 오른쪽으로 채운다. 칸 $d(i, j)$은 $d(i-1, j-1)$, $d(i-1, j)$, $d(i, j-1)$에만 기대며 이들은 모두 $d(i, j)$보다 먼저 셈된다.

"kitten"과 "sitting"에 대한 표는 다음과 같다:

|       |   | s | i | t | t | i | n | g |
|-------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|       | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| **k** | 1 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| **i** | 2 | 2 | 1 | 2 | 3 | 4 | 5 | 6 |
| **t** | 3 | 3 | 2 | 1 | 2 | 3 | 4 | 5 |
| **t** | 4 | 4 | 3 | 2 | 1 | 2 | 3 | 4 |
| **e** | 5 | 5 | 4 | 3 | 2 | 2 | 3 | 4 |
| **n** | 6 | 6 | 5 | 4 | 3 | 3 | 2 | 3 |

답 $d(6, 7) = 3$은 오른쪽 아래 구석에 있다.

## 구현

```python
"""
동적 짜기로 하는 고침 거리(레벤슈타인 거리).

한 글줄을 다른 글줄로 바꾸는 데 드는 가장 적은 끼우기, 지우기,
갈음하기의 수를 셈한다. 연산 거슬러 좇기와 공간을 줄인
변형을 곁들인다.
"""

# === 여느 동적 짜기 풀이 ===

def edit_distance(s1: str, s2: str) -> int:
    """두 글줄 사이 고침 거리를 셈한다.

    인수:
        s1: 본디 글줄.
        s2: 목표 글줄.

    반환값:
        가장 적은 고침 연산 수.
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 바탕 경우
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # 표를 채운다
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # 지우기
                    dp[i][j - 1],      # 끼우기
                    dp[i - 1][j - 1]   # 갈음하기
                )

    return dp[m][n]


# === 거슬러 좇아 연산 되찾기 ===

def edit_operations(s1: str, s2: str) -> list[str]:
    """고침 연산의 차례를 되찾는다.

    인수:
        s1: 본디 글줄.
        s2: 목표 글줄.

    반환값:
        연산 설명의 목록.
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1]
                )

    # 거슬러 좇기
    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i - 1] == s2[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(f"Replace '{s1[i-1]}' with '{s2[j-1]}' at position {i}")
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(f"Delete '{s1[i-1]}' at position {i}")
            i -= 1
        else:
            ops.append(f"Insert '{s2[j-1]}' at position {i + 1}")
            j -= 1

    ops.reverse()
    return ops


# === 공간을 줄인 판 ===

def edit_distance_optimized(s1: str, s2: str) -> int:
    """가로줄 둘을 쓰는, 공간을 줄인 고침 거리.

    인수:
        s1: 본디 글줄.
        s2: 목표 글줄.

    반환값:
        가장 적은 고침 연산 수.
    """
    m, n = len(s1), len(s2)
    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev

    return prev[n]


# === 시연 ===

if __name__ == "__main__":
    s1, s2 = "kitten", "sitting"

    dist = edit_distance(s1, s2)
    print(f"edit_distance('{s1}', '{s2}') = {dist}")

    ops = edit_operations(s1, s2)
    print(f"\nOperations ({len(ops)} total):")
    for op in ops:
        print(f"  {op}")

    dist_opt = edit_distance_optimized(s1, s2)
    print(f"\nSpace-optimized result: {dist_opt}")

    # 보기 하나 더
    a, b = "intention", "execution"
    print(f"\nedit_distance('{a}', '{b}') = {edit_distance(a, b)}")
```

**출력:**

```
edit_distance('kitten', 'sitting') = 3

Operations (3 total):
  Replace 'k' with 's' at position 1
  Replace 'e' with 'i' at position 5
  Insert 'g' at position 7

Space-optimized result: 3

edit_distance('intention', 'execution') = 5
```

## 복잡도

| 갈래 | 여느 것 | 공간 줄임 |
|--------|:--------:|:---------------:|
| Time   | $O(mn)$  | $O(mn)$         |
| Space  | $O(mn)$  | $O(\min(m, n))$ |

여느 풀이는 $(m+1) \times (n+1)$ 표를 채워 시간과 공간이 $O(mn)$이다. 가로줄마다 바로 앞 가로줄에만 기대므로 공간을 줄인 판은 가로줄 둘만 써서 공간을 $O(\min(m, n))$으로 줄인다. 맞바꿈은 거슬러 좇기(실제 연산 되찾기)에 표 전체가 필요하다는 것이다.

## 변형

- **무게 붙은 고침 거리.** 연산마다 값이 다르다. 되돌이 관계식에서 상수 1을 연산마다의 값 $c_{\text{ins}}, c_{\text{del}}, c_{\text{rep}}$으로 갈음한다.
- **최장 공통 부분 차례.** 갈음 값을 무한으로(또는 지우기 + 끼우기를 뜻하는 2로) 두면 고침 거리가 $m + n - 2 \cdot \text{LCS}(s_1, s_2)$과 같아진다.
- **다메라우-레벤슈타인 거리.** 네 번째 연산인 이웃한 두 글자 자리 바꿈을 더한다.

## 응용

- **맞춤법 살피기.** 고침 거리가 작은 사전 낱말을 찾아 고칠 것을 알려 준다.
- **DNA 차례 맞추기.** 유전 차례 사이 닮음을 잰다.
- **차이 보기 도구.** 파일 판 사이 가장 작은 바뀜 모임을 셈한다.
- **자연말 처리.** 찾기와 앎 찾아오기에서 어림잡아 글줄 맞추기.

## 참고 문헌

- Wagner, R. A., & Fischer, M. J. (1974). The string-to-string correction problem. *Journal of the ACM*, 21(1), 168--173.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 14: Dynamic Programming.
- Levenshtein, V. I. (1966). Binary codes capable of correcting deletions, insertions, and reversals. *Soviet Physics Doklady*, 10(8), 707--710.

## 연습문제

**연습문제 1.**
동적 짜기 표를 채워 "kitten"과 "sitting" 사이 고침 거리를 셈하라.

??? success "연습문제 1 풀이"
    고침 거리는 3이다. (1) kitten $\to$ sitten(k $\to$ s로 갈음), (2) sitten $\to$ sittin(e $\to$ i로 갈음), (3) sittin $\to$ sitting(g 끼움). 동적 짜기 표의 마지막 칸: $dp[6][7] = 3$. 칸마다 $dp[i][j] = \min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1] + (s_1[i] \neq s_2[j]))$이다. $\square$

---

**연습문제 2.**
고침 거리가 삼각 부등식 $d(A, C) \leq d(A, B) + d(B, C)$을 채움을 밝혀라.

??? success "연습문제 2 풀이"
    $A$을 $B$으로 바꾸는 연산($d(A,B)$이 듦) 뒤에 $B$을 $C$으로 바꾸는 연산($d(B,C)$이 듦)을 이으면 $A$을 $C$으로 바꾸는 데 모두 $d(A,B) + d(B,C)$이 든다. $d(A,C)$이 가장 적은 값이므로 $d(A,C) \leq d(A,B) + d(B,C)$이다. 그래서 고침 거리는 글줄 위의 잣대가 된다. $\square$

---

**연습문제 3.**
고침 거리 알고리즘의 공간 복잡도를 $O(mn)$에서 $O(\min(m, n))$으로 어떻게 줄이는가?

??? success "연습문제 3 풀이"
    동적 짜기 표의 가로줄마다 바로 앞 가로줄에만 기대므로 가로줄 둘(지금과 앞선 것)만 지닌다. 긴 글줄을 세로줄로, 짧은 글줄을 가로줄로 두고 되풀이하면 $O(\min(m,n))$ 공간을 쓴다. 그러면 실제 고침 연산을 다시 세울 수 없다(표 전체가 필요하다). 공간을 줄이고도 연산을 되찾으려면 히르슈베르크 알고리즘(나누어 이기기 + 공간을 아끼는 동적 짜기)을 $O(mn)$ 시간, $O(\min(m,n))$ 공간에 쓴다. $\square$

---

**연습문제 4.**
이웃한 글자의 자리 바꿈을 허락하도록 고침 거리를 고쳐라(다메라우-레벤슈타인 거리). 되돌이 관계식은 어떻게 바뀌는가?

??? success "연습문제 4 풀이"
    네 번째 연산을 더한다. 곧 $s_1[i] = s_2[j-1]$이고 $s_1[i-1] = s_2[j]$이면 값 1로 자리를 바꿀 수 있다. 되돌이 관계식은 $dp[i][j] = \min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1] + (s_1[i] \neq s_2[j]), dp[i-2][j-2] + 1)$이 되며 마지막 항은 자리 바꿈 조건이 성립할 때만 쓴다. "teh" $\to$ "the" 같은 흔한 오타를 잡는다. 시간과 공간은 그대로 $O(mn)$이다. $\square$
