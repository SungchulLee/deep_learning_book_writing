# 뒤집어도 같은 글 나누기

뒤집어도 같은 글은 앞으로 읽으나 뒤로 읽으나 같다. **뒤집어도 같은 글 나누기** 문제는 글줄이 주어질 때 저마다 뒤집어도 같은 부분 글줄로 나누는 데 드는 가장 적은 자르기 수를 묻는다. 이 문제는 동적 짜기 아래 문제 둘, 곧 어느 부분 글줄이 뒤집어도 같은지 미리 셈하는 것과 가장 좋은 자르는 자리를 찾는 것을 아울러 2차원 동적 짜기의 풍성한 보기가 된다.

---

## 1. 문제 서술

길이 $n$인 글줄 $s$이 주어질 때 $s_i$마다 뒤집어도 같도록 $s$을 부분 글줄 $s_1, s_2, \ldots, s_k$으로 나누는 데 드는 가장 적은 자르기 수를 찾아라.

**보기.** $s = \text{``aab''}$이면 자리 1 뒤에서 한 번 자르면 $\{\text{``aa''}, \text{``b''}\}$이 되고 둘 다 뒤집어도 같다. 가장 적은 자르기 수는 1이다.

---

## 2. 걸음 1 — 뒤집어도 같은지 표

먼저 부분 글줄 $s[i \ldots j]$이 뒤집어도 같은지 적어 두는 참거짓 표 $P[i][j]$을 미리 셈한다. 부분 글줄 $s[i \ldots j]$이 뒤집어도 같을 필요충분조건은 다음과 같다:

$$
P[i][j] = \begin{cases} \text{true} & \text{if } i = j \\ s[i] = s[j] & \text{if } j = i + 1 \\ s[i] = s[j] \text{ and } P[i+1][j-1] & \text{if } j > i + 1 \end{cases}
$$

$P[i+1][j-1]$이 필요할 때 늘 마련되어 있도록 부분 글줄 길이가 늘어나는 차례로 이 표를 채운다.

---

## 3. 걸음 2 — 가장 적은 자르기

$\text{cuts}[j]$을 $s[0 \ldots j]$을 뒤집어도 같은 글로 나누는 데 드는 가장 적은 자르기 수라 하자:

$$
\text{cuts}[j] = \begin{cases} 0 & \text{if } P[0][j] = \text{true} \\ \displaystyle \min_{0 \le i \le j,\, P[i][j]} \bigl\{ \text{cuts}[i-1] + 1 \bigr\} & \text{otherwise} \end{cases}
$$

답은 $\text{cuts}[n-1]$이다.

!!! tip "직관"
    자리 $j$마다 마지막이 될 수 있는 뒤집어도 같은 글 $s[i \ldots j]$을 모두 살핀다. 뒤집어도 같으면 값은 자르기 한 번에 $s[0 \ldots i-1]$의 가장 좋은 풀이를 더한 것이다.

---

## 4. 복잡도

| 갈래 | 값 |
|---|---|
| 시간 | $O(n^2)$ |
| 공간 | 뒤집어도 같은지 표에 $O(n^2)$, 자르기 배열에 $O(n)$ |
| 아래 문제 | 뒤집어도 같은지 살피기 $O(n^2)$ + 자르기 셈하기 $O(n)$ |

---

## 5. 풀이 예제

$s = \text{``abac''}$일 때(0부터 셈):

**뒤집어도 같은지 표 $P$:**

| $P[i][j]$ | a | b | a | c |
|---|---|---|---|---|
| a | T | F | T | F |
| b | | T | F | F |
| a | | | T | F |
| c | | | | T |

**자르기 배열:**

- $\text{cuts}[0] = 0$(``a''은 뒤집어도 같다)
- $\text{cuts}[1] = 1$(``ab''은 뒤집으면 다르다. 가장 좋은 것: ``a'' | ``b'')
- $\text{cuts}[2] = 0$(``aba''은 뒤집어도 같다)
- $\text{cuts}[3] = 1$(``abac''은 뒤집으면 다르다. 가장 좋은 것: ``aba'' | ``c'')

가장 적은 자르기: 1.

---

## 6. 파이썬 구현

```python
"""
뒤집어도 같은 글 나누기 — 동적 짜기로 하는 가장 적은 자르기.

뒤집어도 같은지 표를 미리 셈한 뒤 글줄을 뒤집어도 같은 부분 글줄로
나누는 가장 적은 자르기 수를 찾는다.
"""

# === 뒤집어도 같은 글 나누기 ===

def min_palindrome_cuts(s: str) -> int:
    """뒤집어도 같은 글 나누기의 가장 적은 자르기 수를 돌려준다.

    시간: O(n^2), 공간: O(n^2).
    """
    n = len(s)
    if n <= 1:
        return 0

    # 걸음 1: 뒤집어도 같은지 표를 세운다
    is_pal = [[False] * n for _ in range(n)]

    for i in range(n):
        is_pal[i][i] = True

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if length == 2:
                is_pal[i][j] = (s[i] == s[j])
            else:
                is_pal[i][j] = (s[i] == s[j]) and is_pal[i + 1][j - 1]

    # 걸음 2: 가장 적은 자르기를 셈한다
    cuts = [0] * n
    for j in range(n):
        if is_pal[0][j]:
            cuts[j] = 0
        else:
            cuts[j] = j  # 최악의 경우: 글자마다 자른다
            for i in range(1, j + 1):
                if is_pal[i][j]:
                    cuts[j] = min(cuts[j], cuts[i - 1] + 1)

    return cuts[n - 1]

# === 나눔 다시 세우기 ===

def palindrome_partition(s: str) -> list[str]:
    """가장 좋은 뒤집어도 같은 글 나눔 하나를 돌려준다."""
    n = len(s)
    if n <= 1:
        return [s] if s else []

    is_pal = [[False] * n for _ in range(n)]
    for i in range(n):
        is_pal[i][i] = True
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if length == 2:
                is_pal[i][j] = (s[i] == s[j])
            else:
                is_pal[i][j] = (s[i] == s[j]) and is_pal[i + 1][j - 1]

    cuts = [0] * n
    split_at = [-1] * n
    for j in range(n):
        if is_pal[0][j]:
            cuts[j] = 0
            split_at[j] = 0
        else:
            cuts[j] = j
            split_at[j] = j
            for i in range(1, j + 1):
                if is_pal[i][j] and cuts[i - 1] + 1 < cuts[j]:
                    cuts[j] = cuts[i - 1] + 1
                    split_at[j] = i

    # 거슬러 좇아 나눔을 세운다
    parts = []
    j = n - 1
    while j >= 0:
        i = split_at[j]
        parts.append(s[i:j + 1])
        j = i - 1

    return list(reversed(parts))

# === 메인 ===

if __name__ == "__main__":
    test_cases = ["aab", "abac", "abcba", "abcdef"]
    for s in test_cases:
        num_cuts = min_palindrome_cuts(s)
        partition = palindrome_partition(s)
        print(f"'{s}' -> {num_cuts} cuts: {partition}")
    # 내임:
    # 'aab' -> 자르기 1번: ['aa', 'b']
    # 'abac' -> 자르기 1번: ['aba', 'c']
    # 'abcba' -> 자르기 0번: ['abcba']
    # 'abcdef' -> 자르기 5번: ['a', 'b', 'c', 'd', 'e', 'f']
```

**출력:**

```
'aab' -> 1 cuts: ['aa', 'b']
'abac' -> 1 cuts: ['aba', 'c']
'abcba' -> 0 cuts: ['abcba']
'abcdef' -> 5 cuts: ['a', 'b', 'c', 'd', 'e', 'f']
```

---

## 연습문제

**연습문제 1.**
뒤집어도 같은 글 나누기의 상태, 옮아감, 바탕 경우를 가려내어라.

??? success "연습문제 1 풀이"
    **상태**는 아래 문제를 적는 데 필요한 앎을 담는다. **옮아감**(되돌이 관계식)은 어떤 상태의 가장 좋은 값을 더 작은 상태로 나타낸다. **바탕 경우**는 곧바로 풀 수 있는 가장 작은 아래 문제의 값을 준다. 이 셋이 함께 동적 짜기 풀이를 온전히 정한다. $\square$

---

**연습문제 2.**
뒤집어도 같은 글 나누기의 위에서 아래로(적어 두기) 짜기와 아래에서 위로(표 채우기) 짜기를 견주어라. 어느 쪽이 나으며 왜 그런가?

??? success "연습문제 2 풀이"
    **위에서 아래로**: 곳간을 곁들인 되돌이. 정말 필요한 아래 문제만 셈한다(게으른 값매김). 되돌이 관계식에서 옮겨 적기 쉽다. 되돌이 깊이 문제가 생길 수 있다. **아래에서 위로**: 되풀이로 기댐 차례에 따라 표를 채운다. 필요 없는 것까지 모든 아래 문제를 셈한다. 되돌이 군더더기가 없다. 공간을 줄이기 쉽다. 이 문제에서는 아래 문제가 모두 필요하면 아래에서 위로가 흔히 낫고, 닿지 않는 아래 문제가 많으면 위에서 아래로가 낫다. $\square$

---

**연습문제 3.**
뒤집어도 같은 글 나누기의 시간 복잡도와 공간 복잡도는 무엇인가? 공간을 더 줄일 수 있는가?

??? success "연습문제 3 풀이"
    시간 복잡도는 상태의 수에 상태마다의 옮아감 값을 곱한 것으로 정해진다. 공간은 담아 두는 상태의 수와 같다. 옮아감이 앞선 상태 가운데 한정된 몇 개에만 기대면(예컨대 2차원 표의 바로 앞 가로줄) 그 상태만 기억 공간에 두어 공간을 줄일 수 있으며, 흔히 $O(n^2)$에서 $O(n)$으로 줄어든다. $\square$

---

**연습문제 4.**
뒤집어도 같은 글 나누기의 알고리즘을 네가 고른 작은 보기에 대해 좇아라. 동적 짜기 표의 값을 보여라.

??? success "연습문제 4 풀이"
    작은 들임(예컨대 $n = 5$이나 짧은 글줄/배열)을 골라라. 동적 짜기 표를 한 걸음씩 채우면서 각 칸이 앞서 셈한 칸에서 어떻게 나오는지 보여라. 마지막 답을 막무가내로 다 세어 본 것과 견주어 확인하라. 이렇게 좇아 보면 되돌이 관계식이 옳음을 확인하고 알고리즘에 대한 직관이 선다. $\square$

## 정리하며

이 마당은 문제 서술、걸음 1 — 뒤집어도 같은지 표、걸음 2 — 가장 적은 자르기、복잡도을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 14. MIT Press.
