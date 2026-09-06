# 낱말 나누기

낱말 나누기 문제는 주어진 글줄을 사전에 있는 낱말의 차례로 나눌 수 있는지 묻는다. 상태가 글줄에서의 자리를 뜻하고 옮아감이 그 자리에서 끝나는 가능한 마지막 낱말을 모두 살피는, 1차원 동적 짜기의 자연스러운 쓰임새이다. 자연말 처리(토막내기), 찾기 물음 뜯어보기, 도메인 이름 살피기에서 나온다.

## 문제 서술

길이 $n$인 글줄 $s$과 사전 $D$(옳은 낱말의 모임)이 주어질 때 $s$을 저마다 $D$에 든 낱말 하나 이상으로 나눌 수 있는지 정하라.

**보기:** $s = \texttt{"leetcode"}$이고 $D = \{\texttt{"leet"}, \texttt{"code"}\}$이면 $s$을 $\texttt{"leet"} + \texttt{"code"}$으로 나눌 수 있으므로 답은 참이다.

**보기:** $s = \texttt{"catsandog"}$이고 $D = \{\texttt{"cats"}, \texttt{"dog"}, \texttt{"sand"}, \texttt{"and"}, \texttt{"cat"}\}$이면 답은 거짓이다.

## 점화식

$dp[i]$을 앞머리 $s[0..i-1]$(처음 $i$개 글자)을 사전 낱말로 나눌 수 있으면 참이라 하자. 자리 $i$마다 $0 \le j < i$인 가능한 마지막 낱말 $s[j..i-1]$을 모두 살핀다:

$$
dp[i] = \bigvee_{\substack{0 \le j < i \\ s[j..i-1] \in D}} dp[j]
$$

바탕 경우는 $dp[0] = \text{true}$이다(빈 앞머리는 뻔히 나뉜다).

말로 하면, 닿을 수 있는 앞선 자리 $j$이 있고 $j$부터 $i$까지의 부분 글줄이 옳은 사전 낱말이면 자리 $i$에 닿을 수 있다.

## 표 채우기

```python
"""
낱말 나누기: 글줄을 사전 낱말로 나눌 수 있는지 정한다.
"""


# ===================================================================
# 방식 1: 표 채우기(아래에서 위로)
# ===================================================================
def word_break(s: str, word_dict: list[str]) -> bool:
    """s을 나눌 수 있는지 살핀다. 시간: O(n^2 * L), 공간: O(n)."""
    n = len(s)
    words = set(word_dict)
    dp = [False] * (n + 1)
    dp[0] = True

    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in words:
                dp[i] = True
                break

    return dp[n]
```

바깥 되풀이가 $n$번 돈다. 안쪽 되풀이는 최대 $n$번 돌고 부분 글줄 살피기마다 $O(L)$이 든다. 여기서 $L$은 가장 긴 낱말의 길이이다. 전체 시간은 $O(n^2 L)$이며, $j$을 낱말 길이 창으로만 좁히면 나아진다.

## 가장 긴 낱말 길이로 줄이기

사전 낱말의 길이에 한계가 있으므로 안쪽 되풀이를 좁힐 수 있다:

```python
# ===================================================================
# 방식 2: 가장 긴 낱말 길이로 줄임
# ===================================================================
def word_break_optimized(s: str, word_dict: list[str]) -> bool:
    """안쪽 되풀이 범위를 좁힌 판. 시간: O(n * L), 공간: O(n)."""
    n = len(s)
    words = set(word_dict)
    max_len = max(len(w) for w in words) if words else 0
    dp = [False] * (n + 1)
    dp[0] = True

    for i in range(1, n + 1):
        for j in range(max(0, i - max_len), i):
            if dp[j] and s[j:i] in words:
                dp[i] = True
                break

    return dp[n]
```

가장 긴 낱말 길이 $L$까지의 부분 글줄만 살피면 자리마다 안쪽 되풀이가 최대 $L$번 돌아 $O(nL)$ 시간이 든다.

## 나눈 것 다시 세우기

실제 낱말 나눔을 찾으려면 어느 자르는 점이 옳은 나눔으로 이어지는지 좇는다:

```python
# ===================================================================
# 방식 3: 다시 세우기 곁들임
# ===================================================================
def word_break_segment(s: str, word_dict: list[str]) -> list[str] | None:
    """옳은 나눔 하나를 돌려주고, 할 수 없으면 None을 돌려준다."""
    n = len(s)
    words = set(word_dict)
    dp = [False] * (n + 1)
    parent = [-1] * (n + 1)
    dp[0] = True

    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in words:
                dp[i] = True
                parent[i] = j
                break

    if not dp[n]:
        return None

    # 되짚어 낱말을 되찾는다
    result = []
    idx = n
    while idx > 0:
        result.append(s[parent[idx]:idx])
        idx = parent[idx]

    return list(reversed(result))
```

## 복잡도

| 방법 | 시간 | 공간 |
|----------|------|-------|
| 기본 표 채우기 | $O(n^2 L)$ | $O(n)$ |
| 줄임 | $O(nL)$ | $O(n + \|D\|)$ |
| 다시 세우기 곁들임 | $O(n^2 L)$ | $O(n)$ |

여기서 $n$은 글줄의 길이, $L$은 가장 긴 낱말의 길이, $|D|$은 사전의 크기이다.

```python
# ===================================================================
# 메인
# ===================================================================
if __name__ == "__main__":
    test_cases = [
        ("leetcode", ["leet", "code"]),
        ("applepenapple", ["apple", "pen"]),
        ("catsandog", ["cats", "dog", "sand", "and", "cat"]),
    ]
    for s, dictionary in test_cases:
        result = word_break(s, dictionary)
        segmentation = word_break_segment(s, dictionary)
        print(f"s='{s}' -> {result}, segmentation={segmentation}")
```

**출력:**
```
s='leetcode' -> True, segmentation=['leet', 'code']
s='applepenapple' -> True, segmentation=['apple', 'pen', 'apple']
s='catsandog' -> False, segmentation=None
```

!!! note "모든 나눔을 찾는 변형"
    더 어려운 변형은 **가능한 모든** 나눔을 묻는다. 적어 두기를 곁들인 되짚기로 옳은 자르기를 모두 살펴야 하며 내놓는 것의 크기가 지수가 될 수 있다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 14. MIT Press.

## 연습문제

**연습문제 1.**
낱말 나누기의 상태, 옮아감, 바탕 경우를 가려내어라.

??? success "연습문제 1 풀이"
    **상태**는 아래 문제를 적는 데 필요한 앎을 담는다. **옮아감**(되돌이 관계식)은 어떤 상태의 가장 좋은 값을 더 작은 상태로 나타낸다. **바탕 경우**는 곧바로 풀 수 있는 가장 작은 아래 문제의 값을 준다. 이 셋이 함께 동적 짜기 풀이를 온전히 정한다. $\square$

---

**연습문제 2.**
낱말 나누기의 위에서 아래로(적어 두기) 짜기와 아래에서 위로(표 채우기) 짜기를 견주어라. 어느 쪽이 나으며 왜 그런가?

??? success "연습문제 2 풀이"
    **위에서 아래로**: 곳간을 곁들인 되돌이. 정말 필요한 아래 문제만 셈한다(게으른 값매김). 되돌이 관계식에서 옮겨 적기 쉽다. 되돌이 깊이 문제가 생길 수 있다. **아래에서 위로**: 되풀이로 기댐 차례에 따라 표를 채운다. 필요 없는 것까지 모든 아래 문제를 셈한다. 되돌이 군더더기가 없다. 공간을 줄이기 쉽다. 이 문제에서는 아래 문제가 모두 필요하면 아래에서 위로가 흔히 낫고, 닿지 않는 아래 문제가 많으면 위에서 아래로가 낫다. $\square$

---

**연습문제 3.**
낱말 나누기의 시간 복잡도와 공간 복잡도는 무엇인가? 공간을 더 줄일 수 있는가?

??? success "연습문제 3 풀이"
    시간 복잡도는 상태의 수에 상태마다의 옮아감 값을 곱한 것으로 정해진다. 공간은 담아 두는 상태의 수와 같다. 옮아감이 앞선 상태 가운데 한정된 몇 개에만 기대면(예컨대 2차원 표의 바로 앞 가로줄) 그 상태만 기억 공간에 두어 공간을 줄일 수 있으며, 흔히 $O(n^2)$에서 $O(n)$으로 줄어든다. $\square$

---

**연습문제 4.**
낱말 나누기의 알고리즘을 네가 고른 작은 보기에 대해 좇아라. 동적 짜기 표의 값을 보여라.

??? success "연습문제 4 풀이"
    작은 들임(예컨대 $n = 5$이나 짧은 글줄/배열)을 골라라. 동적 짜기 표를 한 걸음씩 채우면서 각 칸이 앞서 셈한 칸에서 어떻게 나오는지 보여라. 마지막 답을 막무가내로 다 세어 본 것과 견주어 확인하라. 이렇게 좇아 보면 되돌이 관계식이 옳음을 확인하고 알고리즘에 대한 직관이 선다. $\square$
