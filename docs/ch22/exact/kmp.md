# KMP 알고리즘

막무가내 글줄 찾기 알고리즘은 본을 한 자리씩 밀어 최악의 경우 $O(nm)$이 든다. **커누스-모리스-프랫(KMP)** 알고리즘은 맞지 않을 때 본을 얼마나 밀지 알려 주는 **어긋남 함수**(앞가지 함수라고도 한다)를 미리 셈해 겹치는 견줌을 없앤다. 그래서 $O(m)$ 공간을 더 써서 $O(n + m)$ 시간을 이룬다.

---

## 1. 핵심 통찰

글자 $j$개가 맞은 뒤 본의 자리 $j$에서 맞지 않으면 막무가내 방식은 다음 글월 자리에서 찾기를 다시 시작한다. KMP는 본의 어떤 앞가지가 이미 맞은 몫의 뒷가지와 맞을 수 있음을 알아채고, 그 글자를 다시 살피지 않고 본을 앞으로 밀 수 있게 한다.

---

## 2. 어긋남 함수

어긋남 함수 $\pi[j]$은 본 $P[0 \ldots j]$의 진앞가지이면서 뒷가지이기도 한 가장 긴 것의 길이를 담는다. 엄밀히:

$$
\pi[j] = \max\{k : 0 \le k < j \text{ and } P[0 \ldots k-1] = P[j-k+1 \ldots j]\}
$$

$\pi[0] = 0$이다(글자 하나에는 진앞가지가 없다).

**보기.** 본 $P = \text{``ABABAC''}$에 대해:

| $j$ | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| $P[j]$ | A | B | A | B | A | C |
| $\pi[j]$ | 0 | 0 | 1 | 2 | 3 | 0 |

$j = 4$에서 "ABABA"의 가장 긴 진앞가지-뒷가지는 "ABA"(길이 3)이다.

---

## 3. 어긋남 함수 세우기

$\pi[j]$이 $\pi[j-1]$을 넓힌다는 눈썰미로 어긋남 함수를 되풀이하며 $O(m)$ 시간에 셈한다:

1. $\pi[0] = 0$, $k = 0$으로 둔다.
2. $j = 1, 2, \ldots, m-1$에 대해:
    - $k > 0$이고 $P[k] \ne P[j]$인 동안 $k = \pi[k-1]$으로 둔다(물러난다).
    - $P[k] = P[j]$이면 $k$을 하나 늘린다.
    - $\pi[j] = k$으로 둔다.

---

## 4. 탐색 알고리즘

어긋남 함수를 미리 셈해 두면:

1. $j = 0$(본에서의 자리)으로 첫자리매김한다.
2. 글월의 글자 $T[i]$마다:
    - $j > 0$이고 $P[j] \ne T[i]$인 동안 $j = \pi[j-1]$으로 둔다.
    - $P[j] = T[i]$이면 $j$을 하나 늘린다.
    - $j = m$이면 자리 $i - m + 1$에서 맞음을 찾은 것이다. 찾기를 이어 가려 $j = \pi[j-1]$으로 둔다.

---

## 5. 복잡도

| 갈래 | 값 |
|---|---|
| 미리 다듬기 | $O(m)$ |
| 찾기 | $O(n)$ |
| 전체 | $O(n + m)$ |
| 공간 | 어긋남 함수에 $O(m)$ |

!!! tip "왜 O(n + m)인가?"
    글월의 글자마다 많아야 상수 번 견준다. $j$은 글월 글자마다 많아야 1씩 늘고 어긋남 이음을 따라갈 때만 줄어드는데, 줄어드는 총 횟수는 늘어나는 총 횟수 이하이다.

---

## 6. 파이썬 구현

```python
"""
커누스-모리스-프랫(KMP) 글줄 찾기 알고리즘.

본을 미리 다듬어 어긋남 함수를 세운 뒤 글월을 $O(n + m)$ 시간에
훑어 나오는 곳을 모두 찾는다.
"""

# === 어긋남 함수 ===

def compute_failure(pattern: str) -> list[int]:
    """KMP 어긋남 함수(앞가지 함수)를 세운다.

    failure[j] = pattern[0..j]의 진앞가지이면서 뒷가지이기도 한
    가장 긴 것의 길이.
    """
    m = len(pattern)
    failure = [0] * m
    k = 0

    for j in range(1, m):
        while k > 0 and pattern[k] != pattern[j]:
            k = failure[k - 1]
        if pattern[k] == pattern[j]:
            k += 1
        failure[j] = k

    return failure

# === KMP 찾기 ===

def kmp_search(text: str, pattern: str) -> list[int]:
    """KMP로 글월에서 본이 나오는 곳을 모두 찾는다.

    맞은 곳의 시작 번호 목록을 돌려준다.
    """
    n, m = len(text), len(pattern)
    if m == 0:
        return []

    failure = compute_failure(pattern)
    matches = []
    j = 0  # 본에서의 자리

    for i in range(n):
        while j > 0 and pattern[j] != text[i]:
            j = failure[j - 1]
        if pattern[j] == text[i]:
            j += 1
        if j == m:
            matches.append(i - m + 1)
            j = failure[j - 1]

    return matches

# === 메인 ===

if __name__ == "__main__":
    text = "ABABDABACDABABCABAB"
    pattern = "ABABCABAB"

    failure = compute_failure(pattern)
    matches = kmp_search(text, pattern)

    print(f"Text:    {text}")
    print(f"Pattern: {pattern}")
    print(f"Failure: {failure}")
    print(f"Matches at: {matches}")
    # 내임:
    # 글월:    ABABDABACDABABCABAB
    # 본: ABABCABAB
    # 어긋남: [0, 0, 1, 2, 0, 1, 2, 3, 4]
    # 맞은 자리: [9]
```

---

## 7. 풀이 예제

**글월:** `AABABAA`, **본:** `AABA`

"AABA"의 어긋남 함수: $\pi = [0, 1, 0, 1]$.

| 걸음 | $i$ | $T[i]$ | 앞의 $j$ | 맞는가? | 뒤의 $j$ |
|---|---|---|---|---|---|
| 1 | 0 | A | 0 | 예 | 1 |
| 2 | 1 | A | 1 | 예 | 2 |
| 3 | 2 | B | 2 | 아니오, $\pi[1]=1$으로 물러남. $P[1]$=A 대 B: 아니오, $\pi[0]=0$으로 물러남. $P[0]$=A 대 B: 아니오 | 0 |
| 4 | 3 | A | 0 | 예 | 1 |
| 5 | 4 | B | 1 | 아니오, $\pi[0]=0$으로 물러남. $P[0]$=A 대 B: 아니오 | 0 |
| 6 | 5 | A | 0 | 예 | 1 |
| 7 | 6 | A | 1 | 예 | 2 |

온전한 맞음을 찾지 못했다($j = 4$에 이르지 못했다).

---

## 연습문제

**연습문제 1.**
본 "ABABABCA"의 어긋남 함수(부분 맞음 표)를 셈하라.

??? success "연습문제 1 풀이"
    앞가지마다 처리한다:

    - A: 0, B: 0, A: 1, B: 2, A: 3, B: 4, C: 0, A: 1.
    어긋남 함수: $[0, 0, 1, 2, 3, 4, 0, 1]$. 자리 5(둘째 B)에서 "ABABAB"의 진앞가지이면서 뒷가지인 가장 긴 것은 "ABAB"(길이 4)이다. 자리 6(C)에서는 "ABABABC"의 어떤 진앞가지도 뒷가지가 아니므로 값이 0이다. $\square$

---

**연습문제 2.**
글월 "ABABCABABAB"에서 본 "ABAB"을 찾는 KMP를 좇아라. 언제 미는지 보여라.

??? success "연습문제 2 풀이"
    어긋남 함수: $[0, 0, 1, 2]$. A(0), B(1), A(2), B(3)이 맞아 자리 0에서 본을 찾았다. failure[3]=2로 밀어 본의 자리 2에서 찾기를 이어 간다. A(4)이 맞는가? 아니다(C 대 A). failure[1]=0으로 민다. 이어 간다: A(5), B(6), A(7), B(8)이 맞아 자리 5에서 찾았다. failure[3]=2로 민다. A(9), B(10)이 맞아 자리 7에서 찾았다. 맞은 자리는 모두 0, 5, 7이다. $\square$

---

**연습문제 3.**
$n$이 글월 길이, $m$이 본 길이일 때 KMP가 $O(n + m)$ 시간에 돎을 밝혀라.

??? success "연습문제 3 풀이"
    **어긋남 함수 셈하기**: $O(m)$. 되풀이마다 본 가리개를 앞으로 옮기거나(많아야 $m$번) 물리는데(앞으로 옮긴 총수 이하), 그렇다. **찾기 마디**: 글월 가리개 $i$은 많아야 $n$번 앞으로 가고 결코 물러나지 않는다. 본 가리개 $j$의 물러남은 $j$의 앞으로 감의 총수(많아야 $n$번) 이하이다. 찾기의 전체 연산은 $O(n)$이다. 합치면 $O(n + m)$이다. $\square$

---

**연습문제 4.**
KMP를 막무가내 글줄 찾기 알고리즘과 견주어라. 차이가 언제 가장 중요한가?

??? success "연습문제 4 풀이"
    막무가내: 최악의 경우 $O(nm)$(예컨대 글월 "AAAAAA"과 본 "AAAB"). KMP: 늘 $O(n + m)$. 본에 되풀이 짜임이 있어 막무가내 찾기에서 부분 맞음이 많이 나올 때 차이가 가장 크다. 아무 글월과 본에서는 막무가내 찾기의 평균이 $O(n)$이라 KMP와 겨룰 만하다. KMP의 이점은 선형 시간 최악의 경우가 보장된다는 것이며, 실시간 처리와 본에 주기 짜임이 있을 때 꼭 필요하다. $\square$

## 정리하며

이 마당은 핵심 통찰、어긋남 함수、어긋남 함수 세우기、탐색 알고리즘을 차례로 짚었다.

**참고 문헌**

- Knuth, D. E., Morris, J. H., & Pratt, V. R. (1977). Fast pattern matching in strings. *SIAM Journal on Computing*, 6(2), 323-350.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 32. MIT Press.
