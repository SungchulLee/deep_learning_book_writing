# 보이어-무어
보이어-무어 알고리즘은 딱 맞는 글줄 찾기에서 실전 효율이 가장 좋은 알고리즘 가운데 하나이다. 본을 글월에 대고 오른쪽에서 왼쪽으로 훑으며, 맞지 않은 뒤 나쁜 글자 규칙과 좋은 뒷가지 규칙이라는 두 어림짐작으로 글월의 큰 몫을 건너뛴다.

---

## 1. 핵심 생각

본을 왼쪽에서 오른쪽으로 훑는 KMP와 달리 보이어-무어는 맞춰 놓을 때마다 본을 오른쪽에서 왼쪽으로 견준다. 본의 자리 $j$(글월 자리 $i + j$)에서 맞지 않으면 알고리즘이 밀 거리 둘을 셈한다:

1. **나쁜 글자 밀기:** 맞지 않은 글월 글자를 바탕으로.
2. **좋은 뒷가지 밀기:** 이미 맞은 본의 뒷가지를 바탕으로.

알고리즘은 두 밀 거리 가운데 큰 것을 취해 앞으로 나아감을 보장한다.

$$
\text{shift} = \max(\text{bad\_character\_shift}, \text{good\_suffix\_shift})
$$

---

## 2. 온전한 알고리즘

```python
def boyer_moore(text: str, pattern: str) -> list[int]:
    """어림짐작 둘을 다 쓴 보이어-무어 글줄 찾기."""
    n, m = len(text), len(pattern)
    if m == 0 or m > n:
        return []

    # 나쁜 글자 표
    bad_char = {}
    for i in range(m):
        bad_char[pattern[i]] = i

    # 좋은 뒷가지 표
    suffix = [0] * m
    suffix[m - 1] = m
    g = m - 1
    f = 0
    for i in range(m - 2, -1, -1):
        if i > g and suffix[i + m - 1 - f] < i - g:
            suffix[i] = suffix[i + m - 1 - f]
        else:
            g = min(g, i)
            f = i
            while g >= 0 and pattern[g] == pattern[g + m - 1 - f]:
                g -= 1
            suffix[i] = f - g

    good_suffix = [m] * m
    j = 0
    for i in range(m - 1, -1, -1):
        if suffix[i] == i + 1:
            while j < m - 1 - i:
                if good_suffix[j] == m:
                    good_suffix[j] = m - 1 - i
                j += 1
    for i in range(m - 2):
        good_suffix[m - 1 - suffix[i]] = m - 1 - i

    # 탐색
    occurrences = []
    i = 0
    while i <= n - m:
        j = m - 1
        while j >= 0 and pattern[j] == text[i + j]:
            j -= 1
        if j < 0:
            occurrences.append(i)
            i += good_suffix[0]
        else:
            bc_shift = j - bad_char.get(text[i + j], -1)
            gs_shift = good_suffix[j]
            i += max(bc_shift, gs_shift)
    return occurrences

# 예
text = "TRUSTHARDTOOTHBRUSHES"
pattern = "TOOTH"
print(boyer_moore(text, pattern))
# 내놓기: [9]
```

**출력:**

```
[9]
```

---

## 3. 복잡도 분석

- **미리 다듬기:** $|\Sigma|$이 글자 모임의 크기일 때 $O(m + |\Sigma|)$.
- **가장 좋은 경우:** $O(n/m)$. 본의 마지막 글자가 글월에 없으면 걸음마다 자리 $m$개를 건너뛸 수 있으며 이는 선형 아래이다.
- **최악의 경우:** 기본 알고리즘으로는 $O(nm)$. 갈릴 규칙으로 다듬으면 최악의 경우가 $O(n + m)$이 된다.
- **보통의 경우:** 글자 모임이 크면 $O(n/m)$이라 실전에서 가장 빠른 알고리즘 가운데 하나이다.

보이어-무어는 여러 글월 편집기와 유닉스 `grep` 도구가 고르는 알고리즘이다.

# 참고 문헌

[Boyer, Moore - A Fast String Searching Algorithm (1977)](https://doi.org/10.1145/359842.359859)

[Boyer-Moore String Search Algorithm - Wikipedia](https://en.wikipedia.org/wiki/Boyer%E2%80%93Moore_string-search_algorithm)

---

## 연습문제

**연습문제 1.**
막무가내 글자열 짝짓기 알고리즘, KMP, 보이어-무어의 가장 나쁜 경우 시간 복잡도를 견주어라.

??? success "연습문제 1 풀이"
    | 알고리즘 | 가장 나쁜 경우 | 가장 좋은 경우 | 공간 |
    |-----------|-----------|-----------|-------|
    | 막무가내 | $O(nm)$ | $O(n)$ | $O(1)$ |
    | KMP | $O(n + m)$ | $O(n)$ | 어그러짐 함수에 $O(m)$ |
    | 보이어-무어 | $O(nm)$(병적인 경우) | $O(n/m)$(선형 아래!) | $O(m + |\Sigma|)$ |

    KMP는 한 줄 시간을 보장한다. 보이어-무어는 (글자를 건너뛰므로) 실전에서 대개 더 빠르지만 갈릴 다듬기를 쓰지 않으면 가장 나쁜 경우 $O(nm)$이다.

---

**연습문제 2.**
글 $T$ = "ABABCABABD"과 무늬 $P$ = "ABABD"에 대해 알고리즘이 도는 과정을 견줌마다 보이며 좇아라.

??? success "연습문제 2 풀이"
    자리 0에서 시작: P[0]='A'와 T[0]='A' 견줌(맞음), P[1]='B'와 T[1]='B'(맞음), P[2]='A'와 T[2]='A'(맞음), P[3]='B'와 T[3]='B'(맞음), P[4]='D'와 T[4]='C'(어긋남). 어그러짐 함수 또는 밀기 규칙으로 무늬를 민다. 자리 2에서 시작(KMP는 어그러짐 함수로 다시 견주지 않는다). 끝내 자리 5에서 맞는 곳을 찾는다. 이 알고리즘은 모두 많아야 $2n$번 견준다.

---

**연습문제 3.**
KMP의 어그러짐 함수란 무엇인가? 무늬 "ABABCAB"에 대해 셈하여라.

??? success "연습문제 3 풀이"
    어긋남 함수 $\pi[i]$은 $P[0..i]$의 앞가지이면서 뒷가지이기도 한 가장 긴 진앞가지의 길이를 준다. "ABABCAB"에 대해:

    | $i$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
    |-----|---|---|---|---|---|---|---|
    | $P[i]$ | A | B | A | B | C | A | B |
    | $\pi[i]$ | 0 | 0 | 1 | 2 | 0 | 1 | 2 |

    예컨대 "AB"이 "ABAB"의 앞가지이자 뒷가지이므로 $\pi[3] = 2$이다.

---

**연습문제 4.**
라빈-카프에 쓰이는 굴리는 해시 재주를 설명하여라. 헛맞음이 일어날 확률은 얼마인가?

??? success "연습문제 4 풀이"
    라빈-카프는 본의 흩는 값을 셈하고 글월 위로 흩는 창을 미끄러뜨린다. **구르는 흩는 값**은 $O(1)$에 새로 고친다. 곧 $d$이 밑이고 $q$이 소수일 때 $h(T[i+1..i+m]) = (h(T[i..i+m-1]) - T[i] \cdot d^{m-1}) \cdot d + T[i+m] \pmod{q}$이다. 흩는 값은 맞는데 글줄이 다르면 헛맞음이 난다. 아무 소수 $q$에 대해 헛맞음 한 번의 확률은 $O(1/q)$이고 자리 $n-m+1$개에 대한 헛맞음의 기댓값은 $O(n/q)$이다. $q \approx n^2$을 고르면 헛맞음이 기대상 $O(1)$이다.

## 정리하며

이 마당은 핵심 생각、온전한 알고리즘、복잡도 분석을 차례로 짚었다.
