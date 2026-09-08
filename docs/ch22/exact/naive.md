# 막무가내 알고리즘
막무가내 글줄 찾기 알고리즘은 글월 $T[0..n-1]$에서 본 $P[0..m-1]$이 나오는 곳을 모두 찾는 가장 단순한 길이다. 본을 글월 위로 한 자리씩 밀며 자리마다 맞는지 살핀다.

---

## 1. 알고리즘

가능한 맞춤 $i = 0, 1, \ldots, n - m$마다 $P[0..m-1]$을 $T[i..i+m-1]$과 글자마다 견준다. $m$개 글자가 모두 맞으면 자리 $i$에 나왔다고 알린다. 어느 자리든 맞지 않으면 다음 맞춤 $i+1$으로 옮긴다.

$$
\text{밀 거리 } s \in \{0, 1, \ldots, n-m\} \text{ 마다}: \quad \forall \, j \in \{0, \ldots, m-1\} \text{ 에 대해 } T[s+j] = P[j] \text{ 인지 살핀다}
$$

```python
def naive_search(text: str, pattern: str) -> list[int]:
    """글월에서 본이 나오는 시작 번호를 모두 돌려준다."""
    n, m = len(text), len(pattern)
    if m == 0:
        return []
    occurrences = []
    for i in range(n - m + 1):
        match = True
        for j in range(m):
            if text[i + j] != pattern[j]:
                match = False
                break
        if match:
            occurrences.append(i)
    return occurrences

# 예
text = "AABAACAADAABAABA"
pattern = "AABA"
print(naive_search(text, pattern))
# 내놓기: [0, 9, 12]
```

---

## 2. 복잡도 분석

- **최악의 경우:** $O((n - m + 1) \cdot m) = O(nm)$. 부분 맞음이 많을 때 생긴다. 예컨대 $T = \texttt{AAAA\ldots A}$이고 $P = \texttt{AAA\ldots AB}$일 때이다.
- **가장 좋은 경우:** $O(n)$. $P$의 첫 글자가 $T$에 아예 없으면 맞춤마다 곧바로 어긋난다.
- **보통의 경우:** 글자 모임이 큰 아무 글월에서는 맞춤 대부분이 빨리 어긋나므로 $O(n)$이다.
- **공간:** 딸림 공간 $O(1)$(내놓는 목록은 빼고).

막무가내 알고리즘은 바탕 노릇을 한다. 단순해서 짧은 본이나 작은 글월에는 쓸 만하지만 큰 규모의 찾기에는 KMP나 보이어-무어 같은 알고리즘이 낫다.

# 참고 문헌

[Introduction to Algorithms (CLRS), Section 32.1 - The naive string-matching algorithm](https://mitpress.mit.edu/books/introduction-to-algorithms-fourth-edition/)

[Naive Pattern Searching Algorithm](https://www.geeksforgeeks.org/naive-algorithm-for-pattern-searching/)

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

이 마당은 알고리즘、복잡도 분석을 차례로 짚었다.
