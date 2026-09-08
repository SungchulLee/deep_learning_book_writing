# 나쁜 글자 규칙
나쁜 글자 규칙은 보이어-무어 알고리즘에서 맞지 않을 때 본을 얼마나 밀지 정하는 두 어림짐작 가운데 하나이다. 맞지 않게 만든 글월의 글자를 살펴 밀 거리를 셈한다.

---

## 1. 직관

본을 오른쪽에서 왼쪽으로 견줄 때 $T[i+j] \neq P[j]$처럼 맞지 않았다고 하자. 글자 $c = T[i+j]$이 "나쁜 글자"이다. 이 규칙은 $c$이 $P[0..j-1]$에서 마지막으로 어디에 나오는지 묻는다.

- $c$이 본의 자리 $k < j$에 나오면 $P[k]$이 $T[i+j]$과 맞도록 본을 민다. 미는 거리는 $j - k$이다.
- $c$이 $P[0..j-1]$에 없으면 본 전체를 맞지 않은 자리 너머로 민다. 미는 거리는 $j + 1$이다.

$$
\text{bad\_char\_shift}(j, c) = j - \max\{k : k < j \text{ and } P[k] = c\}
$$

그런 $k$이 없으면 $k = -1$을 써서 $j + 1$만큼 민다.

---

## 2. 미리 다듬기

글자마다 본에서 가장 오른쪽 자리를 담는 찾아보기 표를 세운다.

```python
def build_bad_character_table(pattern: str) -> dict[str, int]:
    """보이어-무어의 나쁜 글자 표를 세운다.

    글자마다 본에서 가장 오른쪽 자리에 대응시킨다.
    """
    table = {}
    for i, ch in enumerate(pattern):
        table[ch] = i
    return table

def bad_character_search(text: str, pattern: str) -> list[int]:
    """나쁜 글자 어림짐작만 쓴 보이어-무어."""
    n, m = len(text), len(pattern)
    if m == 0 or m > n:
        return []

    bad_char = build_bad_character_table(pattern)
    occurrences = []
    i = 0

    while i <= n - m:
        j = m - 1
        while j >= 0 and pattern[j] == text[i + j]:
            j -= 1
        if j < 0:
            occurrences.append(i)
            i += 1
        else:
            bc_pos = bad_char.get(text[i + j], -1)
            shift = j - bc_pos
            i += max(1, shift)
    return occurrences

# 예
text = "ABCABCABABC"
pattern = "ABABC"
print(bad_character_search(text, pattern))
# 내놓기: [6]
```

**출력:**

```
[6]
```

---

## 3. 넓힌 나쁜 글자 규칙

단순한 판은 글자마다 가장 오른쪽에 나온 곳만 담는다. **넓힌** 판은 본의 자리 $j$마다, 글자 $c$마다 $P[0..j-1]$에서 $c$이 가장 오른쪽에 나온 곳을 담는다. 더 잘 밀지만 $O(m \cdot |\Sigma|)$ 공간이 든다.

---

## 4. 복잡도

- **미리 다듬기:** 단순한 표는 $O(m + |\Sigma|)$, 넓힌 판은 $O(m \cdot |\Sigma|)$.
- **나쁜 글자 규칙만으로는 선형 아래나 선형의 최악의 경우 성능이 보장되지 않는다.** $O(nm)$으로 무너질 수 있다. 다만 좋은 뒷가지 규칙과 아우르면 보이어-무어가 최악의 경우 $O(n+m)$을 이룬다.

# 참고 문헌

[Boyer, Moore - A Fast String Searching Algorithm (1977)](https://doi.org/10.1145/359842.359859)

[Bad Character Heuristic - GeeksforGeeks](https://www.geeksforgeeks.org/boyer-moore-algorithm-for-pattern-searching/)

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

이 마당은 직관、미리 다듬기、넓힌 나쁜 글자 규칙、복잡도을 차례로 짚었다.
