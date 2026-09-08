# 구르는 흩는 값
구르는 흩는 값(미끄러지는 흩는 값이라고도 한다)은 크기가 정해진 창이 들임 위로 미끄러질 때 효율 좋게 새로 고칠 수 있는 흩는 함수이다. 라빈-카프 글줄 찾기 알고리즘의 고갱이 재주이며 보통의 경우 $O(n+m)$ 본 찾기를 되게 한다.

---

## 1. 다항 구르는 해시

가장 흔한 구르는 흩는 값은 글줄을 소수 $p$으로 나눈 나머지에서 따진 다항식으로 본다. 크기 $d$인 글자 모임의 글줄 $S[0..m-1]$에 대해:

$$
H(S[0..m-1]) = \left(\sum_{i=0}^{m-1} S[i] \cdot d^{m-1-i}\right) \bmod p
$$

창이 $S[i..i+m-1]$에서 $S[i+1..i+m]$으로 미끄러지면 흩는 값을 $O(1)$에 새로 고친다:

$$
H(S[i+1..i+m]) = \left(d \cdot \bigl(H(S[i..i+m-1]) - S[i] \cdot d^{m-1}\bigr) + S[i+m]\right) \bmod p
$$

$d^{m-1} \bmod p$을 한 번 미리 셈해 둔다.

```python
def rabin_karp(text: str, pattern: str, d: int = 256, p: int = 101) -> list[int]:
    """구르는 흩는 값을 쓴 라빈-카프 글줄 찾기."""
    n, m = len(text), len(pattern)
    if m == 0 or m > n:
        return []

    occurrences = []
    h = pow(d, m - 1, p)  # d^(m-1) mod p

    # 처음 해시값 계산
    p_hash = 0
    t_hash = 0
    for i in range(m):
        p_hash = (d * p_hash + ord(pattern[i])) % p
        t_hash = (d * t_hash + ord(text[i])) % p

    for i in range(n - m + 1):
        if p_hash == t_hash:
            # 글자마다 확인한다(헛맞음을 피한다)
            if text[i:i + m] == pattern:
                occurrences.append(i)
        if i < n - m:
            # 흩는 값을 앞으로 굴린다
            t_hash = (d * (t_hash - ord(text[i]) * h) + ord(text[i + m])) % p
            if t_hash < 0:
                t_hash += p

    return occurrences

# 예
text = "GEEKS FOR GEEKS"
pattern = "GEEK"
print(rabin_karp(text, pattern))
# 내놓기: [0, 10]
```

---

## 2. 복잡도 분석

- **미리 다듬기:** 본과 첫 창의 흩는 값을 셈하는 데 $O(m)$.
- **기대 찾기 시간:** $O(n + m)$. 창 새로 고침마다 $O(1)$이다. 헛맞음(흩는 값 부딪침)은 $O(m)$의 확인이 든다. 좋은 흩는 함수라면 헛맞음의 기댓값이 $O(n/p)$이고 $p$이 크면 작다.
- **최악의 경우:** 창마다 흩는 값이 부딪치면(예컨대 글자가 모두 같고 $p$을 잘못 골랐을 때) $O(nm)$.
- **공간:** 딸림 $O(1)$.

---

## 3. 좋은 매개변수 고르기

부딪침을 가장 적게 하려면 $p$을 큰 소수로, $d$을 글자 모임의 크기로 고른다. 서로 얽히지 않은 흩는 함수 둘을 쓰면(겹 흩기) 부딪침 확률이 대략 $1/p^2$으로 줄어 실전에서 헛맞음이 무시할 만해진다.

# 참고 문헌

[Rabin-Karp Algorithm - Wikipedia](https://en.wikipedia.org/wiki/Rabin%E2%80%93Karp_algorithm)

[Introduction to Algorithms (CLRS), Section 32.2 - The Rabin-Karp algorithm](https://mitpress.mit.edu/books/introduction-to-algorithms-fourth-edition/)

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

이 마당은 다항 구르는 해시、복잡도 분석、좋은 매개변수 고르기을 차례로 짚었다.
