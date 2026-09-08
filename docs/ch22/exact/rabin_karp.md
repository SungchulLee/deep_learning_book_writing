# 라빈-카프 알고리즘

막무가내 글줄 찾기는 글월 자리마다 본의 글자를 모두 견주어 최악의 경우 $O(nm)$이 든다. **라빈-카프 알고리즘**은 흩기를 써서 기대 글자 견줌 횟수를 줄인다. 곧 글줄을 글자마다 견주는 대신 흩는 값을 먼저 견주고 그것이 맞을 때만 온전히 견준다. **구르는 흩는 값**이 흩는 값 새로 고침을 $O(1)$으로 만들어 기대 시간이 $O(n + m)$이 된다.

---

## 1. 구르는 흩는 값

구르는 흩는 값은 앞선 흩는 값을 새로 고쳐 글월의 길이 $m$짜리 부분 글줄마다의 흩는 값을 $O(1)$ 시간에 셈한다. 밑 $d$과 나눔수 $q$의 다항 흩기를 쓰면:

$$
h(s[i \ldots i+m-1]) = \left(\sum_{k=0}^{m-1} s[i+k] \cdot d^{m-1-k}\right) \bmod q
$$

창을 자리 $i$에서 $i+1$으로 미끄러뜨릴 때:

$$
h(s[i+1 \ldots i+m]) = \bigl(d \cdot (h(s[i \ldots i+m-1]) - s[i] \cdot d^{m-1}) + s[i+m]\bigr) \bmod q
$$

이 새로 고침은 가장 왼쪽 글자의 몫을 빼고 새로 온 가장 오른쪽 글자를 넣는다.

---

## 2. 알고리즘

1. 본 $P$의 흩는 값과 첫 창 $T[0 \ldots m-1]$의 흩는 값을 셈한다.
2. $h = d^{m-1} \bmod q$을 미리 셈한다.
3. 자리 $i$을 0부터 $n - m$까지:
    - 흩는 값이 맞으면 (헛맞음을 피하려) 글자마다 견주어 확인한다.
    - 구르는 새로 고침으로 다음 창의 흩는 값을 셈한다.

---

## 3. 복잡도

| 갈래 | 값 |
|---|---|
| 기대 시간 | $O(n + m)$ |
| 최악의 경우 시간 | $O(nm)$(흩는 값 부딪침이 많을 때) |
| 공간 | 들임 말고 $O(1)$ |
| 미리 다듬기 | $O(m)$ |

!!! warning "흩는 값 부딪침"
    흩는 값은 맞는데 글줄이 다르면(헛맞음) $O(m)$의 온전한 견줌이 필요하다. 큰 소수 $q$을 고르면 부딪침이 가장 적어진다. 헛맞음의 기댓값은 $O(n/q)$이며 $q$이 크면 무시할 만하다.

---

## 4. 파이썬 구현

```python
"""
라빈-카프 글줄 찾기 알고리즘.

다항 구르는 흩는 값으로 글월에서 본이 나오는 곳을 모두
기대 $O(n + m)$ 시간에 찾는다.
"""

# === 라빈-카프 찾기 ===

def rabin_karp(text: str, pattern: str, d: int = 256, q: int = 101) -> list[int]:
    """라빈-카프로 본문에서 무늬가 나타나는 곳을 모두 찾는다.

    인수:
        text: 찾을 글월.
        pattern: 찾을 본.
        d: 흩는 함수의 밑(글자 모임의 크기).
        q: 흩는 함수의 소수 나눔수.

    반환값:
        본이 나오는 시작 번호의 목록.
    """
    n, m = len(text), len(pattern)
    if m > n or m == 0:
        return []

    matches = []
    h = pow(d, m - 1, q)  # d^(m-1) mod q

    # 처음 해시값 계산
    p_hash = 0  # 무늬의 해시
    t_hash = 0  # 본문 창의 해시
    for i in range(m):
        p_hash = (d * p_hash + ord(pattern[i])) % q
        t_hash = (d * t_hash + ord(text[i])) % q

    # 창을 미끄러뜨린다
    for i in range(n - m + 1):
        if p_hash == t_hash:
            # 글자마다 확인한다(헛맞음을 피한다)
            if text[i:i + m] == pattern:
                matches.append(i)

        # 다음 창의 흩는 값을 셈한다
        if i < n - m:
            t_hash = (d * (t_hash - ord(text[i]) * h) + ord(text[i + m])) % q
            if t_hash < 0:
                t_hash += q

    return matches

# === 여러 본 변형 ===

def rabin_karp_multi(
    text: str, patterns: list[str], d: int = 256, q: int = 101
) -> dict[str, list[int]]:
    """길이가 같은 본 여럿을 찾는다."""
    if not patterns:
        return {}

    m = len(patterns[0])
    results = {p: [] for p in patterns}

    # 본의 흩는 값을 셈한다
    p_hashes = {}
    for p in patterns:
        h_val = 0
        for ch in p:
            h_val = (d * h_val + ord(ch)) % q
        p_hashes.setdefault(h_val, []).append(p)

    n = len(text)
    if m > n:
        return results

    h = pow(d, m - 1, q)
    t_hash = 0
    for i in range(m):
        t_hash = (d * t_hash + ord(text[i])) % q

    for i in range(n - m + 1):
        if t_hash in p_hashes:
            for p in p_hashes[t_hash]:
                if text[i:i + m] == p:
                    results[p].append(i)

        if i < n - m:
            t_hash = (d * (t_hash - ord(text[i]) * h) + ord(text[i + m])) % q
            if t_hash < 0:
                t_hash += q

    return results

# === 메인 ===

if __name__ == "__main__":
    text = "AABAACAADAABAABA"
    pattern = "AABA"

    matches = rabin_karp(text, pattern)
    print(f"Text:    {text}")
    print(f"Pattern: {pattern}")
    print(f"Matches at: {matches}")

    # 여러 본 보기
    patterns = ["AABA", "AACA"]
    multi = rabin_karp_multi(text, patterns)
    print(f"\nMulti-pattern search:")
    for p, idx in multi.items():
        print(f"  '{p}': {idx}")
    # 내임:
    # 글월:    AABAACAADAABAABA
    # 본: AABA
    # 맞은 자리: [0, 9, 12]
    #
    # 여러 본 찾기:
    #   'AABA': [0, 9, 12]
    #   'AACA': [3]
```

**출력:**

```
Text:    AABAACAADAABAABA
Pattern: AABA
Matches at: [0, 9, 12]

Multi-pattern search:
  'AABA': [0, 9, 12]
  'AACA': [3]
```

---

## 5. 풀이 예제

**글월:** `ABCABC`, **본:** `ABC`, $d = 256$, $q = 101$.

1. 본의 흩는 값: $(65 \cdot 256^2 + 66 \cdot 256 + 67) \bmod 101 = 4259907 \bmod 101 = 79$.
2. 창 "ABC"의 흩는 값: 79. **맞음!** 확인: "ABC" = "ABC". 자리 0을 알린다.
3. 구른다: 'A'을 빼고 'A'을 더한다. 창 "BCA"의 흩는 값: $(256 \cdot (79 - 65 \cdot 256^2 \bmod 101) + 65) \bmod 101$. 셈해 보면 맞지 않는다.
4. 구른다: 창 "CAB" — 맞지 않는다.
5. 구른다: 창 "ABC"의 흩는 값 = 79. **맞음!** 확인: "ABC" = "ABC". 자리 3을 알린다.

---

## 연습문제

**연습문제 1.**
라빈-카프의 구르는 흩는 값 재주를 설명하고 그것이 왜 $O(1)$ 흩는 값 새로 고침을 되게 하는지 밝혀라.

??? success "연습문제 1 풀이"
    라빈-카프는 다항 흩기 $H(s[i..i+m-1]) = \sum_{j=0}^{m-1} s[i+j] \cdot d^{m-1-j} \mod q$을 쓴다. 창을 한 자리 미끄러뜨리면 새 흩는 값을 옛 값에서 셈한다. 곧 $H_{\text{new}} = (H_{\text{old}} - s[i] \cdot d^{m-1}) \cdot d + s[i+m] \mod q$이다. 가장 왼쪽 글자를 빼고 가장 오른쪽 글자를 더하는 데 ($d^{m-1} \mod q$을 미리 셈해 두면) $O(1)$이 든다. 구르는 흩는 값이 없으면 흩는 값 셈마다 $O(m)$이 든다. $\square$

---

**연습문제 2.**
라빈-카프의 헛맞음이란 무엇인가? 알고리즘의 복잡도에 어떤 영향을 주는가?

??? success "연습문제 2 풀이"
    **헛맞음**은 글월 창의 흩는 값이 본의 흩는 값과 같은데 실제 글줄은 다를 때(흩는 값 부딪침) 생긴다. 흩는 값이 맞으면 라빈-카프가 글자를 견주어 확인하며 맞음마다 $O(m)$이 든다. 흩는 함수가 잘 흩뜨리면 기대 시간은 $O(n + m)$이다. 창마다 부딪치면(예컨대 글자가 모두 같으면) 최악의 경우 $O(nm)$이다. 큰 소수 $q$을 고르면 창마다의 부딪침 확률이 $O(m/q)$으로 준다. $\square$

---

**연습문제 3.**
라빈-카프는 본 여럿을 한꺼번에 찾을 수 있다. 어떻게 하는지 적어라.

??? success "연습문제 3 풀이"
    본 $k$개의 흩는 값을 모두 셈해 흩는 모임에 담는다. 창을 글월 위로 미끄러뜨리며 구르는 흩는 값을 셈한다. 창마다 그 흩는 값이 모임에 있는지 살핀다(기대 $O(1)$). 맞으면 그 흩는 값을 가진 본 모두와 견주어 확인한다. 길이 $n$인 글월에서 길이 $m$인 본 $k$개를 찾는 데 기대 $O(n + km)$이 든다. KMP를 $k$번 돌리는 것($O(kn)$)보다 빠르다. 본이 여럿이면 아호-코라식이 더 낫다. 곧 $z$이 맞은 수일 때 $O(n + m_{\text{total}} + z)$이다. $\square$

---

**연습문제 4.**
라빈-카프는 기본 꼴에서 왜 몬테카를로 알고리즘인가? 어떻게 라스베이거스로 만들 수 있는가?

??? success "연습문제 4 풀이"
    기본 라빈-카프는 글자마다 확인하지 않고 흩는 값이 맞으면 맞음을 알린다. 이는 몬테카를로이다. 곧 늘 $O(n)$ 시간에 돌지만 헛맞음(틀린 맞음)을 낼 수 있다. 라스베이거스(늘 옳고 기대 다항 시간)로 만들려면 흩는 값이 맞을 때마다 글자 $m$개를 모두 견주어 확인한다. 좋은 흩는 함수라면 기대 시간은 그대로 $O(n + m)$이지만 최악의 경우는 $O(nm)$이 된다. $\square$

## 정리하며

이 마당은 구르는 흩는 값、알고리즘、복잡도、파이썬 구현을 차례로 짚었다.

**참고 문헌**

- Karp, R. M., & Rabin, M. O. (1987). Efficient randomized pattern-matching algorithms. *IBM Journal of Research and Development*, 31(2), 249-260.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 32. MIT Press.
