# 뒷가지 배열의 쓰임새

뒷가지 배열은 특히 최장 공통 앞가지 배열을 곁들이면 갖가지 글줄 처리 문제의 두루 쓰는 바탕이 된다. 예전에는 뒷가지 나무가 필요하던 여러 일, 곧 가장 긴 되풀이 부분 글줄 찾기, 두 글줄의 최장 공통 부분 글줄 찾기, 서로 다른 부분 글줄 세기를 뒷가지 배열로도 똑같이 효율 좋게, 기억 공간은 훨씬 적게 풀 수 있다. 이 절은 가장 중요한 쓰임새를 훑고 저마다의 알고리즘을 준다.

---

## 1. 본 찾기

뒷가지 배열의 가장 바탕이 되는 쓰임새는 글월 $T[0..n-1]$에서 본 $P[0..m-1]$이 나오는 곳을 모두 찾는 것이다.

### 이분 찾기 방식

뒷가지 배열이 뒷가지를 정렬해 담으므로 $P$으로 시작하는 뒷가지는 뒷가지 배열에서 잇닿은 범위 $[\ell, r]$을 이룬다. 이분 찾기를 두 번 해 이 범위를 찾는다:

1. **아래 경계**: suffix($\text{SA}[\ell]$)이 $P$을 앞가지로 갖는 가장 작은 $\ell$을 찾는다
2. **위 경계**: suffix($\text{SA}[r]$)이 $P$을 앞가지로 갖는 가장 큰 $r$을 찾는다

나온 횟수는 $r - \ell + 1$이고 나온 자리는 $k \in [\ell, r]$마다 $\text{SA}[k]$이다.

**시간 복잡도**: 최장 공통 앞가지 배열이 없으면 $O(m \log n)$, 그것으로 이분 찾기를 도우면 $O(m + \log n)$.

??? example "'banana'에서 'an'이 나오는 곳 모두 찾기"
    $T = \texttt{banana\$}$이고 $\text{SA} = [6, 5, 3, 1, 0, 4, 2]$일 때:

    - 아래 경계 이분 찾기가 $\ell = 2$을 찾는다($\text{SA}[2] = 3$의 뒷가지 `ana$`)
    - 위 경계 이분 찾기가 $r = 3$을 찾는다($\text{SA}[3] = 1$의 뒷가지 `anana$`)
    - 본 `an`이 자리 3과 1에 나온다

---

## 2. 가장 긴 되풀이 부분 글줄

**가장 긴 되풀이 부분 글줄(LRS)**은 $T$에 적어도 두 번 나오는 가장 긴 글줄이다. 최장 공통 앞가지 배열이 있으면 그저 최댓값이다:

$$
\text{LRS length} = \max_{1 \leq k \leq n} \text{LCP}[k]
$$

실제 부분 글줄은 $k^* = \arg\max_k \text{LCP}[k]$일 때 $T[\text{SA}[k^*] .. \text{SA}[k^*] + \text{LCP}[k^*] - 1]$이다.

**시간 복잡도**: 뒷가지 배열과 최장 공통 앞가지 배열을 세운 뒤 $O(n)$.

??? example "'banana'의 가장 긴 되풀이 부분 글줄"
    $T = \texttt{banana\$}$이고 $\text{LCP} = [0, 0, 1, 3, 0, 0, 2]$일 때:

    - 최장 공통 앞가지 값의 최댓값은 3이고 자리 $k = 3$에 나온다
    - $\text{SA}[3] = 1$이므로 가장 긴 되풀이 부분 글줄은 $T[1..3] = \texttt{ana}$이다
    - 실제로 `ana`이 자리 1과 3에 나온다

---

## 3. 두 글줄의 최장 공통 부분 글줄

글줄 $S_1$과 $S_2$이 주어질 때 **최장 공통 부분 글줄(LCS)**은 둘을 가르개로 이어 붙여 찾을 수 있다:

$$
T = S_1 \cdot \texttt{\#} \cdot S_2 \cdot \texttt{\$}
$$

여기서 $\texttt{\#}$과 $\texttt{\$}$은 어느 글줄에도 없는 서로 다른 파수 글자이다. $T$의 뒷가지 배열과 최장 공통 앞가지 배열을 세운 뒤 서로 다른 글줄에서 온 이웃한 뒷가지 사이의 최장 공통 앞가지 최댓값을 훑는다.

엄밀히 $n_1 = |S_1|$이라 하자. 자리 $i$에서 시작하는 뒷가지는 $i < n_1$이면 $S_1$에, $i > n_1$이면 $S_2$에 든다. 최장 공통 부분 글줄의 길이는 다음과 같다:

$$
\text{LCS 길이} = \max_{\substack{1 \leq k \leq |T| \\ \text{SA}[k] \text{ 와 } \text{SA}[k-1] \text{ 가} \\ \text{서로 다른 글줄에서 옴}}} \text{LCP}[k]
$$

**시간 복잡도**: 모두 $O(n_1 + n_2)$.

---

## 4. 서로 다른 부분 글줄 세기

$T$의 모든 부분 글줄은 어떤 뒷가지의 앞가지이다. 뒷가지 $\text{SA}[k]$은 부분 글줄 $(n - \text{SA}[k])$개를 보탠다(길이 1부터 $n - \text{SA}[k]$까지의 앞가지). 다만 그 가운데 $\text{LCP}[k]$개는 정렬 차례에서 앞선 뒷가지와 나눠 갖는다. 서로 다른 부분 글줄의 총수는 다음과 같다:

$$
D = \sum_{k=0}^{n} (n - \text{SA}[k]) - \sum_{k=1}^{n} \text{LCP}[k]
$$

**시간 복잡도**: 세운 뒤 $O(n)$.

---

## 5. 최장 공통 앞가지 묻기

(뒷가지 배열에서 이웃한 것뿐 아니라) 아무 두 뒷가지의 최장 공통 앞가지도 **구간 최소 묻기(RMQ)** 성질로 셈할 수 있다:

$$
\text{lcp}(\text{suffix}(\text{SA}[i]),\; \text{suffix}(\text{SA}[j])) = \min_{i < k \leq j} \text{LCP}[k]
$$

최장 공통 앞가지 배열 위에 성긴 표를 $O(n \log n)$ 미리 다듬기 시간에 세우면 물음마다 $O(1)$ 시간에 답할 수 있다.

!!! tip "선형 미리 다듬기로 하는 구간 최소 묻기"
    (줄이기를 거치면 최장 공통 앞가지 배열이 채우는) $\pm 1$ 구간 최소 묻기의 특별한 경우에 벤더-파라크-콜튼 알고리즘을 쓰면 미리 다듬기에 $O(n)$ 시간만 들면서 묻기 시간은 $O(1)$으로 지킨다.

---

## 6. 부분 글줄의 사전 차례 견줌

부분 글줄 $T[i..i+\ell_1-1]$과 $T[j..j+\ell_2-1]$이 주어질 때 다음으로 사전 차례를 $O(1)$ 시간에 정할 수 있다:

1. 구간 최소 묻기로 $L = \text{lcp}(\text{suffix}(i), \text{suffix}(j))$을 $O(1)$에 셈한다
2. $L \geq \min(\ell_1, \ell_2)$이면 짧은 부분 글줄이 사전 차례로 앞선다($\ell_1 = \ell_2$이면 같다)
3. 아니면 $T[i + L]$과 $T[j + L]$을 견준다

그러면 $O(n)$ 미리 다듬기 뒤에 아무 부분 글줄이나 $O(1)$에 견줄 수 있다.

---

## 7. 본이 나온 횟수 세기

나온 곳을 찾는 것을 넘어, 이분 찾기로 범위 $[\ell, r]$을 찾으면 본 $P$이 $T$에 몇 번 나오는지 **셀** 수 있다. 자리를 모두 세지 않아도 그 수는 $r - \ell + 1$이다.

---

## 8. 쓰임새 간추리기

| 문제 | 자료 짜임 | 시간 |
|---------|---------------|------|
| 본 찾기 | 뒷가지 배열 | $O(m \log n)$ |
| 본 찾기 | 뒷가지 배열 + 최장 공통 앞가지 | $O(m + \log n)$ |
| 가장 긴 되풀이 부분 글줄 | 뒷가지 배열 + 최장 공통 앞가지 | $O(n)$ |
| 최장 공통 부분 글줄 | 뒷가지 배열 + 최장 공통 앞가지 | $O(n_1 + n_2)$ |
| 서로 다른 부분 글줄 세기 | 뒷가지 배열 + 최장 공통 앞가지 | $O(n)$ |
| 아무 두 뒷가지의 최장 공통 앞가지 | 뒷가지 배열 + 최장 공통 앞가지 + 구간 최소 묻기 | 묻기 $O(1)$ |
| 부분 글줄 사전 차례 견줌 | 뒷가지 배열 + 최장 공통 앞가지 + 구간 최소 묻기 | 묻기 $O(1)$ |

---

## 9. 구현

```python
"""
뒷가지 배열의 쓰임새: 본 찾기, 가장 긴 되풀이 부분 글줄,
서로 다른 부분 글줄 세기.
"""

# === 뒷가지 배열 세우기 ===

def build_suffix_array(text: str) -> list[int]:
    """앞가지 곱절 늘리기로 뒷가지 배열을 세운다."""
    n = len(text)
    rank = [ord(c) for c in text]
    sa = list(range(n))
    k = 1
    while k < n:
        def key(i, _k=k, _r=rank[:]):
            return (_r[i], _r[i + _k] if i + _k < n else -1)
        sa.sort(key=key)
        new_rank = [0] * n
        for j in range(1, n):
            prev = (rank[sa[j - 1]],
                    rank[sa[j - 1] + k] if sa[j - 1] + k < n else -1)
            curr = (rank[sa[j]],
                    rank[sa[j] + k] if sa[j] + k < n else -1)
            new_rank[sa[j]] = new_rank[sa[j - 1]] + (1 if curr != prev else 0)
        rank = new_rank
        if rank[sa[-1]] == n - 1:
            break
        k *= 2
    return sa

# === 가사이 알고리즘 ===

def build_lcp(text: str, sa: list[int]) -> list[int]:
    """가사이 알고리즘으로 최장 공통 앞가지 배열을 셈한다."""
    n = len(sa)
    rank = [0] * n
    for k in range(n):
        rank[sa[k]] = k
    lcp = [0] * n
    h = 0
    for i in range(n):
        r = rank[i]
        if r > 0:
            j = sa[r - 1]
            while i + h < n and j + h < n and text[i + h] == text[j + h]:
                h += 1
            lcp[r] = h
            h = max(h - 1, 0)
        else:
            h = 0
    return lcp

# === 쓰임새 ===

def pattern_search(text: str, sa: list[int], pattern: str) -> list[int]:
    """뒷가지 배열에 이분 찾기를 써서 글월에서 본이 나오는 곳을 모두 찾는다."""
    n = len(text)
    m = len(pattern)

    # 아래 경계
    lo, hi = 0, n - 1
    while lo < hi:
        mid = (lo + hi) // 2
        suffix = text[sa[mid]:sa[mid] + m]
        if suffix < pattern:
            lo = mid + 1
        else:
            hi = mid
    left = lo

    # 위 경계
    lo, hi = left, n - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        suffix = text[sa[mid]:sa[mid] + m]
        if suffix > pattern:
            hi = mid - 1
        else:
            lo = mid
    right = lo

    if text[sa[left]:sa[left] + m] != pattern:
        return []
    return sorted(sa[k] for k in range(left, right + 1))

def longest_repeated_substring(text: str, sa: list[int],
                                lcp: list[int]) -> str:
    """가장 긴 되풀이 부분 글줄을 찾는다."""
    max_lcp = max(lcp)
    if max_lcp == 0:
        return ""
    k = lcp.index(max_lcp)
    return text[sa[k]:sa[k] + max_lcp]

def count_distinct_substrings(text: str, sa: list[int],
                               lcp: list[int]) -> int:
    """서로 다른 부분 글줄의 수를 센다."""
    n = len(text)
    total = sum(n - sa[k] for k in range(n))
    duplicates = sum(lcp[k] for k in range(1, n))
    return total - duplicates

# === 메인 ===

if __name__ == "__main__":
    text = "banana$"
    sa = build_suffix_array(text)
    lcp = build_lcp(text, sa)

    print(f"Text: {text}")
    print(f"SA:  {sa}")
    print(f"LCP: {lcp}")

    positions = pattern_search(text, sa, "ana")
    print(f"\n'ana' found at positions: {positions}")

    lrs = longest_repeated_substring(text, sa, lcp)
    print(f"Longest repeated substring: '{lrs}'")

    count = count_distinct_substrings(text, sa, lcp)
    print(f"Distinct substrings: {count}")
```

---

## 연습문제

**연습문제 1.**
뒷가지 배열의 쓰임새의 핵심 자료 짜임이나 개념과 그 으뜸 쓰임새를 설명하라.

??? success "연습문제 1 풀이"
    뒷가지 배열의 쓰임새은 글줄이나 차례 자료를 미리 다듬고 묻는 효율 좋은 길을 준다. 으뜸 쓰임새는 부분 글줄, 본, 들임의 짜임 성질에 대한 되풀이되는 물음에 답하는 것이다. 미리 다듬기가 다룰 만한 시간에 자료 짜임을 세우고 나면 맨바닥에서 다시 다듬는 것보다 훨씬 빠르게 물음에 답할 수 있다. $\square$

---

**연습문제 2.**
뒷가지 배열의 쓰임새을 세우는 시간 복잡도는 무엇인가? 으뜸 연산의 묻기 시간은 무엇인가?

??? success "연습문제 2 풀이"
    세우는 시간은 쓰는 알고리즘에 달렸다. 흔한 한계는 $n$이 들임 크기일 때 $O(n)$에서 $O(n \log n)$ 사이이다. 묻기는 흔히 본 찾기에 $O(m)$($m$은 물음 길이), 미리 셈한 성질에 $O(1)$이 든다. 공간 복잡도는 흔히 $O(n)$이거나 $\sigma$이 글자 모임의 크기일 때 $O(n\sigma)$이다. $\square$

---

**연습문제 3.**
뒷가지 배열의 쓰임새을 더 단순한 다른 방식과 견주어라. 더 정교한 짜임은 언제 값어치가 있는가?

??? success "연습문제 3 풀이"
    더 단순한 방식(예컨대 막무가내 훑기나 정렬)은 묻기 시간이 더 길지만 세우는 군더더기가 적다. 정교한 짜임은 다음일 때 값어치가 있다. (1) 같은 자료에 물음을 많이 던져 세우는 값이 고르게 나뉠 때, (2) 묻기 시간이 결정적일 때(실시간 쓰임새), (3) 자료가 커서 점근 나아짐이 실전에서 중요할 때이다. 작은 자료에 물음을 한 번 던지는 경우에는 상수 인수가 작은 단순한 방식이 더 빠를 수 있다. $\square$

---

**연습문제 4.**
들임 글줄 "banana"에 대해 뒷가지 배열의 쓰임새을 세우는 것을 좇아라. 중간 걸음을 보여라.

??? success "연습문제 4 풀이"
    "banana"($n = 6$)에 대해: 글줄을 글자마다(또는 뒷가지마다) 처리하며 자료 짜임을 조금씩 세운다. 마지막 짜임은 뒷가지 "banana", "anana", "nana", "ana", "na", "a"을 모두 담는다. 결과의 핵심 성질을 확인할 수 있다. 곧 공통 앞가지를 나눠 쓰고, 뒷가지 차례가 지켜지며, 부분 글줄에 대한 모든 물음을 그 짜임에서 답할 수 있다. $\square$

## 정리하며

이 마당은 본 찾기、가장 긴 되풀이 부분 글줄、두 글줄의 최장 공통 부분 글줄、서로 다른 부분 글줄 세기을 차례로 짚었다.

**참고 문헌**

- Manber, U. and Myers, G. (1993). *Suffix arrays: A new method for on-line string searches*. SIAM Journal on Computing, 22(5), 935-948.
- Abouelhoda, M. I., Kurtz, S., and Ohlebusch, E. (2004). *Replacing suffix trees with enhanced suffix arrays*. Journal of Discrete Algorithms, 2(1), 53-86.
