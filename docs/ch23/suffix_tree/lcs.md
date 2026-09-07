# 가장 긴 공통 부분문자열

주어진 글줄 둘 이상에 잇닿은 부분 글줄로 나오는 가장 긴 글줄을 찾는 것은 생물정보학(DNA 차례 견주기), 표절 알아내기, 자료 겹침 없애기에 쓰이는 바탕 문제이다. 부분 글줄 짝을 모두 살피는 막무가내 방식은 길이 $n$과 $m$인 글줄 둘에 $O(n^2 m)$이 든다. 뒷가지 나무나 뒷가지 배열을 쓰면 $O(n + m)$ 시간에 풀 수 있다. 이 절은 두 방식을 모두 보인다.

## 문제 서술

길이 $n$인 글줄 $S_1$과 길이 $m$인 글줄 $S_2$이 주어질 때 $w$이 $S_1$과 $S_2$ 모두의 잇닿은 부분 글줄인 가장 긴 글줄 $w$을 찾아라.

형식적으로 다음과 같다.

$$
\text{LCS}(S_1, S_2) = \arg\max_{w} |w| \quad \text{단, } w \text{ 는 } S_1 \text{ 과 } S_2 \text{ 모두의 부분 글줄}
$$

최장 공통 부분 글줄의 **길이**를 흔히 $\text{lcstr}(S_1, S_2)$으로 적는다.

!!! warning "최장 공통 부분열과 최장 공통 부분문자열"
    (잇닿은) **최장 공통 부분 글줄**과 (꼭 잇닿지 않아도 되는) **최장 공통 부분 차례**를 헷갈리지 마라. 부분 차례 문제는 동적 짜기로 $O(nm)$ 시간에 풀고, 부분 글줄 문제는 뒷가지 짜임으로 $O(n + m)$에 푼다.

## 꼬리말 나무로 푸는 길

### 넓힌 꼬리말 나무

서로 다른 파수로 두 글줄을 이어 붙여 **넓힌 뒷가지 나무**를 세운다:

$$
T = S_1 \cdot \texttt{\#} \cdot S_2 \cdot \texttt{\$}
$$

여기서 $\texttt{\#}$과 $\texttt{\$}$은 어느 글줄에도 없는 서로 다른 글자이다. $T$의 넓힌 뒷가지 나무는 $S_1$과 $S_2$의 뒷가지를 모두 담는다.

### 안쪽 마디에 표시하기

나무를 세운 뒤 잎마다 그 뒷가지가 어느 글줄에 드는지로 이름표를 붙인다:

- 뒷가지가 자리 $i < n$($S_1$ 안)에서 시작하면 그 잎은 **$S_1$ 잎**이다
- 뒷가지가 자리 $i > n$($S_2$ 안)에서 시작하면 그 잎은 **$S_2$ 잎**이다

아래 나무에 $S_1$ 잎과 $S_2$ 잎이 적어도 하나씩 있으면 안쪽 마디 $v$은 **함께 쓰는** 마디이다. 이 표시는 아래에서 위로 돌아보며 $O(n + m)$ 시간에 셈할 수 있다.

### 최장 공통 부분 글줄 찾기

최장 공통 부분 글줄은 **가장 깊은 함께 쓰는 안쪽 마디**, 곧 길 이름표가 가장 긴 함께 쓰는 마디에 맞닿는다:

$$
\text{LCS} = \text{path}(v^*) \quad \text{where } v^* = \arg\max_{\substack{v \text{ shared} \\ v \text{ internal}}} \text{depth}(v)
$$

**시간 복잡도**: (우코넨 알고리즘으로) 넓힌 뒷가지 나무를 세우는 데 $O(n + m)$, 아래에서 위로 표시하고 최대 깊이를 찾는 데 $O(n + m)$이 든다. 모두 합하면:

$$
T(n, m) = O(n + m)
$$

## 꼬리말 배열로 푸는 길

### 이어 붙이기와 세우기

글줄을 $T = S_1 \cdot \texttt{\#} \cdot S_2 \cdot \texttt{\$}$으로 이어 붙이고 $T$의 뒷가지 배열과 최장 공통 앞가지 배열을 세운다.

### 최장 공통 부분 글줄 훑어 찾기

최장 공통 부분 글줄의 길이는 뒷가지 배열에서 **서로 다른 글줄**에서 온 이웃한 뒷가지 사이의 최장 공통 앞가지 최댓값과 같다:

$$
\text{lcstr}(S_1, S_2) = \max_{\substack{1 \leq k \leq |T| \\ \text{SA}[k] \text{ 와 } \text{SA}[k-1] \text{ 가} \\ \text{서로 다른 글줄에서 옴}}} \text{LCP}[k]
$$

자리 $i$에서 시작하는 뒷가지는 $i \leq n - 1$이면 $S_1$에, $i \geq n + 1$이면 $S_2$에 든다(자리 $n$은 가르개 $\texttt{\#}$이다).

??? example "'abcde'과 'bcdef'의 가장 긴 공통 부분문자열"
    Concatenate: $T = \texttt{abcde\#bcdef\$}$

    뒷가지 배열과 최장 공통 앞가지 배열을 세운 뒤 서로 다른 글줄의 뒷가지 사이 최장 공통 앞가지 최댓값을 훑어 찾는다.

    뒷가지 `bcde#bcdef$`($S_1$, 자리 1)과 `bcdef$`($S_2$, 자리 6)이 앞가지 `bcde`을 나눠 가져 $\text{LCP} = 4$이다.

    따라서 $\text{LCS} = \texttt{bcde}$이고 길이는 4이다.

## 여러 글자열로 넓히기

최장 공통 부분 글줄 문제는 글줄 $k$개 $S_1, S_2, \ldots, S_k$으로 자연스레 넓어진다. 서로 다른 가르개로 글줄을 모두 이어 붙인다:

$$
T = S_1 \cdot \texttt{c}_1 \cdot S_2 \cdot \texttt{c}_2 \cdots S_k \cdot \texttt{c}_k
$$

$T$의 뒷가지 나무나 뒷가지 배열을 세운다. 뒷가지 나무에서는 아래 나무에 $k$개 글줄의 잎이 모두 든 가장 깊은 안쪽 마디를 찾는다. 뒷가지 배열에서는 최장 공통 앞가지 배열에 미끄러지는 창을 써서 $k$개 글줄의 뒷가지가 모두 든 범위의 최장 공통 앞가지 최댓값을 찾는다.

**시간 복잡도**: 뒷가지 나무 방식은 $O(n_1 + n_2 + \cdots + n_k)$.

## 구현

```python
"""
뒷가지 배열과 최장 공통 앞가지 배열로 찾는 최장 공통 부분 글줄.
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


# === 최장 공통 부분 글줄 ===

def longest_common_substring(s1: str, s2: str) -> str:
    """s1과 s2의 최장 공통 부분 글줄을 찾는다.

    매개변수
    ----------
    s1 : str
        첫 들임 글줄.
    s2 : str
        둘째 들임 글줄.

    반환값
    -------
    str
        최장 공통 부분 글줄.
    """
    separator = "#"
    sentinel = "$"
    text = s1 + separator + s2 + sentinel
    n1 = len(s1)

    sa = build_suffix_array(text)
    lcp = build_lcp(text, sa)

    best_len = 0
    best_pos = 0

    for k in range(1, len(text)):
        pos_prev = sa[k - 1]
        pos_curr = sa[k]

        # 뒷가지가 서로 다른 글줄에서 왔는지 살핀다
        from_s1_prev = pos_prev < n1
        from_s1_curr = pos_curr < n1

        if from_s1_prev != from_s1_curr and lcp[k] > best_len:
            best_len = lcp[k]
            best_pos = sa[k]

    return text[best_pos:best_pos + best_len]


# === 메인 ===

if __name__ == "__main__":
    s1 = "abcdefg"
    s2 = "cdefxyz"
    result = longest_common_substring(s1, s2)
    print(f"S1: '{s1}'")
    print(f"S2: '{s2}'")
    print(f"LCS: '{result}' (length {len(result)})")

    s1 = "banana"
    s2 = "ananas"
    result = longest_common_substring(s1, s2)
    print(f"\nS1: '{s1}'")
    print(f"S2: '{s2}'")
    print(f"LCS: '{result}' (length {len(result)})")
```

## 복잡도 비교

| 방법 | 시간 | 공간 |
|--------|------|-------|
| 막무가내(모든 부분 글줄 짝) | $O(n^2 m)$ | $O(1)$ |
| 동적 짜기 | $O(nm)$ | $O(nm)$ 또는 $O(\min(n,m))$ |
| 뒷가지 나무 | $O(n + m)$ | $O(n + m)$ |
| 뒷가지 배열 + 최장 공통 앞가지 | $O(n + m)$ | $O(n + m)$ |

뒷가지 바탕 방식은 가장 좋은 선형 시간을 이루며 큰 들임에 낫다.

## 참고 문헌

- Gusfield, D. (1997). *Algorithms on Strings, Trees, and Sequences*. Cambridge University Press, Chapter 7.
- Hui, L. C. K. (1992). *Color set size problem with applications to string matching*. CPM 1992, LNCS 644, pp. 230-243.

## 연습문제

**연습문제 1.**
최장 공통 부분 글줄의 핵심 자료 짜임이나 개념과 그 으뜸 쓰임새를 설명하라.

??? success "연습문제 1 풀이"
    최장 공통 부분 글줄은 글줄이나 차례 자료를 미리 다듬고 묻는 효율 좋은 길을 준다. 으뜸 쓰임새는 부분 글줄, 본, 들임의 짜임 성질에 대한 되풀이되는 물음에 답하는 것이다. 미리 다듬기가 다룰 만한 시간에 자료 짜임을 세우고 나면 맨바닥에서 다시 다듬는 것보다 훨씬 빠르게 물음에 답할 수 있다. $\square$

---

**연습문제 2.**
최장 공통 부분 글줄을 세우는 시간 복잡도는 무엇인가? 으뜸 연산의 묻기 시간은 무엇인가?

??? success "연습문제 2 풀이"
    세우는 시간은 쓰는 알고리즘에 달렸다. 흔한 한계는 $n$이 들임 크기일 때 $O(n)$에서 $O(n \log n)$ 사이이다. 묻기는 흔히 본 찾기에 $O(m)$($m$은 물음 길이), 미리 셈한 성질에 $O(1)$이 든다. 공간 복잡도는 흔히 $O(n)$이거나 $\sigma$이 글자 모임의 크기일 때 $O(n\sigma)$이다. $\square$

---

**연습문제 3.**
최장 공통 부분 글줄을 더 단순한 다른 방식과 견주어라. 더 정교한 짜임은 언제 값어치가 있는가?

??? success "연습문제 3 풀이"
    더 단순한 방식(예컨대 막무가내 훑기나 정렬)은 묻기 시간이 더 길지만 세우는 군더더기가 적다. 정교한 짜임은 다음일 때 값어치가 있다. (1) 같은 자료에 물음을 많이 던져 세우는 값이 고르게 나뉠 때, (2) 묻기 시간이 결정적일 때(실시간 쓰임새), (3) 자료가 커서 점근 나아짐이 실전에서 중요할 때이다. 작은 자료에 물음을 한 번 던지는 경우에는 상수 인수가 작은 단순한 방식이 더 빠를 수 있다. $\square$

---

**연습문제 4.**
들임 글줄 "banana"에 대해 최장 공통 부분 글줄을 세우는 것을 좇아라. 중간 걸음을 보여라.

??? success "연습문제 4 풀이"
    "banana"($n = 6$)에 대해: 글줄을 글자마다(또는 뒷가지마다) 처리하며 자료 짜임을 조금씩 세운다. 마지막 짜임은 뒷가지 "banana", "anana", "nana", "ana", "na", "a"을 모두 담는다. 결과의 핵심 성질을 확인할 수 있다. 곧 공통 앞가지를 나눠 쓰고, 뒷가지 차례가 지켜지며, 부분 글줄에 대한 모든 물음을 그 짜임에서 답할 수 있다. $\square$
