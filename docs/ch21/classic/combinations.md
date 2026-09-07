# 조합 만들어 내기

자리 바꿈은 원소의 **차례**를 따지지만, 특징 고르기, 위원회 짜기, 시험 사례 만들기 같은 많은 문제는 **어느** 원소를 골랐느냐만 따진다. 원소 $n$개의 $k$-조합은 차례가 없는 크기 $k$짜리 부분 모임이다. 되짚기는 원소마다 넣거나 빼는 결정을 하고 딱 $k$개를 고른다는 단순한 제약을 두어 $\binom{n}{k}$개 조합을 모두 만든다.

## 문제 서술

**들임.** 서로 다른 원소 $n$개의 모임 $\{a_1, a_2, \ldots, a_n\}$과
정수 $0 \leq k \leq n$.

**내놓기.** 크기 $k$인 부분 모임 $\binom{n}{k} = \frac{n!}{k!(n-k)!}$개 모두.

## 되짚기로 세우기

### 방식 1 --- 넣거나 빼는 결정

원소를 $a_1, a_2, \ldots, a_n$ 차례로 처리한다. 원소 $a_i$에서 지금 부분 모임에 넣을지 건너뛸지 정한다.

- **결정 $i$**($i = 1, \ldots, n$): $a_i$을 넣거나 뺀다.
- **갈래 수**: 층마다 2.
- **온전한 나무**: 잎이 $2^n$개(모든 부분 모임)이고 그 가운데 $\binom{n}{k}$개가 딱 $k$개를 넣은 것이다.

**가지치기.** 두 조건에서 일찍 멈출 수 있다:

1. **너무 많이 고름**: 넣은 원소 수가 이미 $k$이면 남은 원소를 모두 뺀다(더 가지를 뻗을 필요가 없다).
2. **남은 것이 너무 적음**: 넣은 원소 수에 남은 원소 수를 더한 것이 $k$보다 작으면 쳐 낸다. $k$에 이를 수 없기 때문이다.

엄밀히, $a_1, \ldots, a_i$을 처리한 뒤 넣은 원소의 수를 $c$이라 하자. $c > k$이거나 $c + (n - i) < k$이면 쳐 낸다.

### 방식 2 --- 앞으로 고르기

조합의 $j$번째 원소를 $(j-1)$번째로 고른 원소 뒤에 오는 것 가운데서 고른다. 그러면 번호가 늘어나는 차례가 되어 겹치는 부분 모임이 저절로 없어진다.

```
COMBINATIONS(start, combo, k, n):
    if len(combo) == k:
        output combo
        return

    for i = start to n - (k - len(combo)) + 1:
        combo.append(a[i])
        COMBINATIONS(i + 1, combo, k, n)
        combo.pop()
```

되풀이의 위 경계 `n - (k - len(combo)) + 1`은 조합을 채울 만큼 원소가 남지 않은 갈래를 쳐 낸다.

## 파이썬 구현

```python
"""
되짚기로 원소 n개의 k-조합을 모두 만든다.

가지치기를 곁들인 앞으로 고르기를 보인다.
"""


# === 앞으로 고르기 방식 ===============================================

def combinations(elements, k):
    """*elements*의 원소 k개짜리 부분 모임을 모두 돌려준다."""
    n = len(elements)
    results = []
    combo = []

    def backtrack(start):
        if len(combo) == k:
            results.append(combo[:])
            return

        remaining_needed = k - len(combo)
        for i in range(start, n - remaining_needed + 1):
            combo.append(elements[i])
            backtrack(i + 1)
            combo.pop()

    backtrack(0)
    return results


# === 넣거나 빼는 방식 =================================================

def combinations_binary(elements, k):
    """넣거나 빼는 결정으로 원소 k개짜리 부분 모임을 모두 돌려준다."""
    n = len(elements)
    results = []
    combo = []

    def backtrack(i):
        if len(combo) == k:
            results.append(combo[:])
            return
        if i == n:
            return
        # 가지치기: 남은 원소가 모자란다
        if len(combo) + (n - i) < k:
            return

        # a[i]을 넣는다
        combo.append(elements[i])
        backtrack(i + 1)
        combo.pop()

        # a[i]을 뺀다
        backtrack(i + 1)

    backtrack(0)
    return results


# === 메인 =====================================================================

if __name__ == "__main__":
    elements = [1, 2, 3, 4, 5]
    k = 3

    print(f"All {k}-combinations of {elements}:")
    for c in combinations(elements, k):
        print(f"  {c}")

    print(f"\nTotal: {len(combinations(elements, k))} combinations")
    print(f"Expected: C({len(elements)}, {k}) = "
          f"{len(combinations(elements, k))}")
```

**출력:**
```
All 3-combinations of [1, 2, 3, 4, 5]:
  [1, 2, 3]
  [1, 2, 4]
  [1, 2, 5]
  [1, 3, 4]
  [1, 3, 5]
  [1, 4, 5]
  [2, 3, 4]
  [2, 3, 5]
  [2, 4, 5]
  [3, 4, 5]

Total: 10 combinations
Expected: C(5, 3) = 10
```

## 복잡도 분석

**시간 복잡도.** 앞으로 고르기 방식은 잎을 정확히 $\binom{n}{k}$개 만든다. 잎마다 조합을 베끼는 데 $O(k)$이 든다. 안쪽 마디의 총수는 다음 이하이다

$$
\sum_{j=0}^{k} \binom{n}{j}
$$

안쪽 마디마다 $O(1)$ 일감(덧붙이기 하나와 빼내기 하나)을 하므로 전체 시간은 다음과 같다

$$
T = O\!\left(k \cdot \binom{n}{k}\right)
$$

내놓음 자체의 크기가 $\Theta\!\left(k \cdot \binom{n}{k}\right)$이므로 이는 내놓음에 대해 가장 좋다.

**공간 복잡도.** 되돌이 깊이는 많아야 $\min(k, n)$이고 조합 그릇은 많아야 $k$개를 담으므로 내놓기 말고 $O(k)$ 공간이 든다.

## 모든 부분 모임 만들어 내기

$k$을 $0, 1, \ldots, n$까지 돌리면 $2^n$개 부분 모임을 모두 만든다. 같은 일을 $k$을 고정하지 않고 두 갈래 결정 나무(넣기/빼기) 하나로도 할 수 있다:

```
ALL_SUBSETS(i, subset):
    if i == n:
        output subset
        return

    // Include a[i]
    subset.append(a[i])
    ALL_SUBSETS(i + 1, subset)
    subset.pop()

    // Exclude a[i]
    ALL_SUBSETS(i + 1, subset)
```

이는 $O(n \cdot 2^n)$ 시간과 $O(n)$ 공간으로 $2^n$개 부분 모임을 모두 만든다.

## 자리 바꿈과의 관계

조합과 자리 바꿈은 다음으로 이어진다

$$
P(n, k) = k! \cdot \binom{n}{k}
$$

여기서 $P(n, k) = n! / (n - k)!$은 $k$-자리 바꿈의 수이다. $k$-자리 바꿈을 모두 만들려면 먼저 $k$-조합을 모두 만들고 조합마다 자리를 바꾼다.

## 참고 문헌

- Skiena, *The Algorithm Design Manual*, 9장: Combinatorial Search,
  [algorist.com](https://www.algorist.com/)

## 연습문제

**연습문제 1.**
조합 만들어 내기의 고갱이 생각과 그것이 풀이 공간을 어떻게 짜임새 있게 살피는지 설명하라.

??? success "연습문제 1 풀이"
    조합 만들어 내기은 풀이 공간을 나무로 보고 살피며 마디마다 어중간한 풀이를 뜻한다. 마디마다 알고리즘은 어중간한 풀이를 넓히고 될 수 있는지 제약을 살핀다. 어중간한 풀이가 제약을 어기거나 (가장 좋거나 옳은 온전한 풀이로 이어질 수 없음이 밝혀지면) 알고리즘은 **가지를 쳐**(되짚어) 그 아래 나무 전체를 살피지 않는다. 가지치기가 찾기 공간의 큰 몫을 없애므로 막무가내보다 효율이 좋다. $\square$

---

**연습문제 2.**
조합 만들어 내기의 최악의 경우 시간 복잡도는 무엇인가? 가지치기는 언제 찾기 공간을 크게 줄이는가?

??? success "연습문제 2 풀이"
    최악의 경우(가지치기가 없으면) 알고리즘이 풀이 공간 전체를 살피며 이는 흔히 지수나 계승이다. 곧 갈래 수가 $b$이고 깊이가 $d$이면 $O(b^d)$, 자리 바꿈 문제이면 $O(n!)$이다. 가지치기는 다음일 때 찾기를 크게 줄인다. (1) 제약이 빡빡해 될 수 없는 갈래가 많을 때, (2) 좋은 묶음이 갈래를 일찍 없앨 때, (3) 차례를 매기는 어림짐작이 그럴듯한 갈래를 먼저 살필 때이다. 실전에서 가지치기는 도는 시간을 자릿수만큼 줄일 수 있다. $\square$

---

**연습문제 3.**
조합 만들어 내기의 가지치기 조건을 적어라. 무엇이 좋은 가지치기 잣대를 만드는가?

??? success "연습문제 3 풀이"
    가지치기 잣대는 어중간한 풀이를 언제 버릴지 정한다. 좋은 잣대는 다음과 같다. (1) **될 수 있음**: 어중간한 풀이가 이미 제약을 어긴다. (2) **묶음**: 어중간한 풀이를 가장 좋게 마무리해도 여태 가장 좋은 풀이보다 나을 수 없다. (3) **누름**: 다른 어중간한 풀이가 적어도 그만큼 좋음이 밝혀진다. 잘 듣는 가지치기 잣대는 따지기 값싸고 큰 아래 나무를 없앤다. $\square$

---

**연습문제 4.**
작은 경우에 조합 만들어 내기을 짜고 살핀 마디의 수를 전체 찾기 공간의 크기와 견주어 세어라.

??? success "연습문제 4 풀이"
    작은 경우(예컨대 N-여왕에서 $n = 8$, 배낭에서 담이 20)에는 전체 찾기 공간에 마디가 수백만 개일 수 있지만 가지치기가 잘 들면 수천 개만 살핀다. (살핀 수 / 전체) 비가 가지치기가 얼마나 잘 드는지 값으로 나타낸다. 제약이 잘 걸린 문제에서는 이 비가 1% 아래일 수 있어 되짚기가 막무가내보다 힘이 셈을 보여 준다. $\square$
