# 자리 바꿈 만들어 내기

차례 짜기, 배정, 길 잡기 같은 많은 조합 문제는 원소 모임의 모든 차례를 살펴야 한다. 원소 $n$개의 **자리 바꿈**은 $\{1, \ldots, n\}$에서 자기 자신으로 가는 일대일 대응이며 그런 대응이 $n!$개 있다. 되짚기는 원소를 다시 쓰지 않게 하면서 자리마다 하나씩 채워 $n!$개 자리 바꿈을 짜임새 있게 만든다.

## 문제 서술

**들임.** 서로 다른 원소 $n$개의 모임 $\{a_1, a_2, \ldots, a_n\}$.

**내놓기.** 그 모임의 자리 바꿈 $n!$개 모두.

## 되짚기로 세우기

### 상태 공간 나무

- **결정 $k$**($k = 1, \ldots, n$): 아직 쓰지 않은 원소 가운데 어느 것이 자리 $k$에 놓일지 고른다.
- **Branching factor**: $n - k + 1$ at level $k$ (the number of unused elements).
- **Full tree**: the tree has $n!$ leaves, one per permutation.

### 될 수 있는지 살피기

제약은 원소마다 꼭 한 번 나온다는 것뿐이다. 될 수 있는지 살피기는 "원소 $a_i$을 자리 $1, \ldots, k-1$에서 이미 썼는가?"로 줄어든다. 참거짓 배열 `used[i]`이 이를 $O(1)$에 답한다.

## 방식 1 --- 골라 쓰기

At each level $k$, iterate over all $n$ elements and select those not yet used.

```
PERMUTATIONS(perm, k, n):
    if k == n:
        output perm
        return

    for i = 0 to n - 1:
        if not used[i]:
            perm[k] = a[i]
            used[i] = True
            PERMUTATIONS(perm, k + 1, n)
            used[i] = False
```

### 복잡도

이 알고리즘은 잎을 정확히 $n!$개 만든다. 안쪽 마디마다 쓰지 않은 원소를 찾으려 `used` 배열을 $O(n)$에 훑는다. 안쪽 마디의 총수는
is

$$
\sum_{k=0}^{n-1} \frac{n!}{(n - k)!}
$$

이며 마디마다 $O(n)$ 일감을 하므로 전체 시간은 $O(n \cdot n!)$이다. 내놓는 것 자체의 크기가 $\Theta(n \cdot n!)$이므로 상수 인수까지 내놓기에 견주어 가장 좋다.

## 방식 2 --- 맞바꾸기(힙 꼴)

다른 길은 원소를 자리에 맞바꿔 넣어 `used` 배열을 아예 쓰지 않는 것이다. 층 $k$에서 자리 $k, k+1, \ldots, n-1$의 원소를 저마다 자리 $k$으로 맞바꾸고, 되돌이한 뒤, 도로 맞바꾼다.

```
PERMUTE_SWAP(arr, k, n):
    if k == n:
        output arr
        return

    for i = k to n - 1:
        swap(arr[k], arr[i])
        PERMUTE_SWAP(arr, k + 1, n)
        swap(arr[k], arr[i])       // undo
```

이 방식은 딸림 자료 짜임을 쓰지 않고 배열을 제자리에서 자리 바꾼다. 마디마다의 일감이 $O(1)$(맞바꿈 한 번)이므로 내놓기를 빼면 전체 일감이 $O(n!)$이다.

## 파이썬 구현

```python
"""
되짚기로 목록의 자리 바꿈을 모두 만든다.

골라 쓰는 방식과 맞바꾸는 방식을 모두 보인다.
"""


# === 골라 쓰는 방식 =================================================

def permutations_select(elements):
    """쓴 것 배열로 자리 바꿈을 모두 만든다."""
    n = len(elements)
    results = []
    perm = [None] * n
    used = [False] * n

    def backtrack(k):
        if k == n:
            results.append(perm[:])
            return
        for i in range(n):
            if not used[i]:
                perm[k] = elements[i]
                used[i] = True
                backtrack(k + 1)
                used[i] = False

    backtrack(0)
    return results


# === 맞바꾸는 방식 =====================================================

def permutations_swap(elements):
    """원소를 제자리에서 맞바꾸어 자리 바꿈을 모두 만든다."""
    arr = list(elements)
    n = len(arr)
    results = []

    def backtrack(k):
        if k == n:
            results.append(arr[:])
            return
        for i in range(k, n):
            arr[k], arr[i] = arr[i], arr[k]
            backtrack(k + 1)
            arr[k], arr[i] = arr[i], arr[k]

    backtrack(0)
    return results


# === 메인 =====================================================================

if __name__ == "__main__":
    elements = [1, 2, 3]

    print("Selection-based permutations:")
    for p in permutations_select(elements):
        print(f"  {p}")

    print(f"\nSwap-based permutations:")
    for p in permutations_swap(elements):
        print(f"  {p}")

    print(f"\nTotal: {len(permutations_select(elements))} permutations of "
          f"{len(elements)} elements")
```

**출력:**
```
Selection-based permutations:
  [1, 2, 3]
  [1, 3, 2]
  [2, 1, 3]
  [2, 3, 1]
  [3, 1, 2]
  [3, 2, 1]

Swap-based permutations:
  [1, 2, 3]
  [1, 3, 2]
  [2, 1, 3]
  [2, 3, 1]
  [3, 2, 1]
  [3, 1, 2]

Total: 6 permutations of 3 elements
```

!!! note "차례의 차이"

    두 방식은 자리 바꿈을 다른 차례로 낸다. 골라 쓰기는 원소가 처음에 정렬되어 있으면 사전 차례로 만든다. 맞바꾸기는 (여전히 모두를 내지만) 다른 차례로 만든다.

## 복잡도 분석

| 잣대 | 골라 쓰기 | 맞바꾸기 |
|--------|----------------|------------|
| 잎 | $n!$ | $n!$ |
| 안쪽 마디 | $\sum_{k=0}^{n-1} n!/(n-k)!$ | 같음 |
| 마디마다의 일감 | $O(n)$ | $O(1)$ |
| 전체 시간 | $O(n \cdot n!)$ | $O(n!)$ + 내놓기 $O(n \cdot n!)$ |
| 공간 | $O(n)$ | $O(n)$(되돌이 쌓임) |

길이 $n$짜리 자리 바꿈 $n!$개를 만들려면 내놓는 것을 적는 데만 $\Omega(n \cdot n!)$ 시간이 들므로 두 방식 모두 내놓기에 견주어 가장 좋다.

## 사전 차례로 만들어 내기

한꺼번에 모두가 아니라 **다음** 자리 바꿈만 필요할 때는 다음 자리 바꿈 알고리즘이 자리 바꿈마다 고르게 나눈 $O(1)$ 시간에 사전 차례로 만든다:

1. $a[i] < a[i + 1]$인 가장 큰 번호 $i$을 찾는다. 없으면 지금
   자리 바꿈이 마지막이다.
2. $a[i] < a[j]$인 가장 큰 번호 $j > i$을 찾는다.
3. $a[i]$과 $a[j]$을 맞바꾼다.
4. 뒷가지 $a[i+1], \ldots, a[n-1]$을 뒤집는다.

이 되풀이 방식은 $O(1)$ 공간만 더 쓰고 되짚기의 되돌이 군더더기를 피한다.

## 참고 문헌

- Skiena, *The Algorithm Design Manual*, 9장: Combinatorial Search,
  [algorist.com](https://www.algorist.com/)
- Sedgewick, "Permutation Generation Methods," *ACM Computing Surveys*, 1977

## 연습문제

**연습문제 1.**
자리 바꿈 만들어 내기의 고갱이 생각과 그것이 풀이 공간을 어떻게 짜임새 있게 살피는지 설명하라.

??? success "연습문제 1 풀이"
    자리 바꿈 만들어 내기은 풀이 공간을 나무로 보고 살피며 마디마다 어중간한 풀이를 뜻한다. 마디마다 알고리즘은 어중간한 풀이를 넓히고 될 수 있는지 제약을 살핀다. 어중간한 풀이가 제약을 어기거나 (가장 좋거나 옳은 온전한 풀이로 이어질 수 없음이 밝혀지면) 알고리즘은 **가지를 쳐**(되짚어) 그 아래 나무 전체를 살피지 않는다. 가지치기가 찾기 공간의 큰 몫을 없애므로 막무가내보다 효율이 좋다. $\square$

---

**연습문제 2.**
자리 바꿈 만들어 내기의 최악의 경우 시간 복잡도는 무엇인가? 가지치기는 언제 찾기 공간을 크게 줄이는가?

??? success "연습문제 2 풀이"
    최악의 경우(가지치기가 없으면) 알고리즘이 풀이 공간 전체를 살피며 이는 흔히 지수나 계승이다. 곧 갈래 수가 $b$이고 깊이가 $d$이면 $O(b^d)$, 자리 바꿈 문제이면 $O(n!)$이다. 가지치기는 다음일 때 찾기를 크게 줄인다. (1) 제약이 빡빡해 될 수 없는 갈래가 많을 때, (2) 좋은 묶음이 갈래를 일찍 없앨 때, (3) 차례를 매기는 어림짐작이 그럴듯한 갈래를 먼저 살필 때이다. 실전에서 가지치기는 도는 시간을 자릿수만큼 줄일 수 있다. $\square$

---

**연습문제 3.**
자리 바꿈 만들어 내기의 가지치기 조건을 적어라. 무엇이 좋은 가지치기 잣대를 만드는가?

??? success "연습문제 3 풀이"
    가지치기 잣대는 어중간한 풀이를 언제 버릴지 정한다. 좋은 잣대는 다음과 같다. (1) **될 수 있음**: 어중간한 풀이가 이미 제약을 어긴다. (2) **묶음**: 어중간한 풀이를 가장 좋게 마무리해도 여태 가장 좋은 풀이보다 나을 수 없다. (3) **누름**: 다른 어중간한 풀이가 적어도 그만큼 좋음이 밝혀진다. 잘 듣는 가지치기 잣대는 따지기 값싸고 큰 아래 나무를 없앤다. $\square$

---

**연습문제 4.**
작은 경우에 자리 바꿈 만들어 내기을 짜고 살핀 마디의 수를 전체 찾기 공간의 크기와 견주어 세어라.

??? success "연습문제 4 풀이"
    작은 경우(예컨대 N-여왕에서 $n = 8$, 배낭에서 담이 20)에는 전체 찾기 공간에 마디가 수백만 개일 수 있지만 가지치기가 잘 들면 수천 개만 살핀다. (살핀 수 / 전체) 비가 가지치기가 얼마나 잘 드는지 값으로 나타낸다. 제약이 잘 걸린 문제에서는 이 비가 1% 아래일 수 있어 되짚기가 막무가내보다 힘이 셈을 보여 준다. $\square$
