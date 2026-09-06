# 부분 모임 합

부분 모임 합 문제는 주어진 정수 모임에 원소의 합이 정해진 목표가 되는 부분 모임이 있는지 묻는다. 카프의 21가지 NP 완전 문제 가운데 하나이며 차례 짜기, 암호, 자원 나누기에서 아래 문제로 자주 나온다. 되짚기는 원소마다 넣거나 빼는 결정을 해서 이를 풀며, 힘 있는 가지치기 규칙 둘이 $2^n$개 부분 모임을 다 세는 막무가내에 견주어 찾기 공간을 크게 줄인다.

## 문제 서술

**들임.** 양의 정수 $n$개의 모임 $S = \{a_1, a_2, \ldots, a_n\}$과 목표 값 $T > 0$.

**내놓기.** $\sum_{a \in A} a = T$인 부분 모임 $A \subseteq S$, 또는 그런 부분 모임이 없다는 알림.

!!! info "양의 정수라는 가정"

    양의 정수로 좁히면 아래의 "목표 넘음" 가지치기 규칙을 쓸 수 있다. 아무(음수일 수도 있는) 정수를 다루는 두루 쓰는 문제도 NP 완전이지만 가지치기 분석이 달라진다.

## 되짚기로 세우기

### 상태 공간 나무

- **결정 $i$**($i = 1, \ldots, n$): $a_i$을 부분 모임에 넣거나($x_i = 1$) 뺀다($x_i = 0$).
- **갈래 수**: 층마다 2.
- **온전한 나무**: 잎이 $2^n$개이며 저마다 서로 다른 부분 모임에 맞닿는다.

### 될 수 있는지 살피기(가지치기)

$\text{sum}_k = \sum_{i=1}^{k} x_i \, a_i$을 앞선 $k$번 결정 뒤의 흐르는 합, $\text{remaining}_k = \sum_{i=k+1}^{n} a_i$을 아직 정하지 않은 원소의 합이라 하자.

가지치기 조건이 둘 있다:

1. **목표 넘음**: $\text{sum}_k > T$이면 쳐 낸다. 양의 정수를 더 넣으면 합이 커질 뿐이다.

2. **목표에 못 미침**: $\text{sum}_k + \text{remaining}_k < T$이면 쳐 낸다. 남은 원소를 모두 넣어도 목표에 이를 수 없다.

뒷합 배열을 미리 셈해 두면 두 조건 모두 $O(1)$에 따진다.

### 정렬 어림짐작

찾기 앞서 원소를 **큰 차례로** 정렬하면 좋은 점이 둘 있다:

1. 큰 원소가 흐르는 합을 $T$ 너머로 빨리 밀어 목표 넘음 가지치기가 더 일찍 걸린다.
2. 뒷합이 더 빨리 줄어 목표에 못 미침 가지치기도 더 일찍 걸린다.

## 알고리즘

```
SUBSET_SUM(i, current_sum, target, suffix_sum):
    if current_sum == target:
        report solution
        return True

    if i == n:
        return False

    // Pruning
    if current_sum > target:
        return False                   // over-target
    if current_sum + suffix_sum[i] < target:
        return False                   // under-target

    // Include a[i]
    if SUBSET_SUM(i + 1, current_sum + a[i], target, suffix_sum):
        return True

    // Exclude a[i]
    if SUBSET_SUM(i + 1, current_sum, target, suffix_sum):
        return True

    return False
```

## 파이썬 구현

```python
"""
목표 넘음과 목표에 못 미침 가지치기를 곁들인 되짚기 부분 모임 합 풀개.

양의 정수 모임과 목표가 주어질 때 합이 목표가 되는 부분 모임을 찾거나
없다고 알린다.
"""


# === 풀개 ===================================================================

def subset_sum(numbers, target):
    """합이 *target*이 되는 *numbers*의 부분 모임을 찾는다.

    부분 모임을 목록으로 돌려주고, 풀이가 없으면 None을 돌려준다.
    원소는 양의 정수라고 가정한다.
    """
    nums = sorted(numbers, reverse=True)
    n = len(nums)

    # 목표에 못 미침 가지치기를 위해 뒷합을 미리 셈한다
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + nums[i]

    result = []

    def backtrack(i, current):
        if current == target:
            return True
        if i == n:
            return False
        if current > target:                # 목표 넘음 가지치기
            return False
        if current + suffix[i] < target:    # 목표에 못 미침 가지치기
            return False

        # nums[i]을 넣는다
        result.append(nums[i])
        if backtrack(i + 1, current + nums[i]):
            return True
        result.pop()

        # nums[i]을 뺀다
        if backtrack(i + 1, current):
            return True

        return False

    if backtrack(0, 0):
        return result
    return None


# === 풀이를 모두 찾기 ======================================================

def subset_sum_all(numbers, target):
    """합이 *target*이 되는 *numbers*의 부분 모임을 모두 찾는다."""
    nums = sorted(numbers, reverse=True)
    n = len(nums)

    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + nums[i]

    solutions = []
    current_subset = []

    def backtrack(i, current):
        if current == target:
            solutions.append(current_subset[:])
            return                      # 일찍 돌아가지 않는다 — 모두 찾는다
        if i == n:
            return
        if current > target:
            return
        if current + suffix[i] < target:
            return

        # nums[i]을 넣는다
        current_subset.append(nums[i])
        backtrack(i + 1, current + nums[i])
        current_subset.pop()

        # nums[i]을 뺀다
        backtrack(i + 1, current)

    backtrack(0, 0)
    return solutions


# === 메인 =====================================================================

if __name__ == "__main__":
    numbers = [3, 7, 1, 8, 4, 12, 5]
    target = 15

    print(f"Numbers: {numbers}")
    print(f"Target:  {target}\n")

    result = subset_sum(numbers, target)
    if result is not None:
        print(f"One solution: {result}  (sum = {sum(result)})")
    else:
        print("No solution exists.")

    all_results = subset_sum_all(numbers, target)
    print(f"\nAll solutions ({len(all_results)} total):")
    for s in all_results:
        print(f"  {s}  (sum = {sum(s)})")
```

**출력:**
```
Numbers: [3, 7, 1, 8, 4, 12, 5]
Target:  15

One solution: [12, 3]  (sum = 15)

All solutions (5 total):
  [12, 3]  (sum = 15)
  [8, 7]  (sum = 15)
  [8, 4, 3]  (sum = 15)
  [7, 5, 3]  (sum = 15)
  [7, 4, 3, 1]  (sum = 15)
```

## 복잡도 분석

**시간 복잡도.** 최악의 경우 두 가지치기 규칙 모두 어떤 갈래도 없애지 못해 알고리즘이 $2^n$개 잎을 모두 들른다:

$$
T(n) = O(2^n)
$$

정렬 어림짐작과 두 가지치기 규칙을 함께 쓰면 대개의 경우 실제로 도는 시간이 훨씬 짧다. 다만 다항 시간 알고리즘은 알려져 있지 않다(이 문제는 NP 완전이다).

**공간 복잡도.** 되돌이 깊이가 $n$이고 뒷합 배열이 $O(n)$ 공간을 쓴다. 전체 공간은 $O(n)$이다.

## 동적 계획과의 견줌

목표 $T$이 그리 크지 않으면 부분 모임 합 문제는 시간 $O(nT)$, 공간 $O(T)$인 유사 다항 시간 동적 짜기 풀이가 있다. 되짚기 방식은 다음일 때 낫다:

- $T$이 아주 클 때(동적 짜기 표가 너무 커진다).
- 풀이 하나만 필요할 때(되짚기는 처음 하나를 찾고 멈출 수 있다).
- 가지치기 규칙이 잘 들 때(갈래가 일찍 많이 잘린다).

| 방법 | 시간 | 공간 | 언제 가장 좋은가 |
|--------|------|-------|-----------|
| 되짚기 | 최악 $O(2^n)$ | $O(n)$ | $n$이 작고 $T$이 크며 가지치기가 셀 때 |
| 동적 짜기 | $O(nT)$ | $O(T)$ | $n$과 $T$이 웬만할 때 |

## 참고 문헌

- Karp, "Reducibility among Combinatorial Problems," 1972
- Skiena, *The Algorithm Design Manual*, 9장: Combinatorial Search,
  [algorist.com](https://www.algorist.com/)

## 연습문제

**연습문제 1.**
부분 모임 합의 고갱이 생각과 그것이 풀이 공간을 어떻게 짜임새 있게 살피는지 설명하라.

??? success "연습문제 1 풀이"
    부분 모임 합은 풀이 공간을 나무로 보고 살피며 마디마다 어중간한 풀이를 뜻한다. 마디마다 알고리즘은 어중간한 풀이를 넓히고 될 수 있는지 제약을 살핀다. 어중간한 풀이가 제약을 어기거나 (가장 좋거나 옳은 온전한 풀이로 이어질 수 없음이 밝혀지면) 알고리즘은 **가지를 쳐**(되짚어) 그 아래 나무 전체를 살피지 않는다. 가지치기가 찾기 공간의 큰 몫을 없애므로 막무가내보다 효율이 좋다. $\square$

---

**연습문제 2.**
부분 모임 합의 최악의 경우 시간 복잡도는 무엇인가? 가지치기는 언제 찾기 공간을 크게 줄이는가?

??? success "연습문제 2 풀이"
    최악의 경우(가지치기가 없으면) 알고리즘이 풀이 공간 전체를 살피며 이는 흔히 지수나 계승이다. 곧 갈래 수가 $b$이고 깊이가 $d$이면 $O(b^d)$, 자리 바꿈 문제이면 $O(n!)$이다. 가지치기는 다음일 때 찾기를 크게 줄인다. (1) 제약이 빡빡해 될 수 없는 갈래가 많을 때, (2) 좋은 묶음이 갈래를 일찍 없앨 때, (3) 차례를 매기는 어림짐작이 그럴듯한 갈래를 먼저 살필 때이다. 실전에서 가지치기는 도는 시간을 자릿수만큼 줄일 수 있다. $\square$

---

**연습문제 3.**
부분 모임 합의 가지치기 조건을 적어라. 무엇이 좋은 가지치기 잣대를 만드는가?

??? success "연습문제 3 풀이"
    가지치기 잣대는 어중간한 풀이를 언제 버릴지 정한다. 좋은 잣대는 다음과 같다. (1) **될 수 있음**: 어중간한 풀이가 이미 제약을 어긴다. (2) **묶음**: 어중간한 풀이를 가장 좋게 마무리해도 여태 가장 좋은 풀이보다 나을 수 없다. (3) **누름**: 다른 어중간한 풀이가 적어도 그만큼 좋음이 밝혀진다. 잘 듣는 가지치기 잣대는 따지기 값싸고 큰 아래 나무를 없앤다. $\square$

---

**연습문제 4.**
작은 경우에 부분 모임 합을 짜고 살핀 마디의 수를 전체 찾기 공간의 크기와 견주어 세어라.

??? success "연습문제 4 풀이"
    작은 경우(예컨대 N-여왕에서 $n = 8$, 배낭에서 담이 20)에는 전체 찾기 공간에 마디가 수백만 개일 수 있지만 가지치기가 잘 들면 수천 개만 살핀다. (살핀 수 / 전체) 비가 가지치기가 얼마나 잘 드는지 값으로 나타낸다. 제약이 잘 걸린 문제에서는 이 비가 1% 아래일 수 있어 되짚기가 막무가내보다 힘이 셈을 보여 준다. $\square$
