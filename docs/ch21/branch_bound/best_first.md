# 가장 좋은 것 먼저 가지 뻗어 묶기

여느 되짚기는 찾기 나무를 깊이 먼저 살피므로 좋은 풀이를 만나기 앞서 그럴듯하지 않은 갈래에 시간을 버릴 수 있다. **가장 좋은 것 먼저 가지 뻗어 묶기**는 깊이 먼저 쌓임을 우선순위 줄로 갈음해 늘 가장 그럴듯한 마디를 다음에 펼친다. 똑똑한 마디 고르기와 묶기를 아울러, 가장 좋은 것 먼저 찾기는 다 뒤지기보다 훨씬 적은 마디만 살피고도 흔히 가장 좋은 풀이를 찾는다.

---

## 1. 핵심 생각

다음 마디를 쌓임 차례(깊이 먼저)나 줄 차례(너비 먼저)로 고르는 대신, 가장 좋은 것 먼저 찾기는 **가장 좋은 묶음**을 가진 마디, 곧 그 마디에서 이룰 수 있는 목표 값을 가장 낙관해 어림한 마디를 고른다. 가장 작게 하는 문제에서는 아래 묶음이 가장 작은 마디를, 가장 크게 하는 문제에서는 위 묶음이 가장 큰 마디를 펼친다.

---

## 2. 알고리즘

**들임:** 묶는 함수를 갖춘 조합 가장 좋게 하기 문제.

1. 뿌리 마디를 그 묶음을 열쇠로 삼아 우선순위 줄에 넣는다.
2. `best_solution`을 (가장 작게 할 때는) $\infty$, (가장 크게 할 때는) $-\infty$으로 둔다.
3. 우선순위 줄이 비지 않은 동안:
    - 묶음이 가장 좋은 마디를 꺼낸다.
    - 마디의 묶음이 `best_solution`보다 나쁘면 **가지를 쳐** 건너뛴다.
    - 마디가 온전한 풀이이면 나아졌을 때 `best_solution`을 새로 고친다.
    - 아니면 **가지를 뻗는다**. 곧 자식 마디를 만들고 묶음을 셈해 그럴듯한 것을 우선순위 줄에 넣는다.
4. `best_solution`을 돌려준다.

!!! tip "가장 좋은 것 먼저가 도움이 되는 까닭"
    가장 좋은 것 먼저 찾기는 그럴듯한 마디를 앞세우므로 좋은 풀이를 일찍 찾는다. 좋은 풀이를 찾고 나면 그 값이 가지치기 잣대를 빡빡하게 만들어 남은 마디 상당수를 살피지 않고 쳐 낸다.

---

## 3. 깊이 먼저 바탕 가지 뻗어 묶기와의 견줌

| 성질 | 깊이 먼저(쌓임) | 가장 좋은 것 먼저(우선순위 줄) |
|---|---|---|
| 마디 고르기 | 나중에 든 것이 먼저 나감 | 묶음이 가장 좋은 것 먼저 |
| 기억 공간 | 깊이가 $d$일 때 $O(d)$ | 최악의 경우 $O(b^d)$ |
| 가장 좋은 것을 일찍 찾는가? | 보장 없음 | 흔히 그렇다 |
| 가지치기가 드는 정도 | 돌아보는 차례에 달렸다 | 높다 — 좋은 풀이를 일찍 찾는다 |

!!! warning "기억 공간 씀씀이"
    가장 좋은 것 먼저 찾기는 한꺼번에 마디를 많이 담을 수 있다. 최악의 경우 우선순위 줄이 잎 층의 마디를 모두 담는다. 기억 공간이 빠듯하면 깊이 먼저 바탕 가지 뻗어 묶기나 되풀이하며 깊이 늘리기가 나을 수 있다.

---

## 4. 묶는 함수의 품질

가장 좋은 것 먼저 찾기가 잘 드는지는 묶는 함수에 결정적으로 달렸다:

- **빡빡한 묶음**(참으로 가장 좋은 값에 가까운 것)은 마디를 더 많이 쳐 내어 찾기 공간을 줄인다.
- **헐거운 묶음**은 마디를 덜 쳐 내어 막무가내보다 나을 것이 별로 없다.
- 묶음을 셈하는 것이 빨라야 한다. 잘 쳐 내더라도 값비싼 묶음은 알고리즘 전체를 느리게 할 수 있다.

---

## 5. 파이썬 구현

```python
"""
가장 좋은 것 먼저 가지 뻗어 묶기 — 두루 쓰는 얼거리.

쪼갤 수 있는 배낭으로 느슨히 한 것을 묶는 함수로 삼아
0/1 배낭 문제에 가장 좋은 것 먼저 전략을 보인다.
"""

import heapq
from typing import NamedTuple

# === 마디 나타내기 ===

class Node(NamedTuple):
    """가지 뻗어 묶기 찾기 나무의 마디."""
    neg_bound: float   # 부호를 뒤집은 위 묶음(최소 무지로 최대 무지를 흉내내려)
    level: int         # 결정 층(다음에 살필 물건)
    value: int         # 여태 쌓인 값어치
    weight: int        # 여태 쌓인 무게

# === 묶는 함수 ===

def fractional_bound(
    level: int, value: int, weight: int,
    weights: list[int], values: list[int], capacity: int
) -> float:
    """쪼갤 수 있는 배낭으로 느슨히 해 얻는 위 묶음."""
    if weight > capacity:
        return 0.0

    bound = float(value)
    remaining = capacity - weight
    n = len(weights)

    # 남은 물건으로 욕심껏 채운다(값어치 밀도로 정렬)
    for i in range(level, n):
        if weights[i] <= remaining:
            bound += values[i]
            remaining -= weights[i]
        else:
            bound += values[i] * (remaining / weights[i])
            break

    return bound

# === 가장 좋은 것 먼저 가지 뻗어 묶기 ===

def knapsack_best_first(
    weights: list[int], values: list[int], capacity: int
) -> tuple[int, list[int]]:
    """가장 좋은 것 먼저 가지 뻗어 묶기로 0/1 배낭을 푼다.

    부르기 앞서 물건을 값어치/무게 비 내림차순으로 정렬해야 한다.
    (최대 값어치, 고른 물건)을 돌려준다.
    """
    n = len(weights)

    # 물건을 값어치 밀도 내림차순으로 정렬한다
    order = sorted(range(n), key=lambda i: values[i] / weights[i], reverse=True)
    w_sorted = [weights[i] for i in order]
    v_sorted = [values[i] for i in order]

    root_bound = fractional_bound(0, 0, 0, w_sorted, v_sorted, capacity)
    pq = [Node(-root_bound, 0, 0, 0)]
    best_value = 0
    nodes_explored = 0

    while pq:
        node = heapq.heappop(pq)
        neg_bound, level, value, weight = node
        nodes_explored += 1

        if -neg_bound <= best_value:
            continue  # 가지치기: 묶음이 나아질 수 없다

        if level == n:
            if value > best_value:
                best_value = value
            continue

        # 가지 뻗기: 지금 층의 물건을 넣는다
        new_w = weight + w_sorted[level]
        new_v = value + v_sorted[level]
        if new_w <= capacity:
            if new_v > best_value:
                best_value = new_v
            inc_bound = fractional_bound(
                level + 1, new_v, new_w, w_sorted, v_sorted, capacity
            )
            if inc_bound > best_value:
                heapq.heappush(pq, Node(-inc_bound, level + 1, new_v, new_w))

        # 가지 뻗기: 지금 층의 물건을 뺀다
        exc_bound = fractional_bound(
            level + 1, value, weight, w_sorted, v_sorted, capacity
        )
        if exc_bound > best_value:
            heapq.heappush(pq, Node(-exc_bound, level + 1, value, weight))

    return best_value, nodes_explored

# === 메인 ===

if __name__ == "__main__":
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 8

    max_val, explored = knapsack_best_first(weights, values, capacity)
    print(f"Weights: {weights}")
    print(f"Values:  {values}")
    print(f"Capacity: {capacity}")
    print(f"Maximum value: {max_val}")
    print(f"Nodes explored: {explored}")
    # 내임:
    # 무게: [2, 3, 4, 5]
    # 값어치:  [3, 4, 5, 6]
    # 담이: 8
    # 최대 값어치: 10
    # 살핀 마디: 7
```

---

## 연습문제

**연습문제 1.**
가장 좋은 것 먼저 가지 뻗어 묶기의 고갱이 생각과 그것이 풀이 공간을 어떻게 짜임새 있게 살피는지 설명하라.

??? success "연습문제 1 풀이"
    가장 좋은 것 먼저 가지 뻗어 묶기은 풀이 공간을 나무로 보고 살피며 마디마다 어중간한 풀이를 뜻한다. 마디마다 알고리즘은 어중간한 풀이를 넓히고 될 수 있는지 제약을 살핀다. 어중간한 풀이가 제약을 어기거나 (가장 좋거나 옳은 온전한 풀이로 이어질 수 없음이 밝혀지면) 알고리즘은 **가지를 쳐**(되짚어) 그 아래 나무 전체를 살피지 않는다. 가지치기가 찾기 공간의 큰 몫을 없애므로 막무가내보다 효율이 좋다. $\square$

---

**연습문제 2.**
가장 좋은 것 먼저 가지 뻗어 묶기의 최악의 경우 시간 복잡도는 무엇인가? 가지치기는 언제 찾기 공간을 크게 줄이는가?

??? success "연습문제 2 풀이"
    최악의 경우(가지치기가 없으면) 알고리즘이 풀이 공간 전체를 살피며 이는 흔히 지수나 계승이다. 곧 갈래 수가 $b$이고 깊이가 $d$이면 $O(b^d)$, 자리 바꿈 문제이면 $O(n!)$이다. 가지치기는 다음일 때 찾기를 크게 줄인다. (1) 제약이 빡빡해 될 수 없는 갈래가 많을 때, (2) 좋은 묶음이 갈래를 일찍 없앨 때, (3) 차례를 매기는 어림짐작이 그럴듯한 갈래를 먼저 살필 때이다. 실전에서 가지치기는 도는 시간을 자릿수만큼 줄일 수 있다. $\square$

---

**연습문제 3.**
가장 좋은 것 먼저 가지 뻗어 묶기의 가지치기 조건을 적어라. 무엇이 좋은 가지치기 잣대를 만드는가?

??? success "연습문제 3 풀이"
    가지치기 잣대는 어중간한 풀이를 언제 버릴지 정한다. 좋은 잣대는 다음과 같다. (1) **될 수 있음**: 어중간한 풀이가 이미 제약을 어긴다. (2) **묶음**: 어중간한 풀이를 가장 좋게 마무리해도 여태 가장 좋은 풀이보다 나을 수 없다. (3) **누름**: 다른 어중간한 풀이가 적어도 그만큼 좋음이 밝혀진다. 잘 듣는 가지치기 잣대는 따지기 값싸고 큰 아래 나무를 없앤다. $\square$

---

**연습문제 4.**
작은 경우에 가장 좋은 것 먼저 가지 뻗어 묶기을 짜고 살핀 마디의 수를 전체 찾기 공간의 크기와 견주어 세어라.

??? success "연습문제 4 풀이"
    작은 경우(예컨대 N-여왕에서 $n = 8$, 배낭에서 담이 20)에는 전체 찾기 공간에 마디가 수백만 개일 수 있지만 가지치기가 잘 들면 수천 개만 살핀다. (살핀 수 / 전체) 비가 가지치기가 얼마나 잘 드는지 값으로 나타낸다. 제약이 잘 걸린 문제에서는 이 비가 1% 아래일 수 있어 되짚기가 막무가내보다 힘이 셈을 보여 준다. $\square$

## 정리하며

이 마당은 핵심 생각、알고리즘、깊이 먼저 바탕 가지 뻗어 묶기와의 견줌、묶는 함수의 품질을 차례로 짚었다.

**참고 문헌**

- Skiena, S. S. (2020). *The Algorithm Design Manual* (3rd ed.), Chapter 9. Springer.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.
