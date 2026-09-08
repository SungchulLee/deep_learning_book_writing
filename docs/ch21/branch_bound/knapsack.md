# 가지 뻗어 묶기로 푸는 0/1 배낭

0/1 배낭 문제는 동적 짜기로 $O(nW)$ 시간에 풀 수 있지만 $W$이 아주 크면 이 유사 다항 방식은 현실성이 없다. 가지 뻗어 묶기는 다른 길을 준다. 곧 물건마다 넣거나 빼는 두 갈래 결정 나무를 살피면서 묶는 함수로 더 나은 풀이로 이어질 수 없는 아래 나무를 쳐 낸다. 실전에서 좋은 묶음을 갖춘 가지 뻗어 묶기는 다 뒤지기보다 배낭 문제를 훨씬 빨리 푼다.

---

## 1. 찾기 나무의 짜임

층 $i$의 마디마다 물건 $1, \ldots, i$에 대한 어중간한 결정을 뜻한다. 두 자식은 다음에 맞닿는다:

- 물건 $i+1$을 **넣는다**: 그 무게와 값어치를 흐르는 합에 더한다.
- 물건 $i+1$을 **뺀다**: 바뀜 없이 다음 물건으로 간다.

층 $n$에서 온전한 풀이에 이른다(모든 물건이 정해졌다). 최악의 경우 나무의 잎은 $2^n$개이다.

---

## 2. 쪼갤 수 있게 느슨히 해 묶기

값어치 대 무게 비 $r_i = v_i / w_i$이 큰 차례로 물건을 정렬한다. 쌓인 값어치가 $V$이고 남은 담이가 $C$인 마디에서 남은 담이를 쪼갠 물건으로 욕심껏 채워 위 묶음을 셈한다:

$$
\text{UB}(V, C, \text{level}) = V + \sum_{i=\text{level}}^{k-1} v_i + v_k \cdot \frac{C - \sum_{i=\text{level}}^{k-1} w_i}{w_k}
$$

여기서 $k$은 온전히 들어가지 않는 첫 물건이다. 이 쪼갤 수 있게 느슨히 한 위 묶음은 늘 그 아래 나무의 가장 좋은 정수 풀이 이상이다.

---

## 3. 알고리즘

1. $v_i / w_i$ 내림차순으로 물건을 정렬한다.
2. `best_value = 0`으로 첫자리매김한다.
3. 깊이 먼저(또는 가장 좋은 것 먼저)로 두 갈래 나무를 살핀다.
4. 마디마다 위 묶음을 셈한다.
5. 위 묶음이 `best_value` 이하이면 **가지를 친다**.
6. 잎 마디에서 지금 풀이가 더 좋으면 `best_value`을 새로 고친다.

---

## 4. 풀이 예제

**물건**(비로 정렬): $(w, v) = \{(2, 12), (4, 20), (6, 18), (9, 18)\}$, 담이 $W = 15$.

비: $r = [6, 5, 3, 2]$.

- **뿌리**: 위 묶음 $= 12 + 20 + 18 \cdot (9/6) = 12 + 20 + 27 = 59$(쪼갠 값). 살핀다.
- **물건 1을 넣는다**($V=12, C=13$): 위 묶음 = $12 + 20 + 18 \cdot (3/6) = 41$. 살핀다.
- **물건 2를 넣는다**($V=32, C=9$): 위 묶음 = $32 + 18 \cdot (9/6) = 59$. 실은 위 묶음 $= 32 + 18 = 50$이다(물건 3이 들어간다). 이어 간다.
- ... 마침내 가장 좋은 값을 찾는다.

---

## 5. 파이썬 구현

```python
"""
0/1 배낭 — 깊이 먼저를 쓴 가지 뻗어 묶기.

쪼갤 수 있는 배낭으로 느슨히 한 것을 위 묶음으로 삼아 여태 가장 좋은 풀이를
넘을 수 없는 갈래를 쳐 낸다.
"""

# === 쪼갤 수 있게 느슨히 해 얻는 위 묶음 ===

def upper_bound(
    level: int, value: int, weight: int,
    weights: list[int], values: list[int], capacity: int
) -> float:
    """지금 마디에서의 쪼갤 수 있는 배낭 위 묶음."""
    if weight > capacity:
        return 0.0

    bound = float(value)
    remaining = capacity - weight
    n = len(weights)

    for i in range(level, n):
        if weights[i] <= remaining:
            bound += values[i]
            remaining -= weights[i]
        else:
            bound += values[i] * (remaining / weights[i])
            break

    return bound

# === 가지 뻗어 묶기(깊이 먼저) ===

def knapsack_branch_bound(
    weights: list[int], values: list[int], capacity: int
) -> tuple[int, list[int]]:
    """깊이 먼저 가지 뻗어 묶기로 0/1 배낭을 푼다.

    (최대 값어치, 본디 차례의 고른 번호)를 돌려준다.
    """
    n = len(weights)
    # 값어치/무게 비 내림차순으로 정렬한다
    order = sorted(range(n), key=lambda i: values[i] / weights[i], reverse=True)
    w = [weights[i] for i in order]
    v = [values[i] for i in order]

    best_value = 0
    best_selection = [0] * n
    current = [0] * n
    nodes_explored = 0

    def dfs(level: int, curr_value: int, curr_weight: int) -> None:
        nonlocal best_value, best_selection, nodes_explored
        nodes_explored += 1

        if level == n:
            if curr_value > best_value:
                best_value = curr_value
                best_selection = current[:]
            return

        # 이 층의 물건을 넣어 본다
        if curr_weight + w[level] <= capacity:
            current[level] = 1
            ub = upper_bound(
                level + 1, curr_value + v[level],
                curr_weight + w[level], w, v, capacity
            )
            if ub > best_value:
                dfs(level + 1, curr_value + v[level], curr_weight + w[level])
            current[level] = 0

        # 이 층의 물건을 빼 본다
        ub = upper_bound(level + 1, curr_value, curr_weight, w, v, capacity)
        if ub > best_value:
            dfs(level + 1, curr_value, curr_weight)

    dfs(0, 0, 0)

    # 고른 물건을 본디 번호로 되돌린다
    selected = [order[i] for i in range(n) if best_selection[i]]
    return best_value, sorted(selected)

# === 메인 ===

if __name__ == "__main__":
    weights = [10, 20, 30]
    values = [60, 100, 120]
    capacity = 50

    max_val, selected = knapsack_branch_bound(weights, values, capacity)
    print(f"Weights:  {weights}")
    print(f"Values:   {values}")
    print(f"Capacity: {capacity}")
    print(f"Max value: {max_val}")
    print(f"Selected items: {selected}")
    # 내임:
    # 무게:  [10, 20, 30]
    # 값어치:   [60, 100, 120]
    # 담이: 50
    # 최대 값어치: 220
    # 고른 물건: [1, 2]
```

**출력:**

```
Weights:  [10, 20, 30]
Values:   [60, 100, 120]
Capacity: 50
Max value: 220
Selected items: [1, 2]
```

---

## 6. 동적 계획과의 견줌

| 갈래 | 동적 짜기 | 가지 뻗어 묶기 |
|---|---|---|
| 시간 복잡도 | $O(nW)$ 유사 다항 | 최악의 경우 지수 |
| 언제 빠른가 | $W$이 작거나 가운데일 때 | 묶음이 빡빡하고 물건이 적을 때 |
| 공간 | $O(nW)$ 또는 $O(W)$ | 쌓임 $O(n)$ + 묶음 셈하기 |
| $W$이 클 때 | 현실성 없음 | 좋은 묶음이면 흔히 쓸 만함 |

!!! warning "최악의 경우 복잡도"
    가지 뻗어 묶기의 최악의 경우 시간은 $O(2^n)$이다. 실전의 이점은 가지치기에서 오며 이는 문제의 경우와 묶음 품질에 달렸다. 어떤 경우에는 나무 거의 전체를 살핀다.

---

## 연습문제

**연습문제 1.**
가지 뻗어 묶기로 푸는 0/1 배낭의 고갱이 생각과 그것이 풀이 공간을 어떻게 짜임새 있게 살피는지 설명하라.

??? success "연습문제 1 풀이"
    가지 뻗어 묶기로 푸는 0/1 배낭은 풀이 공간을 나무로 보고 살피며 마디마다 어중간한 풀이를 뜻한다. 마디마다 알고리즘은 어중간한 풀이를 넓히고 될 수 있는지 제약을 살핀다. 어중간한 풀이가 제약을 어기거나 (가장 좋거나 옳은 온전한 풀이로 이어질 수 없음이 밝혀지면) 알고리즘은 **가지를 쳐**(되짚어) 그 아래 나무 전체를 살피지 않는다. 가지치기가 찾기 공간의 큰 몫을 없애므로 막무가내보다 효율이 좋다. $\square$

---

**연습문제 2.**
가지 뻗어 묶기로 푸는 0/1 배낭의 최악의 경우 시간 복잡도는 무엇인가? 가지치기는 언제 찾기 공간을 크게 줄이는가?

??? success "연습문제 2 풀이"
    최악의 경우(가지치기가 없으면) 알고리즘이 풀이 공간 전체를 살피며 이는 흔히 지수나 계승이다. 곧 갈래 수가 $b$이고 깊이가 $d$이면 $O(b^d)$, 자리 바꿈 문제이면 $O(n!)$이다. 가지치기는 다음일 때 찾기를 크게 줄인다. (1) 제약이 빡빡해 될 수 없는 갈래가 많을 때, (2) 좋은 묶음이 갈래를 일찍 없앨 때, (3) 차례를 매기는 어림짐작이 그럴듯한 갈래를 먼저 살필 때이다. 실전에서 가지치기는 도는 시간을 자릿수만큼 줄일 수 있다. $\square$

---

**연습문제 3.**
가지 뻗어 묶기로 푸는 0/1 배낭의 가지치기 조건을 적어라. 무엇이 좋은 가지치기 잣대를 만드는가?

??? success "연습문제 3 풀이"
    가지치기 잣대는 어중간한 풀이를 언제 버릴지 정한다. 좋은 잣대는 다음과 같다. (1) **될 수 있음**: 어중간한 풀이가 이미 제약을 어긴다. (2) **묶음**: 어중간한 풀이를 가장 좋게 마무리해도 여태 가장 좋은 풀이보다 나을 수 없다. (3) **누름**: 다른 어중간한 풀이가 적어도 그만큼 좋음이 밝혀진다. 잘 듣는 가지치기 잣대는 따지기 값싸고 큰 아래 나무를 없앤다. $\square$

---

**연습문제 4.**
작은 경우에 가지 뻗어 묶기로 푸는 0/1 배낭을 짜고 살핀 마디의 수를 전체 찾기 공간의 크기와 견주어 세어라.

??? success "연습문제 4 풀이"
    작은 경우(예컨대 N-여왕에서 $n = 8$, 배낭에서 담이 20)에는 전체 찾기 공간에 마디가 수백만 개일 수 있지만 가지치기가 잘 들면 수천 개만 살핀다. (살핀 수 / 전체) 비가 가지치기가 얼마나 잘 드는지 값으로 나타낸다. 제약이 잘 걸린 문제에서는 이 비가 1% 아래일 수 있어 되짚기가 막무가내보다 힘이 셈을 보여 준다. $\square$

## 정리하며

이 마당은 찾기 나무의 짜임、쪼갤 수 있게 느슨히 해 묶기、알고리즘、풀이 예제을 차례로 짚었다.

**참고 문헌**

- Skiena, S. S. (2020). *The Algorithm Design Manual* (3rd ed.), Chapter 9. Springer.
- Horowitz, E., & Sahni, S. (1974). Computing partitions with applications to the knapsack problem. *Journal of the ACM*, 21(2), 277-292.
