# 구간 나누기

구간 일정 짜기가 자원 하나에서 활동의 수를 가장 크게 하는 것이라면, **구간 나누기**는 다른 물음을 던진다. 곧 활동 모음이 주어질 때 모든 활동이 부딪힘 없이 돌아가려면 자원(방, 기계, 처리기)이 최소 몇 개 필요한가? 활동마다 가장 먼저 비는 자원에 배정하는 욕심쟁이 알고리즘이 이를 가장 좋게 풀며, 답은 늘 어느 한때에 겹치는 활동의 최대 수와 같다.

---

## 1. 문제 서술

구간이 $[s_i, f_i)$인 활동 $n$개가 주어질 때, 같은 자원에 배정한 두 활동이 겹치지 않도록 활동마다 자원(방)을 배정하여라. 쓰는 자원의 수를 가장 작게 하여라.

---

## 2. 아래 한계: 깊이

구간 모음의 **깊이**는 어떤 한 점을 함께 담는 구간의 최대 개수이다:

$$
\text{depth} = \max_{t} |\{i : s_i \le t < f_i\}|
$$

가장 많이 겹치는 지점에서 겹치는 활동마다 제 자원이 필요하므로, 어떤 일정도 깊이보다 적은 자원을 쓸 수 없다.

!!! note "정리"
    필요한 자원의 최소 개수는 깊이와 같다.

---

## 3. 욕심쟁이 알고리즘

**전략.** 활동을 시작 시각으로 정렬한다. 활동마다 비어 있는(마지막 활동이 지금 활동의 시작 앞에 끝난) 아무 자원에나 배정한다. 빈 자원이 없으면 새로 하나 연다.

자원마다 마지막 활동의 마침 시각을 열쇠로 하는 최소 힙(우선순위 줄서기)을 쓰면 알고리즘이 가장 먼저 비는 자원을 효율적으로 찾는다.

---

## 4. 올바름

욕심쟁이 알고리즘은 정확히 깊이만큼의 자원을 쓴다. 증명은 두 부분이다:

1. **아래 한계.** (깊이 논증으로) 적어도 깊이만큼의 자원이 필요하다.
2. **위 한계.** 욕심쟁이 알고리즘은 깊이보다 많은 자원을 열지 않는다. 새 자원을 열 때는 이미 있는 자원이 모두 바쁘다는 뜻이고, 곧 지금 활동이 그 자원마다 적어도 활동 하나와 겹친다. 이는 깊이가 열린 자원의 수만큼 늘어났음을 뜻한다.

---

## 5. 구현

```python
"""
최소 무지를 쓴 욕심쟁이 알고리즘으로 하는 구간 나누기.

같은 자원에 놓인 두 일이 겹치지 않도록 가장 적은 자원에
일을 나누어 맡긴다.
"""

import heapq

# === 욕심쟁이 구간 나누기 ===

def interval_partitioning(
    activities: list[tuple[int, int]]
) -> list[list[tuple[int, int]]]:
    """일을 가장 적은 자원으로 나눈다.

    인수:
        activities: (시작, 마침) 짝의 목록.

    반환값:
        자원 배정의 목록. 자원마다 거기 맡긴 일의
        목록이다.
    """
    if not activities:
        return []

    # 시작하는 때로 정렬한다
    sorted_acts = sorted(activities, key=lambda x: x[0])

    # 최소 무지: (마지막 일이 마치는 때, 자원 번호)
    heap = []
    resources = []

    for start, finish in sorted_acts:
        if heap and heap[0][0] <= start:
            # 가장 일찍 마치는 자원을 다시 쓴다
            _, idx = heapq.heappop(heap)
            resources[idx].append((start, finish))
            heapq.heappush(heap, (finish, idx))
        else:
            # 새 자원을 연다
            idx = len(resources)
            resources.append([(start, finish)])
            heapq.heappush(heap, (finish, idx))

    return resources

def compute_depth(activities: list[tuple[int, int]]) -> int:
    """구간 모임의 깊이(최대 겹침)를 셈한다."""
    events = []
    for s, f in activities:
        events.append((s, 1))   # 구간이 시작한다
        events.append((f, -1))  # 구간이 끝난다
    events.sort()

    max_depth = 0
    current = 0
    for _, delta in events:
        current += delta
        max_depth = max(max_depth, current)
    return max_depth

# === 시연 ===

if __name__ == "__main__":
    activities = [
        (0, 3), (1, 4), (2, 5), (3, 7),
        (4, 6), (6, 9), (7, 8),
    ]

    resources = interval_partitioning(activities)
    depth = compute_depth(activities)

    print(f"Number of activities: {len(activities)}")
    print(f"Depth (max overlap): {depth}")
    print(f"Resources needed: {len(resources)}")
    for i, res in enumerate(resources):
        print(f"  Resource {i}: {res}")
```

**출력:**

```
Number of activities: 7
Depth (max overlap): 3
Resources needed: 3
  Resource 0: [(0, 3), (3, 7), (7, 8)]
  Resource 1: [(1, 4), (4, 6), (6, 9)]
  Resource 2: [(2, 5)]
```

때 $t = 2$에서 활동 셋 $[0,3)$, $[1,4)$, $[2,5)$이 겹친다. 알고리즘은 정확히 자원 3개를 써서 깊이라는 아래 한계와 맞아떨어진다.

---

## 6. 복잡도

| 항목 | 비용 |
|--------|:----:|
| Time   | $O(n \log n)$ |
| 공간 | $O(n)$ |

줄 세우기에 $O(n \log n)$이 든다. 활동마다 더미에 한 번 넣고 많아야 한 번 꺼내며 저마다 $O(\log n)$ 시간이 든다.

---

## 7. 응용

- **강의실 배정.** 수업 일정이 주어질 때 강의를 최소한의 방에 배정한다.
- **셈틀 머리 일정 짜기.** 일 모음에 필요한 처리기의 최소 개수를 정한다.
- **차량 길잡기.** 모든 배달 시간 창을 덮는 최소 차량 대수.

---

## 연습문제

**연습문제 1.**
구간 나누기에서 욕심쟁이 고름이 무엇인지 가려내고 왜 가장 좋은 풀이로 이어지는지 밝혀라.

??? success "연습문제 1 풀이"
    욕심쟁이 고름은 걸음마다 그 자리에서 가장 좋은 것을 고른다. Interval Partitioning에서는 이 고름이 욕심쟁이 고름 성질을 채운다. 곧 이 욕심쟁이 고름을 담은 가장 좋은 풀이가 있다. 여기에 가장 좋은 아래 짜임(욕심쟁이 고름 뒤 남은 아래 문제도 같은 전략으로 가장 좋게 풀린다)을 더하면 욕심쟁이 알고리즘이 두루 가장 좋은 풀이를 내놓는다. $\square$

---

**연습문제 2.**
구간 나누기이 가장 좋은 아래 짜임을 갖는지 증명하거나 반증하여라.

??? success "연습문제 2 풀이"
    가장 좋은 아래 짜임이란 문제의 가장 좋은 풀이가 그 아래 문제의 가장 좋은 풀이를 담는다는 뜻이다. Interval Partitioning에서는 욕심쟁이 고름을 하고 나면 남은 문제가 같은 갈래의 더 작은 사례가 된다. 아래 문제의 풀이가 가장 좋지 않다면 그것을 갈음해 온 풀이를 더 낫게 할 수 있는데, 이는 온 풀이가 가장 좋다는 것과 어긋난다. 그러므로 가장 좋은 아래 짜임이 이루어진다. $\square$

---

**연습문제 3.**
구간 나누기의 시간 복잡도는 무엇인가? 가장 값비싼 단계를 가려내어라.

??? success "연습문제 3 풀이"
    시간 복잡도는 (필요하다면) 줄 세우는 걸음과 욕심쟁이 고름 되돌이에 달렸다. 흔히 줄 세우기가 $O(n \log n)$으로 가장 크다. 욕심쟁이 되돌이는 원소마다 한 번씩 다루어 $O(n)$이다. 모두 $O(n \log n)$이다. 들임이 미리 줄 세워져 있으면 알고리즘은 $O(n)$에 돈다. $\square$

---

**연습문제 4.**
(구간 나누기에서 쓴 것이 아닌) 다른 욕심쟁이 전략은 가장 좋은 풀이를 내지 못함을 보이는 반례를 들어라.

??? success "연습문제 4 풀이"
    문제의 짜임과 맞지 않는 다른 욕심쟁이 잣대를 생각해 보자. 이 잣대는 뒤에 더 좋은 고름을 막는 원소를 고를 수 있다. 어긋나는 보기는 잘못된 욕심쟁이 잣대가 가장 좋지 않은 결과를 낼 수 있음을 보이며, 문제마다 그에 맞는 욕심쟁이 고름 성질을 반드시 증명해야 하는 까닭을 드러낸다. $\square$

## 정리하며

이 마당은 문제 서술、아래 한계: 깊이、욕심쟁이 알고리즘、올바름을 차례로 짚었다.

**참고 문헌**

- Kleinberg, J., & Tardos, E. (2006). *Algorithm Design*. Pearson. 4장: Greedy Algorithms.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), 16장: Greedy Algorithms.
