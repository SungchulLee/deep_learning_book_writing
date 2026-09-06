# 거꾸로 풀기
**거꾸로 풀기**는 바라는 결과에서 시작해 첫 상태 쪽으로 따져 가는 것이다. 짓기 문제, 놀이 이론, 앞 방향의 고름이 너무 많은 문제에 특히 힘세다.

## 언제 거꾸로 풀까

- 끝 상태는 또렷한데 시작 상태의 가짓수가 많을 때.
- 앞으로 흉내 내면 지수로 갈라지지만 거꾸로 하면 모일 때.
- 연산을 되돌리는 일이 얽힐 때.

## 보기: 1로 줄이기

**문제:** 수 $n$이 주어질 때 1로 줄이는 최소 연산 횟수를 찾아라. 연산: 1 빼기, (짝수면) 2로 나누기, (3으로 나누어떨어지면) 3으로 나누기.

```python
from collections import deque

def min_operations_to_one(n):
    if n == 1:
        return 0
    visited = {n}
    queue = deque([(n, 0)])
    while queue:
        val, steps = queue.popleft()
        candidates = [val - 1]
        if val % 2 == 0:
            candidates.append(val // 2)
        if val % 3 == 0:
            candidates.append(val // 3)
        for next_val in candidates:
            if next_val == 1:
                return steps + 1
            if next_val not in visited and next_val > 0:
                visited.add(next_val)
                queue.append((next_val, steps + 1))
    return -1

print(min_operations_to_one(10))  # 내놓기: 3 (10->9->3->1)
```

## 보기: 과녁 배열 짓기

**문제:** 과녁 배열이 주어질 때 최대 원소를 (최대 - 나머지의 합)으로 되풀이해 바꾸어 `[1, 1, ..., 1]`에서 지을 수 있는지 판정하여라.

**핵심 통찰:** 거꾸로 푼다. 과녁이 $x > y$인 `[x, y]`이면 앞 상태는 `[x - y, y]`였다. 이는 사실상 최대 공약수 알고리즘이다.

```python
import heapq

def is_possible(target):
    total = sum(target)
    heap = [-x for x in target]  # 음수로 만든 최대 더미
    heapq.heapify(heap)
    while True:
        largest = -heapq.heappop(heap)
        rest = total - largest
        if largest == 1 or rest == 1:
            return True
        if rest == 0 or largest <= rest:
            return False
        largest %= rest  # 거꾸로 풀기: 여러 걸음을 한꺼번에 되돌린다
        if largest == 0:
            largest = rest
        total = rest + largest
        heapq.heappush(heap, -largest)

print(is_possible([9, 3, 5]))  # 내놓기: True
```

## 밝힘에서 거꾸로 풀기

거꾸로 풀기는 밝힘 설계에도 도움이 된다. 밝히고 싶은 것에서 시작해 거기에 이르는 데 충분한 조건을 정한다.

# 참고 문헌

- Polya, G. *How to Solve It*, Princeton University Press, 1945.
- LeetCode 1354번: Construct Target Array With Multiple Sums.

## 연습문제

**연습문제 1.**
이 마디의 주제와 딸린 단순한 마르코프 결정 과정을 생각하여라. 상태 3개와 움직임 2개의 작은 보기에서 관련 양을 손으로 셈하여라.

??? success "연습문제 1 풀이"
    상태 $S = \{s_1, s_2, s_3\}$과 움직임 $A = \{a_1, a_2\}$을 뜻매김한다. 옮김 확률과 보상을 매긴다. 상태-움직임 짝마다 기대 즉시 보상과 옮김 분포를 셈한다. 이 마디의 뜻매김과 식으로 바라는 양을 셈한다. 상태 자리가 작아 정확히 셈할 수 있어 추상 적기가 구체 숫자로 어떻게 옮겨지는지 보여 준다. $\square$

---

**연습문제 2.**
이 마디에서 다룬 핵심 성질이나 모임 결과를 밝혀라. 여김을 또렷이 적고 어느 것이 꼭 필요한지 가려내어라.

??? success "연습문제 2 풀이"
    밝힘은 그 연산자에 오므리는 옮김 정리를 써서 따라온다. 깎기 인수가 $\gamma < 1$인 유한 마르코프 결정 과정을 여기면 그 연산자는 상한 노름에서 $\gamma$오므리기다. 바나흐 고정점 정리에 따라 되풀이해 쓰면 $k$이 되풀이 횟수일 때 빠르기 $O(\gamma^k)$으로 하나뿐인 고정점에 모인다. 유한하다는 여김이 보상이 가둬짐을 보장하고 깎기 인수 $\gamma < 1$이 오므리기 성질에 꼭 필요하다. $\square$

---

**연습문제 3.**
이 마디에서 밝힌 알고리즘이나 셈을 단순한 격자 세상에 대해 파이썬으로 짜라. $\epsilon = 0.01$ 안으로 모이는 데 필요한 되풀이 횟수를 알려라.

??? success "연습문제 3 풀이"
    모서리에 마침 상태가 있고 고른 아무 방침을 쓰는 $4 \times 4$ 격자 세상이 여느 시험 사례가 된다. 짜기는 모든 상태의 가장 큰 바뀜이 $\epsilon$ 아래로 떨어질 때까지 고침 규칙을 되풀이한다. 깎기 인수에 따라 보통 50~200번 되풀이하면 모인다. 핵심 짜기 세부는 맞춘 고침보다 빨리 모이도록 제자리 고침(가우스-자이델 방식)을 쓰는 것이다. $\square$

---

**연습문제 4.**
이 마디에서 밝힌 길에 본디 있는 근본 한계나 맞바꿈을 다루어라. 뒤 장의 더 나아간 방법이 이 한계를 어떻게 넘는가?

??? success "연습문제 4 풀이"
    표로 하는 길은 모든 상태(어쩌면 움직임까지)를 늘어놓아야 하는데 이어지거나 차원이 높은 상태 자리에서는 될 일이 아니다. 차원의 저주는 상태 변수의 수에 따라 상태 수가 지수로 늘어남을 뜻한다. 함수 어림(33~34장)은 그 함수를 신경망으로 잡을 두어 나타내고 닮은 상태에 걸쳐 넓혀 이를 넘는다. 다만 새 어려움이 생긴다. 모임이 더는 보장되지 않으며 함수 어림, 띄워 올리기, 벗어난 방침 익히기의 죽음의 삼각이 발산을 일으킬 수 있다. $\square$
