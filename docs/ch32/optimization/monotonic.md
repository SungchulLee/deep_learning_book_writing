# 한 방향 쌓기와 줄
**한 방향 쌓기**(또는 **한 방향 줄**)는 원소를 넣을 때 줄 세운 차례를 지켜, **다음으로 큰/작은 원소**나 **미끄러지는 창의 최소/최대**가 얽힌 문제를 효율 좋게 푼다.

---

## 1. 한 방향 쌓기

한 방향 쌓기는 새 원소를 넣기 전에 한 방향 차례를 어기는 원소를 꺼낸다.

### 다음으로 큰 원소

원소마다 오른쪽에서 자기보다 큰 첫 원소를 찾는다.

```python
def next_greater_element(arr):
    n = len(arr)
    result = [-1] * n
    stack = []  # 번호. arr 값이 줄어든다
    for i in range(n):
        while stack and arr[stack[-1]] < arr[i]:
            result[stack.pop()] = arr[i]
        stack.append(i)
    return result

print(next_greater_element([4, 5, 2, 10, 8]))
# Output: [5, 10, 10, -1, -1]
```

### 막대 그림의 가장 큰 직사각형

```python
def largest_rectangle_histogram(heights):
    stack = []
    max_area = 0
    heights.append(0)  # 보초
    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)
    heights.pop()  # 보초를 없앤다
    return max_area

print(largest_rectangle_histogram([2, 1, 5, 6, 2, 3]))  # Output: 10
```

**출력:**

```
10
```

---

## 2. 한 방향 줄(두 끝 줄)

한 방향 두 끝 줄은 **미끄러지는 창의 최소/최대** 묻기를 원소마다 고루 나누어 $O(1)$에 답한다.

### 미끄러지는 창의 최대

```python
from collections import deque

def sliding_window_max(arr, k):
    dq = deque()  # 번호를 담는다. arr 값이 줄어든다
    result = []
    for i in range(len(arr)):
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        while dq and arr[dq[-1]] <= arr[i]:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            result.append(arr[dq[0]])
    return result

print(sliding_window_max([1, 3, -1, -3, 5, 3, 6, 7], 3))
# Output: [3, 3, 5, 5, 6, 7]
```

**출력:**

```
[3, 3, 5, 5, 6, 7]
```

---

## 3. 복잡도

| 연산 | 시간 | 공간 |
|---|---|---|
| 다음으로 큰 원소 | $O(n)$ | $O(n)$ |
| 막대 그림의 가장 큰 직사각형 | $O(n)$ | $O(n)$ |
| 미끄러지는 창의 최대/최소 | 모두 $O(n)$ | $O(k)$ |

원소마다 많아야 한 번 넣고 한 번 꺼내므로 연산마다 고루 나누어 $O(1)$이다.

# 참고 문헌

- LeetCode: [84. Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)
- LeetCode: [239. Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/)

---

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

## 정리하며

이 마당은 한 방향 쌓기、한 방향 줄(두 끝 줄)、복잡도을 차례로 짚었다.
