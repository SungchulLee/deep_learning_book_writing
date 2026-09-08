# 조금씩 쌓는 설계
**조금씩 쌓는 설계**(**온라인** 또는 **흐름** 알고리즘이라고도 한다)는 원소를 하나씩 다루며 걸음마다 올바른 풀이를 지녀 답을 쌓는다. 많은 효율 좋은 알고리즘의 바탕이다.

---

## 1. 조금씩 쌓는 틀

$$\text{Solution}(n) = \text{Update}(\text{Solution}(n-1), \text{element}_n)$$

걸음마다 다음 원소를 담고 답을 $O(1)$이나 $O(\log n)$ 때에 고쳐 전체로 효율 좋은 풀이를 이룬다.

---

## 2. 보기: 카데인 알고리즘(조금씩 쌓는 최대 부분 배열)

```python
def kadane(arr):
    max_ending_here = arr[0]
    max_so_far = arr[0]
    for i in range(1, len(arr)):
        # 조금씩 쌓는 결정: 지금 부분 배열을 늘릴지 새로 시작할지
        max_ending_here = max(arr[i], max_ending_here + arr[i])
        max_so_far = max(max_so_far, max_ending_here)
    return max_so_far

print(kadane([-2, 1, -3, 4, -1, 2, 1, -5, 4]))  # Output: 6
```

**출력:**

```
6
```

---

## 3. 보기: 조금씩 쌓는 볼록 껍질

앤드루의 한 방향 사슬 알고리즘은 점을 하나씩 더하며 껍질을 지닌다:

```python
def cross(O, A, B):
    return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0])

def convex_hull(points):
    points = sorted(set(points))
    if len(points) <= 1:
        return points

    # 아래 껍질을 조금씩 쌓는다
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # 위 껍질을 조금씩 쌓는다
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]

points = [(0, 0), (1, 1), (2, 2), (0, 2), (2, 0), (1, 0)]
print(convex_hull(points))
```

**출력:**

```
[(0, 0), (2, 0), (2, 2), (0, 2)]
```

---

## 4. 보기: 이어지는 가운뎃값(조금씩 쌓기)

원소가 올 때마다 가운뎃값을 얻으려 더미 둘을 지닌다:

```python
import heapq

class RunningMedian:
    def __init__(self):
        self.lo = []  # 최대 더미(음수로 만듦)
        self.hi = []  # 최소 힙

    def add(self, num):
        heapq.heappush(self.lo, -num)
        heapq.heappush(self.hi, -heapq.heappop(self.lo))
        if len(self.hi) > len(self.lo):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def median(self):
        if len(self.lo) > len(self.hi):
            return -self.lo[0]
        return (-self.lo[0] + self.hi[0]) / 2

rm = RunningMedian()
for x in [2, 1, 5, 7, 2, 0, 5]:
    rm.add(x)
    print(f"Added {x}, median = {rm.median()}")
```

**출력:**

```
Added 2, median = 2
Added 1, median = 1.5
Added 5, median = 2
Added 7, median = 3.5
Added 2, median = 2
Added 0, median = 2.0
Added 5, median = 2
```

# 참고 문헌

- Cormen, T. et al. *Introduction to Algorithms*, MIT Press, 2022.
- Skiena, S. *The Algorithm Design Manual*, Springer, 2020.

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

이 마당은 조금씩 쌓는 틀、보기: 카데인 알고리즘(조금씩 쌓는 최대 부분 배열)、보기: 조금씩 쌓는 볼록 껍질、보기: 이어지는 가운뎃값(조금씩 쌓기)을 차례로 짚었다.
