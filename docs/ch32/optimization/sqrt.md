# 제곱근 쪼개기
**제곱근 쪼개기**는 원소 $n$개의 배열을 크기 $\sqrt{n}$의 덩이로 나누어 묻기와 고치기를 $O(\sqrt{n})$에 하게 한다. 힘으로 미는 $O(n)$과 복잡한 나무 얼개의 $O(\log n)$ 사이의 쓸모 있는 가운데 길이다.

---

## 1. 핵심 생각

배열을 크기 $B = \lfloor \sqrt{n} \rfloor$의 덩이 $\lceil n / B \rceil$개로 나눈다. 덩이마다 모은 값(합, 최소 등)을 미리 셈한다.

- **묻기** $[l, r]$: 가장자리의 조각 덩이를 다루고(저마다 $O(B)$) 가운데의 온전한 덩이를 다룬다($O(n/B)$).
- **고치기** 번호 $i$: $i$을 담은 덩이를 다시 셈한다($O(1)$이나 $O(B)$).

온 묻기 때: $O(B + n/B)$이며 $B = \sqrt{n}$에서 가장 작아져 $O(\sqrt{n})$이 된다.

---

## 2. 짜기: 점 고치기가 있는 구간 합

```python
import math

class SqrtDecomposition:
    def __init__(self, arr):
        self.arr = arr[:]
        self.n = len(arr)
        self.block_size = max(1, int(math.isqrt(self.n)))
        self.num_blocks = (self.n + self.block_size - 1) // self.block_size
        self.blocks = [0] * self.num_blocks
        for i in range(self.n):
            self.blocks[i // self.block_size] += arr[i]

    def update(self, i, val):
        self.blocks[i // self.block_size] += val - self.arr[i]
        self.arr[i] = val

    def query(self, l, r):
        total = 0
        bl = l // self.block_size
        br = r // self.block_size
        if bl == br:
            for i in range(l, r + 1):
                total += self.arr[i]
        else:
            for i in range(l, (bl + 1) * self.block_size):
                total += self.arr[i]
            for b in range(bl + 1, br):
                total += self.blocks[b]
            for i in range(br * self.block_size, r + 1):
                total += self.arr[i]
        return total

arr = [1, 3, 5, 2, 7, 6, 3, 1, 4, 8]
sd = SqrtDecomposition(arr)
print(sd.query(1, 6))   # Output: 26
sd.update(3, 10)
print(sd.query(1, 6))   # Output: 34
```

---

## 3. 복잡도

| 연산 | 시간 |
|---|---|
| 짓기 | $O(n)$ |
| 점 고치기 | $O(1)$ |
| 구간 묻기 | $O(\sqrt{n})$ |

---

## 4. 언제 쓰는가

- $O(\log n)$ 얼개(토막 나무, 이진 색인 나무)가 짜기에 너무 복잡할 때.
- 나무로 다루기 쉽지 않은 연산이 얽힌 문제일 때.
- 모 알고리즘의 쌓기 벽돌로.

# 참고 문헌

- [Sqrt Decomposition -- CP-Algorithms](https://cp-algorithms.com/data_structures/sqrt_decomposition.html)
- Halim, S. & Halim, F. *Competitive Programming 4*, 2020.

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

이 마당은 핵심 생각、짜기: 점 고치기가 있는 구간 합、복잡도、언제 쓰는가을 차례로 짚었다.
