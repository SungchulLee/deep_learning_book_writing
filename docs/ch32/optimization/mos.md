# 모 알고리즘
**모 알고리즘**은 가리개 움직임을 가장 적게 하는 영리한 차례로 묻기를 다루어 오프라인 구간 묻기에 $O((n + q) \sqrt{n})$ 때에 답한다. 미끄러지는 구간에 자료 얼개를 지닐 때 특히 쓸모 있다.

## 핵심 생각

1. 모든 묻기를 $(\lfloor l / \sqrt{n} \rfloor, r)$으로 줄 세운다. 왼쪽 끝점의 덩이를 먼저, 그다음 오른쪽 끝점으로.
2. 지금 구간 $[cur\_l, cur\_r]$을 지니고 원소를 하나씩 늘리거나 줄여 묻기 구간에 이른다.
3. 모든 묻기를 통틀은 가리개 움직임은 $O((n + q)\sqrt{n})$이다.

## 구현

**문제:** 정수 배열이 주어질 때 묻기 $q$개에 답하여라. $[l, r]$에 서로 다른 원소가 몇 개인가?

```python
import math
from collections import defaultdict

def mos_algorithm(arr, queries):
    n = len(arr)
    block = max(1, int(math.isqrt(n)))
    q = len(queries)

    # 묻기를 (l의 덩이, r)으로 줄 세운다
    indexed_queries = sorted(
        enumerate(queries),
        key=lambda x: (x[1][0] // block,
                       x[1][1] if (x[1][0] // block) % 2 == 0 else -x[1][1])
    )

    freq = defaultdict(int)
    distinct = 0
    cur_l, cur_r = 0, -1
    answers = [0] * q

    def add(idx):
        nonlocal distinct
        freq[arr[idx]] += 1
        if freq[arr[idx]] == 1:
            distinct += 1

    def remove(idx):
        nonlocal distinct
        freq[arr[idx]] -= 1
        if freq[arr[idx]] == 0:
            distinct -= 1

    for qi, (l, r) in indexed_queries:
        while cur_l > l:
            cur_l -= 1
            add(cur_l)
        while cur_r < r:
            cur_r += 1
            add(cur_r)
        while cur_l < l:
            remove(cur_l)
            cur_l += 1
        while cur_r > r:
            remove(cur_r)
            cur_r -= 1
        answers[qi] = distinct

    return answers

arr = [1, 2, 1, 3, 2, 1, 4]
queries = [(0, 3), (1, 5), (2, 6), (0, 6)]
print(mos_algorithm(arr, queries))
# Output: [3, 3, 4, 4]
```

## 복잡도

| 부분 | 복잡도 |
|---|---|
| 묻기 줄 세우기 | $O(q \log q)$ |
| 오른쪽 가리개 움직임 | $O(n \sqrt{n})$ |
| 왼쪽 가리개 움직임 | $O(n \sqrt{n})$ |
| 모두 | $O((n + q)\sqrt{n})$ |

## 언제 쓰는가

- 묻기가 **오프라인**이다(모두 미리 알려져 있다).
- 구간에서 원소 하나를 더하거나 빼는 것이 $O(1)$이나 $O(\log n)$이다.
- 그 연산에 효율 좋은 온라인 자료 얼개가 없다.

# 참고 문헌

- [Mo's Algorithm -- CP-Algorithms](https://cp-algorithms.com/data_structures/sqrt_decomposition.html)
- Halim, S. & Halim, F. *Competitive Programming 4*, 2020.

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
