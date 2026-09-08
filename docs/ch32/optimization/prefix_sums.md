# 앞자락 합
**앞자락 합** 배열은 $O(n)$의 앞손질 뒤에 구간 합 묻기에 $O(1)$로 답하게 해 준다. 가장 근본이 되는 다듬기 재주 가운데 하나다.

---

## 1. 정의

배열 $a[0 \ldots n-1]$이 주어질 때 앞자락 합 배열 $P$은 다음과 같이 뜻매김된다:

$$P[0] = 0, \quad P[i] = \sum_{j=0}^{i-1} a[j] = P[i-1] + a[i-1]$$

구간 $[l, r)$의 원소 합은 다음과 같다:

$$\sum_{j=l}^{r-1} a[j] = P[r] - P[l]$$

---

## 2. 1차원 앞자락 합

```python
def build_prefix_sum(arr):
    prefix = [0] * (len(arr) + 1)
    for i in range(len(arr)):
        prefix[i + 1] = prefix[i] + arr[i]
    return prefix

def range_sum(prefix, l, r):
    return prefix[r + 1] - prefix[l]

arr = [3, 1, 4, 1, 5, 9, 2, 6]
prefix = build_prefix_sum(arr)
print(range_sum(prefix, 2, 5))  # 내놓기: 19  (4+1+5+9)
```

---

## 3. 2차원 앞자락 합

행렬에서 앞자락 합은 부분 행렬 합 묻기를 $O(1)$에 하게 해 준다:

$$\text{Sum}(r_1, c_1, r_2, c_2) = P[r_2+1][c_2+1] - P[r_1][c_2+1] - P[r_2+1][c_1] + P[r_1][c_1]$$

```python
def build_2d_prefix(matrix):
    rows, cols = len(matrix), len(matrix[0])
    P = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(rows):
        for j in range(cols):
            P[i+1][j+1] = matrix[i][j] + P[i][j+1] + P[i+1][j] - P[i][j]
    return P

def submatrix_sum(P, r1, c1, r2, c2):
    return P[r2+1][c2+1] - P[r1][c2+1] - P[r2+1][c1] + P[r1][c1]

matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
P = build_2d_prefix(matrix)
print(submatrix_sum(P, 1, 1, 2, 2))  # 내놓기: 28  (5+6+8+9)
```

---

## 4. 쓰임새: 합이 k인 부분 배열 세기

```python
from collections import defaultdict

def count_subarrays_sum_k(arr, k):
    count = 0
    prefix = 0
    freq = defaultdict(int)
    freq[0] = 1
    for x in arr:
        prefix += x
        count += freq[prefix - k]
        freq[prefix] += 1
    return count

print(count_subarrays_sum_k([1, 1, 1], 2))  # Output: 2
```

---

## 5. 복잡도

| 연산 | 시간 | 공간 |
|---|---|---|
| 1차원 앞자락 합 짓기 | $O(n)$ | $O(n)$ |
| 1차원 구간 합 묻기 | $O(1)$ | -- |
| 2차원 앞자락 합 짓기 | $O(nm)$ | $O(nm)$ |
| 2차원 부분 행렬 합 묻기 | $O(1)$ | -- |

# 참고 문헌

- Cormen, T. et al. *Introduction to Algorithms*, MIT Press, 2022.
- [Prefix Sum Array -- GeeksforGeeks](https://www.geeksforgeeks.org/prefix-sum-array-implementation-applications-competitive-programming/)

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

이 마당은 정의、1차원 앞자락 합、2차원 앞자락 합、쓰임새: 합이 k인 부분 배열 세기을 차례로 짚었다.
