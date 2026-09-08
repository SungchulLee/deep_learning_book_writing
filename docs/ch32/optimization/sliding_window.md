# 미끄러지는 창
**미끄러지는 창** 재주는 자료 위를 미끄러지며 늘거나 줄어드는 창(이어진 부분 배열이나 부분 문자열)을 지닌다. 많은 $O(n^2)$이나 $O(nk)$ 문제를 $O(n)$으로 줄인다.

---

## 1. 두 갈래

| 갈래 | 창 크기 | 쓰임새 |
|---|---|---|
| 붙박인 크기 | 늘 $k$ | 크기 $k$인 모든 창의 최대/최소/평균 |
| 바뀌는 크기 | 늘거나 줆 | 조건을 만족하는 가장 긴/짧은 부분 배열 |

---

## 2. 붙박인 크기의 창

**문제:** 크기가 $k$인 부분 배열의 최대 합을 찾아라.

```python
def max_sum_window(arr, k):
    window_sum = sum(arr[:k])
    max_sum = window_sum
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]
        max_sum = max(max_sum, window_sum)
    return max_sum

print(max_sum_window([1, 4, 2, 10, 2, 3, 1, 0, 20], 4))  # Output: 24
```

---

## 3. 바뀌는 크기의 창

**문제:** 글자가 겹치지 않는 가장 긴 부분 문자열의 길이를 찾아라.

```python
def longest_unique_substring(s):
    seen = {}
    left = 0
    max_len = 0
    for right, ch in enumerate(s):
        if ch in seen and seen[ch] >= left:
            left = seen[ch] + 1
        seen[ch] = right
        max_len = max(max_len, right - left + 1)
    return max_len

print(longest_unique_substring("abcabcbb"))  # Output: 3
```

---

## 4. 바뀌는 창: 합이 과녁 이상인 가장 짧은 부분 배열

```python
def min_subarray_sum(arr, target):
    left = 0
    current_sum = 0
    min_len = float('inf')
    for right in range(len(arr)):
        current_sum += arr[right]
        while current_sum >= target:
            min_len = min(min_len, right - left + 1)
            current_sum -= arr[left]
            left += 1
    return min_len if min_len != float('inf') else 0

print(min_subarray_sum([2, 3, 1, 2, 4, 3], 7))  # Output: 2
```

---

## 5. 미끄러지는 창이 통할 때

핵심 요구는 창을 늘리면 조건이 한 방향으로만 움직인다는 것이다(한 방향성). 원소를 더할 때 조건이 좋아질 수도 나빠질 수도 있어 헤아릴 수 없다면 미끄러지는 창을 그대로 쓸 수 없다.

원소마다 많아야 한 번 더하고 한 번 빼므로 모든 미끄러지는 창 알고리즘은 $O(n)$ 때를 이룬다.

# 참고 문헌

- LeetCode Sliding Window tag: [https://leetcode.com/tag/sliding-window/](https://leetcode.com/tag/sliding-window/)
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

이 마당은 두 갈래、붙박인 크기의 창、바뀌는 크기의 창、바뀌는 창: 합이 과녁 이상인 가장 짧은 부분 배열을 차례로 짚었다.
