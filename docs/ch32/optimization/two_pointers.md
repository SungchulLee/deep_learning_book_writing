# 두 가리개
**두 가리개** 재주는 자료 얼개(보통 줄 세운 배열)를 지나는 번호 둘을 써서 어수룩하게 하면 $O(n^2)$이 들 문제를 $O(n)$ 때에 푼다.

## 언제 두 가리개를 쓸까

- 들임이 **줄 세워져** 있다(또는 줄 세울 수 있다).
- 조건을 만족하는 **짝**이나 **부분 배열**을 찾는다.
- 조건이 **한 방향**이다. 가리개 하나를 한쪽으로 옮기면 조건이 좋아지거나 나빠지기만 한다.

## 무늬 1: 양 끝

가리개 둘이 양 끝에서 시작해 안쪽으로 움직인다.

**문제:** 줄 세운 배열에서 합이 과녁인 수 둘을 찾아라.

```python
def two_sum_sorted(arr, target):
    left, right = 0, len(arr) - 1
    while left < right:
        s = arr[left] + arr[right]
        if s == target:
            return (left, right)
        elif s < target:
            left += 1
        else:
            right -= 1
    return None

print(two_sum_sorted([1, 2, 4, 6, 8, 10], 10))  # 내놓기: (1, 4)
```

## 무늬 2: 같은 방향(빠름/느림)

가리개 둘 다 앞에서 시작하고 하나가 더 빨리 움직인다.

**문제:** 줄 세운 배열에서 겹치는 것을 제자리에서 없애라.

```python
def remove_duplicates(arr):
    if not arr:
        return 0
    slow = 0
    for fast in range(1, len(arr)):
        if arr[fast] != arr[slow]:
            slow += 1
            arr[slow] = arr[fast]
    return slow + 1

arr = [1, 1, 2, 2, 3, 4, 4, 5]
length = remove_duplicates(arr)
print(arr[:length])  # 내놓기: [1, 2, 3, 4, 5]
```

## 무늬 3: 세 가리개(세 수 합)

원소 하나를 붙박고 나머지에 두 가리개를 써서 세 수 합을 두 수 합으로 줄인다.

```python
def three_sum(nums):
    nums.sort()
    result = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            s = nums[i] + nums[left] + nums[right]
            if s == 0:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left+1]:
                    left += 1
                while left < right and nums[right] == nums[right-1]:
                    right -= 1
                left += 1
                right -= 1
            elif s < 0:
                left += 1
            else:
                right -= 1
    return result

print(three_sum([-1, 0, 1, 2, -1, -4]))
# Output: [[-1, -1, 2], [-1, 0, 1]]
```

## 복잡도

| 무늬 | 때 | 공간 |
|---|---|---|
| 양 끝 | $O(n)$ | $O(1)$ |
| 빠름/느림 | $O(n)$ | $O(1)$ |
| 세 수 합(붙박기 + 두 가리개) | $O(n^2)$ | 여분 $O(1)$ |

# 참고 문헌

- LeetCode Two Pointers tag: [https://leetcode.com/tag/two-pointers/](https://leetcode.com/tag/two-pointers/)
- Skiena, S. *The Algorithm Design Manual*, Springer, 2020.

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
