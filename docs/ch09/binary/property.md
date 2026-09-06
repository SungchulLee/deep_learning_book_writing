# 힙 성질

힙을 효율적으로 쓰는 모든 길은 구조 불변식 하나, 곧 **힙 성질**에 매여 있다. 노드마다 재귀적으로 적용되는 이 순서 제약이 뿌리에 언제나 끝값(최솟값이나 최댓값) 원소를 두게 한다. 원소 쌍마다 순서가 매겨진 온전히 정렬된 배열과 달리 힙은 부모-자식 관계만 지키게 하여, 전역 순서를 내주고 로그 시간에 넣고 지우는 능력을 얻는다.

## 완전 이진 트리

이진 힙은 **완전 이진 트리** 위에 세워진다. 마지막 층만 빼고 층마다 꽉 차 있고 마지막 층은 왼쪽에서 오른쪽으로 채워지는 이진 트리이다. 이 모양 제약이 노드가 $n$개인 힙의 높이를 다음으로 보장하고

$$
h = \lfloor \log_2 n \rfloor
$$

공간을 하나도 버리지 않는 효율적인 배열 저장을 가능케 한다.

## 최대 힙 성질

뿌리가 아닌 노드 $i$마다 $i$의 값이 그 부모의 값 이하이면 이진 트리가 **최대 힙 성질**을 만족한다.

$$
A[\text{parent}(i)] \ge A[i]
$$

이는 어떤 부분 트리에서도 가장 큰 원소가 언제나 그 부분 트리의 뿌리에 있다는 뜻이다. 귀납법으로 힙 전체에서 가장 큰 원소가 뿌리에 있다.

!!! example "최대 힙 보기"
    최대 힙으로 담긴 배열 `[16, 14, 10, 8, 7, 9, 3, 2, 4, 1]`을 생각해 보자.

    ```
              16
            /    \
          14      10
         /  \    /  \
        8    7  9    3
       / \  /
      2  4 1
    ```

    부모마다 자식 이상이다. $16 \ge 14$, $16 \ge 10$, $14 \ge 8$, $14 \ge 7$ 하는 식이다.

## 최소 힙 성질

뿌리가 아닌 노드 $i$마다 $i$의 값이 그 부모의 값 이상이면 이진 트리가 **최소 힙 성질**을 만족한다.

$$
A[\text{parent}(i)] \le A[i]
$$

가장 작은 원소가 언제나 뿌리에 있다. 최소 힙은 파이썬 `heapq` 모듈의 기본이며, 우선순위가 가장 높은 항목의 열쇠가 가장 작은 우선순위 큐에 자연스러운 선택이다.

!!! example "최소 힙 보기"
    최소 힙으로 담긴 배열 `[1, 2, 3, 8, 7, 9, 10, 14, 4, 16]`을 생각해 보자.

    ```
              1
            /    \
          2        3
         /  \    /   \
        8    7  9    10
       / \  /
     14  4 16
    ```

    부모마다 자식 이하이다. $1 \le 2$, $1 \le 3$, $2 \le 8$, $2 \le 7$ 하는 식이다.

## 부분 순서로 넉넉한 까닭

온전히 정렬된 배열은 최솟값이나 최댓값에 $O(1)$으로 닿게 해 주지만, 정렬된 순서를 지키느라 삽입과 삭제에 $O(n)$이 든다. 힙은 순서 요구를 느슨하게 한다. 형제 관계가 아니라 부모-자식 관계만 지키게 한다. 이 부분 순서가 효율적인 우선순위 큐 연산을 받치기에 꼭 알맞다.

다음 표는 모두 힙 성질에 기대는 핵심 힙 연산의 복잡도를 간추린다.

| 연산 | 설명 | 시간 복잡도 |
|-----------|-------------|-----------------|
| 삽입 (위로 올리기) | 원소를 더하고 위로 가며 힙 성질을 되살린다 | $O(\log n)$ |
| 뿌리 꺼내기 (아래로 내리기) | 뿌리를 없애고 아래로 가며 힙 성질을 되살린다 | $O(\log n)$ |
| 엿보기 | 없애지 않고 뿌리 원소를 돌려준다 | $O(1)$ |
| 힙 세우기 | 정렬되지 않은 배열을 힙으로 바꾼다 | $O(n)$ |
| 힙 정렬 | 힙을 세운 뒤 모든 원소를 꺼낸다 | $O(n \log n)$ |

## 힙 성질 확인하기

간단한 재귀 검사로 배열이 최대 힙 성질을 만족하는지 확인한다. 알고리즘은 노드마다 자식과 견주고 부분 트리로 재귀한다.

```python
"""
힙 성질 확인하기.

배열이 최대 힙 성질이나 최소 힙 성질을 만족하는지
살피는 함수를 준다.
"""


# === 최대 힙 성질 확인 ===

def is_max_heap(arr, i=0):
    """색인 i에서 시작해 arr이 최대 힙 성질을 만족하는지 살핀다."""
    n = len(arr)
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[i]:
        return False
    if right < n and arr[right] > arr[i]:
        return False

    left_ok = is_max_heap(arr, left) if left < n else True
    right_ok = is_max_heap(arr, right) if right < n else True
    return left_ok and right_ok


# === 최소 힙 성질 확인 ===

def is_min_heap(arr, i=0):
    """색인 i에서 시작해 arr이 최소 힙 성질을 만족하는지 살핀다."""
    n = len(arr)
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] < arr[i]:
        return False
    if right < n and arr[right] < arr[i]:
        return False

    left_ok = is_min_heap(arr, left) if left < n else True
    right_ok = is_min_heap(arr, right) if right < n else True
    return left_ok and right_ok


# === 시연 ===

if __name__ == "__main__":
    max_heap = [16, 14, 10, 8, 7, 9, 3, 2, 4, 1]
    print(f"Array: {max_heap}")
    print(f"Is max-heap: {is_max_heap(max_heap)}")
    print(f"Is min-heap: {is_min_heap(max_heap)}")

    min_heap = [1, 2, 3, 8, 7, 9, 10, 14, 4, 16]
    print(f"\nArray: {min_heap}")
    print(f"Is max-heap: {is_max_heap(min_heap)}")
    print(f"Is min-heap: {is_min_heap(min_heap)}")

    not_heap = [3, 16, 10, 8, 7, 9, 1, 2, 4, 14]
    print(f"\nArray: {not_heap}")
    print(f"Is max-heap: {is_max_heap(not_heap)}")
    print(f"Is min-heap: {is_min_heap(not_heap)}")
```

**출력:**
```
Array: [16, 14, 10, 8, 7, 9, 3, 2, 4, 1]
Is max-heap: True
Is min-heap: False

Array: [1, 2, 3, 8, 7, 9, 10, 14, 4, 16]
Is max-heap: False
Is min-heap: True

Array: [3, 16, 10, 8, 7, 9, 1, 2, 4, 14]
Is max-heap: False
Is min-heap: False
```

이 확인은 노드마다 꼭 한 번씩 들르므로 $O(n)$ 시간에 돈다.

## 파이썬 heapq 모듈

파이썬 표준 라이브러리는 `heapq` 모듈로 최소 힙 구현을 준다. 이 모듈은 보통 리스트에서 바로 돌며 최소 힙 성질을 불변식으로 지킨다.

```python
"""
파이썬 heapq 모듈 보이기.

표준 라이브러리의 최소 힙으로 기본 힙 연산을 보인다.
"""

from heapq import heapify, heappop, heappush


# === 기본 힙 연산 ===

if __name__ == "__main__":
    # 정렬되지 않은 리스트로 시작한다
    lst = [4, 5, 1, 2, 3]
    print(f"Original list: {lst}")

    # O(n)에 최소 힙으로 바꾼다
    heapify(lst)
    print(f"After heapify:  {lst}")

    # O(log n)에 최솟값 원소를 꺼낸다
    smallest = heappop(lst)
    print(f"Popped {smallest}, heap is now: {lst}")

    # O(log n)에 새 원소를 넣는다
    heappush(lst, 0)
    print(f"Pushed 0, heap is now:  {lst}")

    # 모든 원소를 정렬된 순서로 꺼낸다
    sorted_result = []
    while lst:
        sorted_result.append(heappop(lst))
    print(f"Sorted extraction: {sorted_result}")
```

**출력:**
```
Original list: [4, 5, 1, 2, 3]
After heapify:  [1, 2, 4, 5, 3]
Popped 1, heap is now: [2, 3, 4, 5]
Pushed 0, heap is now:  [0, 3, 4, 5, 2]
Sorted extraction: [0, 2, 3, 4, 5]
```

??? tip "heapq로 최대 힙 흉내 내기"
    `heapq`가 최소 힙만 주므로, 넣을 때 값의 부호를 뒤집고 꺼낼 때 다시 뒤집는 기법을 흔히 쓴다.

    ```python
    import heapq

    max_heap = []
    for val in [4, 5, 1, 2, 3]:
        heapq.heappush(max_heap, -val)

    # 최댓값을 꺼낸다
    largest = -heapq.heappop(max_heap)  # 5를 돌려준다
    ```

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., and Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 6: Heapsort. MIT Press.
- 파이썬 문서: [heapq — 힙 큐 알고리즘](https://docs.python.org/3/library/heapq.html)


## 연습문제

**연습문제 1.**
힙 성질의 힙 성질을 밝히고 최솟값·최댓값 원소가 언제나 뿌리에 있음을 증명하라.

??? success "연습문제 1 풀이"
    힙 성질은 노드마다 열쇠가 자식보다 작거나 같거나(최소 힙) 크거나 같다(최대 힙)는 것이다. 뿌리에서 잎까지의 어떤 경로에서도 추이성이 성립하므로 뿌리가 모든 원소의 최솟값(또는 최댓값)이다.

---

**연습문제 2.**
배열 $[4, 7, 2, 9, 1, 5, 3]$에서 힙 성질을 따라가라. 단계마다와 그 결과로 나오는 힙을 보여라.

??? success "연습문제 2 풀이"
    이 쪽의 연산을 주어진 배열에 적용하라. 단계마다 배열과 그것이 나타내는 트리를 보여라. 비교와 자리바꿈을 짚어라.

---

**연습문제 3.**
힙 성질의 시간 복잡도를 증명하라. 그 한계는 빡빡한가?

??? success "연습문제 3 풀이"
    이 연산은 뿌리에서 잎까지 또는 잎에서 뿌리까지의 경로를 훑으며 층마다 $O(1)$의 일을 한다. 완전 이진 트리의 높이는 $\lfloor\log_2 n\rfloor$이므로 모두 $O(\log n)$이다. 이 한계는 빡빡하다. 높이 전체를 훑도록 강요하는 입력이 있다. $\square$

---

**연습문제 4.**
$k = 5$이고 어휘가 $n = 32{,}000$인 빔 탐색에서 힙으로 뽑기(원소 $n$개에서 상위 $k$개)와 정렬을 견주어라.

??? success "연습문제 4 풀이"
    정렬은 $O(n\log n) = O(32000 \times 15) \approx 48만$번의 연산이 든다. 힙(크기 $k$인 최소 힙)은 $O(n\log k) = O(32000 \times 2.3) \approx 7만 4천$번이다. 힙이 약 $6.5$배 빠르다. 아니면 `torch.topk`가 GPU에 맞추어 다듬은 부분 정렬 알고리즘을 쓴다.