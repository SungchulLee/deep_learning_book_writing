# 정렬된 리스트 k개 합치기

정렬된 리스트 $k$개를 정렬된 하나로 합치는 일은 외부 정렬, 데이터베이스 질의 처리, 데이터가 미리 정렬된 덩어리로 들어오는 분산 시스템에서 근본 되는 연산이다. $k$개 리스트의 머리에서 최솟값을 되풀이해 고르는 순진한 방법은 비교가 $O(nk)$번 든다. 최소 힙은 지금의 후보 $k$개만 지켜 이를 $O(n \log k)$으로 줄인다. 여기서 $n$은 모든 리스트를 통틀어 원소의 총 개수이다.

---

## 1. 문제 서술

원소를 모두 합쳐 $n$개인 정렬된 리스트 $k$개가 주어졌을 때, $n$개를 모두 담은 정렬된 리스트 하나를 만들어라.

---

## 2. 힙을 쓰는 알고리즘

리스트마다 원소를 하나씩 담은 크기 $k$의 최소 힙을 지키는 것이 요령이다. 단계마다 힙에서 최솟값(아직 처리하지 않은 것 가운데 전체에서 가장 작은 원소)을 꺼내 출력에 덧붙이고, 그 최솟값을 낸 리스트에서 다음 원소를 넣는다.

### 알고리즘

1. $k$개 리스트마다 첫 원소로 최소 힙을 시작하되, 원소마다 어느 리스트의 어느 자리에서 왔는지를 좇는 곁정보를 함께 담는다.
2. 힙이 비어 있지 않은 동안 다음을 되풀이한다.
    - 힙에서 최솟값 원소를 꺼내 출력에 덧붙인다.
    - 그 리스트에 원소가 더 있으면 다음 원소를 힙에 넣는다.
3. 출력 리스트를 돌려준다.

### 왜 O(n log k)인가

전체 $n$개 원소마다 힙에 꼭 한 번 들어가고 꼭 한 번 나온다. 힙에는 원소가 많아야 $k$개 있으므로 힙 연산마다 $O(\log k)$이 든다. 따라서 전체 비용은 다음과 같다.

$$
T(n, k) = n \cdot O(\log k) = O(n \log k)
$$

---

## 3. 여러 방식 견주기

| 방법 | 시간 | 공간 |
|----------|------|-------|
| 단계마다 $k$개 머리를 모두 견주기 | $O(nk)$ | $O(1)$ |
| 크기 $k$의 최소 힙 | $O(n \log k)$ | $O(k)$ |
| 나누어 정복하며 둘씩 합치기 | $O(n \log k)$ | $O(n)$ |

힙 방법과 나누어 정복하는 방법은 점근 시간이 같지만, 힙 방법은 (출력에 드는 $O(n)$ 말고는) 여분 공간이 $O(k)$뿐이고 원소를 흘려보내듯 처리한다.

---

## 4. 구현

```python
"""
최소 힙으로 정렬된 리스트 k개 합치기.

O(n log k)의 힙 기반 k방향 합치기 알고리즘을 보인다.
"""

import heapq

# === 맨바닥부터 구현하기 ===

def merge_k_sorted(lists):
    """최소 힙으로 정렬된 리스트 k개를 정렬된 하나로 합친다.

    힙 항목마다 (값, 리스트 색인, 원소 색인)으로 두어 같은 값의 순서를
    정해진 대로 가르고 원소마다의 출처를 좇는다.

    시간: O(n log k), 공간: 힙에 O(k).
    """
    result = []
    heap = []

    # 리스트마다 첫 원소로 힙을 시작한다
    for i, lst in enumerate(lists):
        if lst:
            # (값, 리스트 색인, 원소 색인)
            heapq.heappush(heap, (lst[0], i, 0))

    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        result.append(val)

        # 같은 리스트의 다음 원소를 넣는다
        next_idx = elem_idx + 1
        if next_idx < len(lists[list_idx]):
            next_val = lists[list_idx][next_idx]
            heapq.heappush(heap, (next_val, list_idx, next_idx))

    return result

# === heapq.merge 쓰기 ===

def merge_k_heapq(lists):
    """파이썬의 heapq.merge로 정렬된 반복 가능 객체 k개를 합친다."""
    return list(heapq.merge(*lists))

# === 시연 ===

if __name__ == "__main__":
    lists = [
        [1, 4, 7, 10],
        [2, 5, 8, 11],
        [3, 6, 9, 12],
    ]

    print("Input lists:")
    for i, lst in enumerate(lists):
        print(f"  List {i}: {lst}")

    result = merge_k_sorted(lists)
    print(f"\nMerged (manual): {result}")

    result2 = merge_k_heapq(lists)
    print(f"Merged (heapq):  {result2}")

    # 길이가 다른 리스트
    print("\n--- Unequal Length Lists ---")
    lists2 = [
        [1, 3, 5, 7, 9, 11],
        [2, 4],
        [6, 8, 10],
        [],
        [0, 12, 13, 14],
    ]

    for i, lst in enumerate(lists2):
        print(f"  List {i}: {lst}")

    result3 = merge_k_sorted(lists2)
    print(f"\nMerged: {result3}")

    # 정렬되었는지 확인한다
    assert result3 == sorted(result3), "Result is not sorted!"
    print("Correctness verified.")
```

**출력:**
```
Input lists:
  List 0: [1, 4, 7, 10]
  List 1: [2, 5, 8, 11]
  List 2: [3, 6, 9, 12]

Merged (manual): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
Merged (heapq):  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

--- Unequal Length Lists ---
  List 0: [1, 3, 5, 7, 9, 11]
  List 1: [2, 4]
  List 2: [6, 8, 10]
  List 3: []
  List 4: [0, 12, 13, 14]

Merged: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
Correctness verified.
```

---

## 5. 연결 리스트 판

정렬된 입력 $k$개가 (면접에서 흔한 문제인) 연결 리스트일 때도 같은 알고리즘을 쓴다. 힙 항목마다 리스트의 지금 노드를 담는다. 최소 노드를 꺼낸 뒤 `node.next`가 있으면 넣는다.

??? example "연결 리스트의 k방향 합치기"
    ```python
    class ListNode:
        def __init__(self, val=0, next=None):
            self.val = val
            self.next = next

    def merge_k_linked(lists):
        heap = []
        for i, node in enumerate(lists):
            if node:
                heapq.heappush(heap, (node.val, i, node))

        dummy = ListNode(0)
        current = dummy

        while heap:
            val, idx, node = heapq.heappop(heap)
            current.next = ListNode(val)
            current = current.next
            if node.next:
                heapq.heappush(heap, (node.next.val, idx, node.next))

        return dummy.next
    ```

---

## 6. 응용

| 응용 | k방향 합치기를 어떻게 쓰는가 |
|------------|------------------------|
| 외부 정렬 | 디스크의 정렬된 구간을 합친다 |
| 맵리듀스 | 매퍼 작업의 정렬된 출력을 합친다 |
| 데이터베이스 조인 | 정렬된 색인 훑기를 합친다 |
| 로그 모으기 | 서버 $k$대의 시각이 찍힌 로그를 합친다 |
| 토너먼트 트리 | 하드웨어의 선택 회로망 |

---

## 연습문제

**연습문제 1.**
정렬된 리스트 k개 합치기의 힙 성질을 밝히고 최솟값·최댓값 원소가 언제나 뿌리에 있음을 증명하라.

??? success "연습문제 1 풀이"
    힙 성질은 노드마다 열쇠가 자식보다 작거나 같거나(최소 힙) 크거나 같다(최대 힙)는 것이다. 뿌리에서 잎까지의 어떤 경로에서도 추이성이 성립하므로 뿌리가 모든 원소의 최솟값(또는 최댓값)이다.

---

**연습문제 2.**
배열 $[4, 7, 2, 9, 1, 5, 3]$에서 정렬된 리스트 k개 합치기를 따라가라. 단계마다와 그 결과로 나오는 힙을 보여라.

??? success "연습문제 2 풀이"
    이 쪽의 연산을 주어진 배열에 적용하라. 단계마다 배열과 그것이 나타내는 트리를 보여라. 비교와 자리바꿈을 짚어라.

---

**연습문제 3.**
정렬된 리스트 k개 합치기의 시간 복잡도를 증명하라. 그 한계는 빡빡한가?

??? success "연습문제 3 풀이"
    이 연산은 뿌리에서 잎까지 또는 잎에서 뿌리까지의 경로를 훑으며 층마다 $O(1)$의 일을 한다. 완전 이진 트리의 높이는 $\lfloor\log_2 n\rfloor$이므로 모두 $O(\log n)$이다. 이 한계는 빡빡하다. 높이 전체를 훑도록 강요하는 입력이 있다. $\square$

---

**연습문제 4.**
$k = 5$이고 어휘가 $n = 32{,}000$인 빔 탐색에서 힙으로 뽑기(원소 $n$개에서 상위 $k$개)와 정렬을 견주어라.

??? success "연습문제 4 풀이"
    정렬은 $O(n\log n) = O(32000 \times 15) \approx 48만$번의 연산이 든다. 힙(크기 $k$인 최소 힙)은 $O(n\log k) = O(32000 \times 2.3) \approx 7만 4천$번이다. 힙이 약 $6.5$배 빠르다. 아니면 `torch.topk`가 GPU에 맞추어 다듬은 부분 정렬 알고리즘을 쓴다.

## 정리하며

이 마당은 문제 서술、힙을 쓰는 알고리즘、여러 방식 견주기、구현을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., and Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 6.5 and Problem 6-2. MIT Press.
- 파이썬 문서: [heapq.merge](https://docs.python.org/3/library/heapq.html#heapq.merge)
