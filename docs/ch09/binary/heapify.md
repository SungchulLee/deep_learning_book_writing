# 힙으로 만들기 (아래로 내리기)

**아래로 내리기** 연산(**힙으로 만들기** 또는 **최대 힙으로 만들기**라고도 한다)은 이진 힙의 근본 되는 고치기 절차이다. 두 부분 트리는 올바른 힙인데 노드 하나가 힙 성질을 어길 때, 아래로 내리기가 그 노드를 트리 아래로 옮겨 불변식을 되살린다. 이 연산이 꺼내기와 힙 세우기와 힙 정렬을 떠받치는 일꾼이다.

## 미리 갖출 조건

노드 $i$에서의 아래로 내리기에는 정해진 조건이 필요하다.

> $\text{left}(i)$과 $\text{right}(i)$을 뿌리로 하는 부분 트리가 둘 다 올바른 힙이다. 노드 $i$ 자신만 힙 성질을 어길 수 있다.

이 조건은 주된 두 쓰임에서 언제나 만족된다.

1. **꺼내기**: 마지막 원소를 뿌리로 옮긴 뒤에도 두 부분 트리는 건드리지 않아 올바른 힙으로 남는다.
2. **힙 세우기**: 노드를 아래에서 위로 처리하므로 부모를 처리하기 전에 두 자식의 부분 트리가 이미 힙이 되어 있다.

## 알고리즘

최대 힙에서 아래로 내리기는 노드 $i$을 자식과 견주고 어긋남이 있으면 더 큰 자식과 맞바꾼다. 힙 성질이 되살아나거나 잎에 닿을 때까지 그 자식의 자리에서 이를 되풀이한다.

### 의사 코드

```
MAX-HEAPIFY(A, i, n):
    largest = i
    left = 2*i + 1
    right = 2*i + 2

    if left < n and A[left] > A[largest]:
        largest = left
    if right < n and A[right] > A[largest]:
        largest = right

    if largest != i:
        swap A[i] and A[largest]
        MAX-HEAPIFY(A, largest, n)
```

되풀이마다 비교를 두 번(노드와 왼쪽 자식, 노드와 오른쪽 자식) 하고 자리바꿈을 많아야 한 번 한다.

## 한 걸음씩 보는 예

뿌리(색인 0)는 성질을 어기지만 두 부분 트리는 올바른 최대 힙을 생각해 보자.

```
Initial state (only root violates):
          2
        /    \
      14      10
     /  \    /  \
    8    7  9    3

Step 1: Compare 2 with children 14 and 10.
        largest = 14 (index 1). Swap 2 and 14.
          14
        /    \
      2       10
     /  \    /  \
    8    7  9    3

Step 2: Compare 2 with children 8 and 7.
        largest = 8 (index 3). Swap 2 and 8.
          14
        /    \
      8       10
     /  \    /  \
    2    7  9    3

Step 3: Node 2 is at index 3 with no children below it in this example.
        2 is a leaf — stop.

Result: Heap property restored in 2 swaps.
```

## 복잡도 분석

아래로 내리기는 뿌리에서 잎까지의 경로를 많아야 하나 훑는다. 노드가 $n$개인 완전 이진 트리의 높이가 $\lfloor \log_2 n \rfloor$이므로 다음과 같다.

$$
T(n) = O(\log n)
$$

더 정확히, 층마다 알고리즘이 (왼쪽 자식과 오른쪽 자식에 대해) 비교를 꼭 2번 하고 자리바꿈을 많아야 1번 한다. 전체 비교 횟수는 많아야 $2 \lfloor \log_2 n \rfloor$이고 전체 자리바꿈 횟수는 많아야 $\lfloor \log_2 n \rfloor$이다.

!!! tip "왜 더 큰 자식과 맞바꾸는가"
    최대 힙에서는 (어긋난 아무 자식이 아니라) **더 큰** 자식과 맞바꾸는 것이 꼭 필요하다. 더 작은 자식과 맞바꾸면 그 자식이 옛 형제의 부모가 되어 새로 어긋남이 생길 수 있다. 더 큰 자식을 고르면 새 부모가 두 자식 이상으로 큼이 보장된다.

## 재귀와 되풀이

재귀 판이 더 깔끔하지만 스택 공간을 $O(\log n)$ 쓴다. 되풀이 판은 공간을 $O(1)$ 쓴다.

```python
"""
이진 힙의 아래로 내리기(힙으로 만들기) 연산.

근본 되는 힙 고치기 연산의 재귀 구현과 되풀이 구현을
함께 준다.
"""


# === 재귀로 아래로 내리기 (최대 힙) ===

def sift_down_recursive(arr, i, n):
    """색인 i에서 최대 힙 성질을 재귀적으로 되살린다.

    미리 갖출 조건: left(i)와 right(i)를 뿌리로 하는 부분 트리가
    올바른 최대 힙이다.
    """
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        sift_down_recursive(arr, largest, n)


# === 되풀이로 아래로 내리기 (최대 힙) ===

def sift_down_iterative(arr, i, n):
    """색인 i에서 최대 힙 성질을 되풀이로 되살린다.

    O(log n)의 스택 공간 대신 O(1)의 보조 공간을 쓴다.
    """
    while True:
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n and arr[left] > arr[largest]:
            largest = left
        if right < n and arr[right] > arr[largest]:
            largest = right

        if largest == i:
            break
        arr[i], arr[largest] = arr[largest], arr[i]
        i = largest


# === 최소 힙의 아래로 내리기 ===

def sift_down_min(arr, i, n):
    """색인 i에서 최소 힙 성질을 되살린다 (되풀이)."""
    while True:
        smallest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n and arr[left] < arr[smallest]:
            smallest = left
        if right < n and arr[right] < arr[smallest]:
            smallest = right

        if smallest == i:
            break
        arr[i], arr[smallest] = arr[smallest], arr[i]
        i = smallest


# === 시연 ===

if __name__ == "__main__":
    # 뿌리에서 성질이 어긋난 최대 힙
    arr1 = [2, 14, 10, 8, 7, 9, 3]
    print(f"Before sift-down (recursive): {arr1}")
    sift_down_recursive(arr1, 0, len(arr1))
    print(f"After sift-down (recursive):  {arr1}")

    # 같은 보기를 되풀이 판으로
    arr2 = [2, 14, 10, 8, 7, 9, 3]
    print(f"\nBefore sift-down (iterative): {arr2}")
    sift_down_iterative(arr2, 0, len(arr2))
    print(f"After sift-down (iterative):  {arr2}")

    # 둘이 같은 결과를 내는지 확인한다
    assert arr1 == arr2, "Mismatch between recursive and iterative"
    print("\nBoth versions produce identical results.")

    # 최소 힙 보기
    arr3 = [10, 2, 3, 8, 7, 9, 4]
    print(f"\nMin-heap before sift-down: {arr3}")
    sift_down_min(arr3, 0, len(arr3))
    print(f"Min-heap after sift-down:  {arr3}")
```

**출력:**
```
Before sift-down (recursive): [2, 14, 10, 8, 7, 9, 3]
After sift-down (recursive):  [14, 8, 10, 2, 7, 9, 3]

Before sift-down (iterative): [2, 14, 10, 8, 7, 9, 3]
After sift-down (iterative):  [14, 8, 10, 2, 7, 9, 3]

Both versions produce identical results.

Min-heap before sift-down: [10, 2, 3, 8, 7, 9, 4]
Min-heap after sift-down:  [2, 7, 3, 8, 10, 9, 4]
```

## 올바름

**고리 불변식**: 되풀이마다 그 시작에서 색인 $i$을 뿌리로 하는 부분 트리가 노드 $i$ 자신만 빼고 어디서나 최대 힙 성질을 만족한다.

- **처음**: 미리 갖출 조건이 이를 보장한다.
- **지킴**: $A[i]$이 자식 가운데 하나보다 작으면 더 큰 자식과 맞바꾼다. 맞바꾼 뒤 옛 자식 자리에는 그 두 자식 이상으로 큰 값이 있다. 새로 어긋날 수 있는 곳은 맞바꾼 원소가 내려앉은 자리뿐이다.
- **끝남**: 고리는 $i$에 자식이 없거나($i$이 잎이거나) $A[i]$이 두 자식 이상으로 클 때 끝난다. 두 경우 모두 부분 트리 전체에서 힙 성질이 성립한다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., and Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 6.2: Maintaining the heap property. MIT Press.


## 연습문제

**연습문제 1.**
힙으로 만들기(아래로 내리기)의 힙 성질을 밝히고 최솟값·최댓값 원소가 언제나 뿌리에 있음을 증명하라.

??? success "연습문제 1 풀이"
    힙 성질은 노드마다 열쇠가 자식보다 작거나 같거나(최소 힙) 크거나 같다(최대 힙)는 것이다. 뿌리에서 잎까지의 어떤 경로에서도 추이성이 성립하므로 뿌리가 모든 원소의 최솟값(또는 최댓값)이다.

---

**연습문제 2.**
배열 $[4, 7, 2, 9, 1, 5, 3]$에서 힙으로 만들기(아래로 내리기)를 따라가라. 단계마다와 그 결과로 나오는 힙을 보여라.

??? success "연습문제 2 풀이"
    이 쪽의 연산을 주어진 배열에 적용하라. 단계마다 배열과 그것이 나타내는 트리를 보여라. 비교와 자리바꿈을 짚어라.

---

**연습문제 3.**
힙으로 만들기(아래로 내리기)의 시간 복잡도를 증명하라. 그 한계는 빡빡한가?

??? success "연습문제 3 풀이"
    이 연산은 뿌리에서 잎까지 또는 잎에서 뿌리까지의 경로를 훑으며 층마다 $O(1)$의 일을 한다. 완전 이진 트리의 높이는 $\lfloor\log_2 n\rfloor$이므로 모두 $O(\log n)$이다. 이 한계는 빡빡하다. 높이 전체를 훑도록 강요하는 입력이 있다. $\square$

---

**연습문제 4.**
$k = 5$이고 어휘가 $n = 32{,}000$인 빔 탐색에서 힙으로 뽑기(원소 $n$개에서 상위 $k$개)와 정렬을 견주어라.

??? success "연습문제 4 풀이"
    정렬은 $O(n\log n) = O(32000 \times 15) \approx 48만$번의 연산이 든다. 힙(크기 $k$인 최소 힙)은 $O(n\log k) = O(32000 \times 2.3) \approx 7만 4천$번이다. 힙이 약 $6.5$배 빠르다. 아니면 `torch.topk`가 GPU에 맞추어 다듬은 부분 정렬 알고리즘을 쓴다.