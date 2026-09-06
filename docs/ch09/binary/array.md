# 배열 표현

이진 힙의 실용적인 효율은 놀랍도록 간단한 통찰에서 크게 온다. 완전 이진 트리는 포인터 없이 납작한 배열에 담을 수 있다는 것이다. (왼쪽에서 오른쪽으로 채워지는) 마지막 층만 빼고 층마다 꽉 차 있으므로, 노드마다 부모와 자식의 자리를 초등 산술로 셈할 수 있다. 이 포인터 없는 배치가 캐시 성능을 아주 좋게 하고 자식·부모 이음에 드는 기억을 0으로 만든다.

## 층 순서로 담기

완전 이진 트리는 노드를 **층 순서**로 놓아 배열에 담는다. 뿌리를 먼저, 그다음 깊이 1의 모든 노드를 왼쪽에서 오른쪽으로, 그다음 깊이 2, 이런 식이다. 그 결과는 빈틈 없는 촘촘한 배열이다.

!!! example "트리에서 배열로 잇대기"
    원소가 10개인 최대 힙을 생각해 보자.

    ```
    Tree view:                    Array view (0-indexed):

              16                  Index: 0  1  2  3  4  5  6  7  8  9
            /    \                Value: 16 14 10  8  7  9  3  2  4  1
          14      10
         /  \    /  \
        8    7  9    3
       / \  /
      2  4 1
    ```

    노드 16은 색인 0에, 그 자식 14와 10은 색인 1과 2에 있는 식이다.

## 색인 공식

배열에서의 부모-자식 관계는 층 순서 배치에서 곧바로 따라 나온다. 관례가 둘 흔하다.

### 0부터 세기 (파이썬 heapq가 쓴다)

색인 $i$의 노드에 대해 다음과 같다.

$$
\text{parent}(i) = \left\lfloor \frac{i - 1}{2} \right\rfloor
$$

$$
\text{left}(i) = 2i + 1
$$

$$
\text{right}(i) = 2i + 2
$$

뿌리는 색인 0에 있다. 전체 원소 수를 $n$이라 할 때 $2i + 1 \ge n$이면 색인 $i$의 노드는 잎이다.

### 1부터 세기 (CLRS가 쓴다)

색인 $i$의 노드에 대해 다음과 같다.

$$
\text{parent}(i) = \left\lfloor \frac{i}{2} \right\rfloor
$$

$$
\text{left}(i) = 2i
$$

$$
\text{right}(i) = 2i + 1
$$

뿌리는 색인 1에 있고 색인 0은 쓰지 않는다. 2로 곱하고 나누는 것이 왼쪽·오른쪽 비트 밀기에 해당하므로 1부터 세는 공식이 조금 더 간단하다.

??? tip "비트 밀기로 다듬기"
    1부터 세는 방식에서 부모, 왼쪽 자식, 오른쪽 자식 연산은 비트 명령 하나로 줄어든다.

    - `parent(i) = i >> 1`
    - `left(i) = i << 1`
    - `right(i) = (i << 1) | 1`

    이는 요즘 CPU가 한 주기에 해내는 일정 시간 연산이다.

## 완전 이진 트리에 배열이 통하는 까닭

배열 표현이 통하는 것은 오로지 완전 이진 트리의 층 순서 순회에 "구멍"이 없기 때문이다. 아무 이진 트리나 쓰면 공간을 엄청나게 버릴 수 있다. 노드 $n$개가 깊이 $n-1$까지 한쪽으로 기운 트리는 크기 $2^n - 1$의 배열이 필요하고 대부분의 칸이 빈다.

완전 이진 트리에서는 배열이 언제나 촘촘하다.

| 성질 | 값 |
|----------|-------|
| 층 $k$의 노드 수 | $2^k$ (마지막 층만 다를 수 있다) |
| 높이 $h$인 완전 트리의 전체 노드 수 | $2^h$과 $2^{h+1} - 1$ 사이 |
| 필요한 배열 크기 | 꼭 $n$ (버리는 칸 없음) |
| 포인터에 드는 기억 | 0 |

## 힙 배열 오가기

다음 구현은 0부터 세는 관례로 부모와 자식을 오가는 법을 보인다.

```python
"""
이진 힙의 배열 표현.

완전 이진 트리의 부모-자식 관계가 납작한 배열에서 간단한
색인 산술로 잇대어짐을 보인다.
"""


# === 색인 오가기 (0부터 세기) ===

def parent(i):
    """노드 i의 부모 색인을 돌려준다."""
    return (i - 1) // 2


def left_child(i):
    """노드 i의 왼쪽 자식의 색인을 돌려준다."""
    return 2 * i + 1


def right_child(i):
    """노드 i의 오른쪽 자식의 색인을 돌려준다."""
    return 2 * i + 2


def is_leaf(i, n):
    """크기 n인 힙에서 노드 i가 잎인지 살핀다."""
    return left_child(i) >= n


# === 트리 그려 보기 ===

def print_heap_tree(arr):
    """부모-자식 관계를 보이는 트리 짜임으로 배열을 찍는다."""
    n = len(arr)
    if n == 0:
        print("Empty heap")
        return

    print(f"Array: {arr}")
    print(f"Size:  {n}\n")

    for i in range(n):
        l = left_child(i)
        r = right_child(i)
        children = []
        if l < n:
            children.append(f"left={arr[l]} (idx {l})")
        if r < n:
            children.append(f"right={arr[r]} (idx {r})")

        parent_info = ""
        if i > 0:
            p = parent(i)
            parent_info = f"  parent={arr[p]} (idx {p})"

        child_info = ", ".join(children) if children else "leaf"
        print(f"  idx {i}: value={arr[i]}{parent_info}  -> {child_info}")


# === 시연 ===

if __name__ == "__main__":
    heap = [16, 14, 10, 8, 7, 9, 3, 2, 4, 1]
    print_heap_tree(heap)

    print("\n--- Index formula verification ---")
    for i in range(len(heap)):
        if i > 0:
            p = parent(i)
            assert heap[p] >= heap[i], f"Max-heap violated at index {i}"
    print("All parent-child relationships satisfy the max-heap property.")
```

**출력:**
```
Array: [16, 14, 10, 8, 7, 9, 3, 2, 4, 1]
Size:  10

  idx 0: value=16  -> left=14 (idx 1), right=10 (idx 2)
  idx 1: value=14  parent=16 (idx 0)  -> left=8 (idx 3), right=7 (idx 4)
  idx 2: value=10  parent=16 (idx 0)  -> left=9 (idx 5), right=3 (idx 6)
  idx 3: value=8  parent=14 (idx 1)  -> left=2 (idx 7), right=4 (idx 8)
  idx 4: value=7  parent=14 (idx 1)  -> left=1 (idx 9)
  idx 5: value=9  parent=10 (idx 2)  -> leaf
  idx 6: value=3  parent=10 (idx 2)  -> leaf
  idx 7: value=2  parent=8 (idx 3)  -> leaf
  idx 8: value=4  parent=8 (idx 3)  -> leaf
  idx 9: value=1  parent=7 (idx 4)  -> leaf

--- Index formula verification ---
All parent-child relationships satisfy the max-heap property.
```

## 캐시 효율

트리를 이어진 배열로 담으면 요즘 하드웨어에서 성능이 크게 좋아진다. 프로세서가 색인 $i$의 원소에 닿을 때 캐시 줄이 이웃한 원소도 함께 실어 온다. 노드의 자식이 색인 $2i+1$과 $2i+2$에 있어 기억 속에서 가까우므로 부모에서 자식으로 갈 때 캐시에 자주 맞는다. 이와 달리 포인터에 바탕한 트리는 노드를 기억 곳곳에 흩어 놓아 순회 중에 캐시를 자주 놓친다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., and Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 6: Heapsort. MIT Press.


## 연습문제

**연습문제 1.**
배열 표현의 힙 성질을 밝히고 최솟값·최댓값 원소가 언제나 뿌리에 있음을 증명하라.

??? success "연습문제 1 풀이"
    힙 성질은 노드마다 열쇠가 자식보다 작거나 같거나(최소 힙) 크거나 같다(최대 힙)는 것이다. 뿌리에서 잎까지의 어떤 경로에서도 추이성이 성립하므로 뿌리가 모든 원소의 최솟값(또는 최댓값)이다.

---

**연습문제 2.**
배열 $[4, 7, 2, 9, 1, 5, 3]$에서 배열 표현을 따라가라. 단계마다와 그 결과로 나오는 힙을 보여라.

??? success "연습문제 2 풀이"
    이 쪽의 연산을 주어진 배열에 적용하라. 단계마다 배열과 그것이 나타내는 트리를 보여라. 비교와 자리바꿈을 짚어라.

---

**연습문제 3.**
배열 표현의 시간 복잡도를 증명하라. 그 한계는 빡빡한가?

??? success "연습문제 3 풀이"
    이 연산은 뿌리에서 잎까지 또는 잎에서 뿌리까지의 경로를 훑으며 층마다 $O(1)$의 일을 한다. 완전 이진 트리의 높이는 $\lfloor\log_2 n\rfloor$이므로 모두 $O(\log n)$이다. 이 한계는 빡빡하다. 높이 전체를 훑도록 강요하는 입력이 있다. $\square$

---

**연습문제 4.**
$k = 5$이고 어휘가 $n = 32{,}000$인 빔 탐색에서 힙으로 뽑기(원소 $n$개에서 상위 $k$개)와 정렬을 견주어라.

??? success "연습문제 4 풀이"
    정렬은 $O(n\log n) = O(32000 \times 15) \approx 48만$번의 연산이 든다. 힙(크기 $k$인 최소 힙)은 $O(n\log k) = O(32000 \times 2.3) \approx 7만 4천$번이다. 힙이 약 $6.5$배 빠르다. 아니면 `torch.topk`가 GPU에 맞추어 다듬은 부분 정렬 알고리즘을 쓴다.