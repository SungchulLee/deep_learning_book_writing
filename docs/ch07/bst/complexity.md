# 복잡도

이진 탐색 트리의 핵심 연산인 [찾기](search.md), [삽입](insertion.md), [삭제](deletion.md), [최솟값·최댓값](min_max.md), [후속자·선행자](successor.md)는 모두 뿌리에서 잎까지, 또는 뿌리에서 어떤 노드까지의 경로를 따라간다. 곧 연산의 실행 시간이 트리의 높이 $h$으로 묶인다. 핵심 물음은 이것이다. $h$은 노드 수 $n$과 어떤 관계인가?

---

## 1. 연산은 $O(h)$이다

이진 탐색 트리의 연산은 들르는 층마다 일정한 양의 일을 하고 되풀이 또는 재귀 호출마다 한 층씩 내려간다. 가장 긴 경로의 변이 $h$개이므로 모든 연산은 $O(h)$ 시간에 끝난다.

| 연산 | 시간 복잡도 |
|---|---|
| 찾기 | $O(h)$ |
| 삽입 | $O(h)$ |
| 삭제 | $O(h)$ |
| 최솟값 / 최댓값 | $O(h)$ |
| 후속자 / 선행자 | $O(h)$ |
| 중위 순회 | $\Theta(n)$ |

순회는 예외이다. 모든 노드를 꼭 한 번씩 들르므로 트리의 모양과 상관없이 $\Theta(n)$이 걸린다.

---

## 2. 최선의 경우: 균형 잡힌 트리

트리가 균형 잡혀 있으면(노드마다 왼쪽과 오른쪽 부분 트리의 크기 차이가 많아야 상수 배) 높이는 다음과 같다.

$$
h = \Theta(\log n)
$$

완벽하게 균형 잡힌 이진 탐색 트리에서는 정렬된 배열의 이진 탐색처럼 비교할 때마다 남은 노드의 절반쯤이 걸러진다. 모든 연산이 $O(\log n)$ 시간에 끝난다.

**예**: 노드가 $n = 15$개인 균형 잡힌 이진 탐색 트리의 높이는 $h = 3$이므로 찾기는 많아야 노드 4개를 들른다.

---

## 3. 최악의 경우: 치우친 트리

열쇠를 정렬된 (또는 거꾸로 정렬된) 순서로 넣으면 새 노드마다 앞 노드의 오른쪽 (또는 왼쪽) 자식이 되어 연결 리스트처럼 보이는 **치우친** 트리가 된다.

```
Insert 1, 2, 3, 4, 5:

1
 \
  2
   \
    3
     \
      4
       \
        5
```

이때 $h = n - 1$이고 모든 연산이 $O(n)$으로 나빠진다.

$$
h = n - 1 \quad \Rightarrow \quad \text{all operations are } O(n)
$$

---

## 4. 평균의 경우: 무작위 삽입

서로 다른 열쇠 $n$개를 무작위 순서로(모든 순열이 똑같이 그럴듯하게) 넣으면 만들어진 이진 탐색 트리의 기대 높이는 다음과 같다.

$$
E[h] = O(\log n)
$$

더 정확히 말하면 열쇠 $n$개로 무작위로 만든 이진 탐색 트리의 기대 높이는 많아야 $3 \ln n \approx 4.33 \log_2 n$이다. [무작위로 만든 이진 탐색 트리](random.md)의 분석으로 CLRS가 증명한 이 결과는 입력이 무작위일 때 치우친 최악의 경우가 잘 일어나지 않음을 보여 준다.

!!! note "기댓값과 보장"
    무작위로 만든 이진 탐색 트리의 기대 높이가 $O(\log n)$이라는 것은 삽입 순서가 고르게 무작위인 순열이라는 가정에 기댄다. 실제 데이터는 무작위가 아닐 때가 많다. 정렬된 입력, 거의 정렬된 입력, 적대적인 입력은 모두 치우친 트리를 만들 수 있다. 균형 이진 탐색 트리 계열(AVL 트리, 레드-블랙 트리, B-트리)은 삽입 순서와 상관없이 **최악의 경우에도** 높이가 $O(\log n)$임을 보장한다.

---

## 5. 요약표

| 트리의 모양 | 높이 $h$ | 연산 시간 | 언제 생기는가 |
|---|---|---|---|
| 완벽하게 균형 잡힘 | $\lfloor \log_2 n \rfloor$ | $O(\log n)$ | 신경 써서 만들 때 |
| 무작위 | 기댓값 $O(\log n)$ | 기댓값 $O(\log n)$ | 무작위 삽입 순서 |
| 치우침 (선형) | $n - 1$ | $O(n)$ | 정렬된 삽입 순서 |

---

## 6. 공간 복잡도

노드가 $n$개인 이진 탐색 트리는 모양과 상관없이 $\Theta(n)$의 공간을 쓴다. 모양은 시간 복잡도에 영향을 주지만 공간에는 주지 않는다.

재귀로 하는 연산은 $O(h)$의 스택 공간을 쓴다. 균형 잡힌 트리에서는 $O(\log n)$, 치우친 트리에서는 $O(n)$이다. 찾기와 삽입을 되풀이로 구현하면 보조 공간이 $O(1)$이다.

```python
"""
이진 탐색 트리의 복잡도 시연.

정렬된 순서와 무작위 순서로 이진 탐색 트리를 만들어, 트리의 모양이
높이와 연산 시간에 어떤 영향을 주는지 보인다.
"""

import random
import time

# === 노드 정의 ===

class Node:
    """이진 탐색 트리의 노드."""

    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

# === 이진 탐색 트리 연산 ===

def insert(root, key):
    """이진 탐색 트리에 열쇠를 넣는다 (스택 넘침을 피하려고 반복으로 한다)."""
    new_node = Node(key)
    if root is None:
        return new_node
    current = root
    while True:
        if key <= current.key:
            if current.left is None:
                current.left = new_node
                return root
            current = current.left
        else:
            if current.right is None:
                current.right = new_node
                return root
            current = current.right

def height(node):
    """트리의 높이를 돌려준다 (너비 우선 탐색 기반의 반복)."""
    if node is None:
        return -1
    from collections import deque
    queue = deque([(node, 0)])
    max_depth = 0
    while queue:
        current, depth = queue.popleft()
        max_depth = max(max_depth, depth)
        if current.left:
            queue.append((current.left, depth + 1))
        if current.right:
            queue.append((current.right, depth + 1))
    return max_depth

def search(root, key):
    """열쇠를 찾고 비교한 횟수를 돌려준다."""
    comparisons = 0
    current = root
    while current is not None:
        comparisons += 1
        if key == current.key:
            return comparisons
        elif key < current.key:
            current = current.left
        else:
            current = current.right
    return comparisons

# === 메인 ===

if __name__ == "__main__":
    n = 1000

    # 정렬된 순서로 삽입 -> 치우친 트리
    sorted_root = None
    for k in range(1, n + 1):
        sorted_root = insert(sorted_root, k)
    sorted_height = height(sorted_root)

    # 무작위 삽입 -> (기대적으로) 균형 잡힌 트리
    keys = list(range(1, n + 1))
    random.seed(42)
    random.shuffle(keys)
    random_root = None
    for k in keys:
        random_root = insert(random_root, k)
    random_height = height(random_root)

    print(f"n = {n}")
    print(f"  Sorted insertion height:  {sorted_height} (worst case: {n - 1})")
    print(f"  Random insertion height:  {random_height} (ideal: {n.bit_length() - 1})")
    print()

    # 탐색 시간 견주기
    target = n // 2
    sorted_comps = search(sorted_root, target)
    random_comps = search(random_root, target)
    print(f"  Search for {target}:")
    print(f"    Sorted tree comparisons: {sorted_comps}")
    print(f"    Random tree comparisons: {random_comps}")
```

**출력:**
```
n = 1000
  Sorted insertion height:  999 (worst case: 999)
  Random insertion height:  22 (ideal: 9)

  Search for 500:
    Sorted tree comparisons: 500
    Random tree comparisons: 12
```

---

## 연습문제

**연습문제 1.**
복잡도에 대해 트리의 높이 $h$으로 시간 복잡도를 밝혀라. $h = O(\log n)$이 되는 것은 언제인가?

??? success "연습문제 1 풀이"
    이 연산은 $O(h)$ 시간이 걸린다. 균형 잡힌 트리에서는 $h = O(\log n)$이다(무작위 삽입 순서에서는 기댓값이 $O(\log n)$이고, 균형 이진 탐색 트리 계열은 이를 보장한다). 한쪽으로 치우친 트리에서는 $h = O(n)$이다(정렬된 순서로 넣은 경우).

---

**연습문제 2.**
열쇠 5, 3, 7, 1, 4, 6, 8을 차례로 넣어 만든 이진 탐색 트리에 복잡도을(를) 적용하라. 단계마다 보여라.

??? success "연습문제 2 풀이"
    차례로 넣어 트리를 만든 뒤 이 쪽의 연산을 적용한다. 단계마다 트리의 모습을 보이고 들르거나 고친 노드를 표시한다.

---

**연습문제 3.**
이진 탐색 트리의 높이를 $h$이라 할 때 복잡도이(가) $O(h)$ 시간에 끝남을 증명하라.

??? success "연습문제 3 풀이"
    이 연산은 뿌리에서 잎으로 가는 경로(또는 그 일부)를 따라가며 노드마다 $O(1)$의 일을 한다. 경로의 길이는 많아야 $h$이다. 따라서 모두 $O(h)$이다. $\square$

---

**연습문제 4.**
이진 탐색 트리의 연산은 딥러닝에서 정렬된 데이터를 다루는 일(예를 들어 빔 탐색에서 상위 $k$개의 점수를 유지하는 일)과 어떤 관계가 있는가?

??? success "연습문제 4 풀이"
    빔 탐색은 가장 좋은 가설 $k$개를 유지하므로 새 후보를 넣고 가장 나쁜 것을 빼는 일을 효율적으로 해야 한다. 이진 탐색 트리는 둘 다 $O(\log k)$에 해낸다. 실제로는 더 간단하고 상수가 작은 힙을 즐겨 쓰지만, 이진 탐색 트리는 범위 검색이나 순위 같은 더 풍부한 질의를 지원한다.

## 정리하며

이 마당은 연산은 $O(h)$이다、최선의 경우: 균형 잡힌 트리、최악의 경우: 치우친 트리、평균의 경우: 무작위 삽입을 차례로 짚었다.

**참고 문헌**

- [Introduction to Algorithms (CLRS), 12.4절](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
