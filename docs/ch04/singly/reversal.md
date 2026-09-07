# 뒤집기

단일 연결 리스트를 뒤집는 일, 즉 마지막 노드를 새 머리로 삼고 모든 포인터의 방향을 뒤집는 일은 연결 리스트의 가장 기본적인 연산 중 하나이다. (회문인지 확인하려고 리스트의 뒤쪽 절반을 뒤집는 것처럼) 여러 알고리즘의 하위 절차로 쓰이며 면접 문제의 흔한 구성 요소이기도 하다. 단일 연결 리스트의 노드는 앞쪽만 가리키므로, 뒤집으려면 남은 리스트를 놓치지 않으면서 각 `next` 포인터를 한 노드씩 조심스럽게 뒤쪽으로 돌려놓아야 한다.

## 반복적 뒤집기

반복 방식은 포인터 세 개로 리스트를 훑으며 한 번의 통과로 각 연결을 뒤집는다.

**알고리즘:**

1. `prev = None`, `current = head`로 초기화한다.
2. `current`가 `None`이 아닌 동안 다음을 반복한다.
    - `next_node = current.next`를 저장한다(리스트의 나머지를 잃지 않기 위해서다).
    - 연결을 뒤집는다: `current.next = prev`.
    - 나아간다: `prev = current`, `current = next_node`.
3. `prev`를 새 머리로 반환한다.

??? example "단계별 추적"

    `1 -> 2 -> 3 -> 4`을 뒤집는다.

    | 단계 | prev | current | next_node | 동작                    |
    |------|------|---------|-----------|---------------------------|
    | 0    | 없음 | 1       | 2         | 1.next = None             |
    | 1    | 1    | 2       | 3         | 2.next = 1                |
    | 2    | 2    | 3       | 4         | 3.next = 2                |
    | 3    | 3    | 4       | None      | 4.next = 3                |

    3단계 후: `prev = Node(4)`, `current = None`이다. `Node(4)`를 새 머리로 반환한다.

    결과: `4 -> 3 -> 2 -> 1`.

**시간 복잡도:** $O(n)$이다. 각 노드를 정확히 한 번씩 방문한다.

**공간 복잡도:** $O(1)$이다. 포인터 변수 세 개만 쓴다.

## 재귀적 뒤집기

재귀 방식은 리스트의 나머지를 먼저 뒤집은 뒤 현재 노드의 포인터를 고친다.

**알고리즘:**

1. 기저 사례: `head`가 `None`이거나 `head.next`가 `None`이면 `head`를 반환한다.
2. `head.next`에서 시작하는 부분 리스트를 재귀적으로 뒤집는다: `new_head = reverse(head.next)`.
3. `head.next.next = head`로 둔다(`head` 다음 노드가 이제 `head`를 되가리킨다).
4. `head.next = None`으로 둔다(옛 앞쪽 연결을 끊는다).
5. `new_head`를 반환한다.

??? example "재귀 추적"

    `1 -> 2 -> 3`을 뒤집는다.

    ```
    reverse(1)
      reverse(2)
        reverse(3)          # 기저 사례: Node(3)을 돌려준다
        3.next = None → set 3.next = 2, 2.next = None
        return Node(3)      # 리스트: 3 -> 2
      2.next = None → set 2.next = 1, 1.next = None
      return Node(3)        # 리스트: 3 -> 2 -> 1
    ```

**시간 복잡도:** $O(n)$이다. 노드마다 재귀 호출이 한 번씩 있다.

**공간 복잡도:** $O(n)$이다. 재귀 스택에 프레임이 $n$개 쌓인다.

!!! warning "스택 넘침 위험"

    재귀 방식은 $O(n)$의 스택 공간을 쓴다. 노드가 수천 개인 리스트에서는 파이썬에서 스택 넘침이 날 수 있다(기본 재귀 한도가 1000이다). 큰 리스트에는 반복 방식이 낫다.

## 부분 리스트 뒤집기

유용한 변형으로 (1에서 시작하는) 위치 $m$과 $n$ 사이의 노드만 뒤집고 나머지는 그대로 두는 것이 있다.

**알고리즘:**

1. 위치 $m - 1$의 노드(뒤집을 구간의 앞 노드)까지 순회한다.
2. 반복 기법으로 위치 $m$에서 $n$까지의 노드 $n - m + 1$개를 뒤집는다.
3. 뒤집힌 구간을 리스트의 나머지와 다시 잇는다.

**시간 복잡도:** $O(n)$이다. 리스트를 한 번 훑는다.

## 구현

```python
"""단일 연결 리스트의 뒤집기 연산."""


# === 노드 클래스 ===
class Node:
    """단일 연결 리스트의 노드 하나."""

    def __init__(self, data, next_node=None):
        self.data = data
        self.next = next_node

    def __repr__(self):
        return f"Node({self.data})"


# === 도우미 함수 ===
def build_list(values):
    """반복 가능한 객체에서 연결 리스트를 만들고 머리 노드를 돌려준다."""
    head = None
    for val in reversed(values):
        head = Node(val, head)
    return head


def to_list(head):
    """모든 노드의 값을 파이썬 리스트로 모은다."""
    result = []
    current = head
    while current is not None:
        result.append(current.data)
        current = current.next
    return result


# === 반복적 뒤집기 ===
def reverse_iterative(head):
    """Reverse a linked list iteratively.

    시간: O(n), 공간: O(1).
    """
    prev = None
    current = head
    while current is not None:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    return prev


# === 재귀적 뒤집기 ===
def reverse_recursive(head):
    """Reverse a linked list recursively.

    시간: O(n), 공간: 재귀 스택 때문에 O(n).
    """
    if head is None or head.next is None:
        return head
    new_head = reverse_recursive(head.next)
    head.next.next = head
    head.next = None
    return new_head


# === 부분 리스트 뒤집기 ===
def reverse_between(head, m, n):
    """Reverse nodes from position m to n (1-indexed).

    시간: O(n), 공간: O(1).
    """
    if m == n:
        return head

    dummy = Node(0, head)
    prev = dummy
    for _ in range(m - 1):
        prev = prev.next

    # 노드 n - m + 1개 뒤집기
    current = prev.next
    for _ in range(n - m):
        next_node = current.next
        current.next = next_node.next
        next_node.next = prev.next
        prev.next = next_node

    return dummy.next


# === 시연 ===
if __name__ == "__main__":
    # 반복적 뒤집기
    head = build_list([1, 2, 3, 4, 5])
    print(f"Original:    {to_list(head)}")
    head = reverse_iterative(head)
    print(f"Reversed:    {to_list(head)}")

    # 재귀적 뒤집기
    head = build_list([10, 20, 30, 40])
    head = reverse_recursive(head)
    print(f"\nRecursive:   {to_list(head)}")

    # 부분 뒤집기 (위치 2부터 4까지)
    head = build_list([1, 2, 3, 4, 5])
    head = reverse_between(head, 2, 4)
    print(f"\nPartial [2,4]: {to_list(head)}")
```

**출력:**
```
Original:    [1, 2, 3, 4, 5]
Reversed:    [5, 4, 3, 2, 1]

Recursive:   [40, 30, 20, 10]

Partial [2,4]: [1, 4, 3, 2, 5]
```

## 복잡도 요약

| 변형               | 시간   | 공간  |
|-----------------------|--------|--------|
| 반복적 뒤집기    | $O(n)$ | $O(1)$ |
| 재귀적 뒤집기    | $O(n)$ | $O(n)$ |
| 부분 리스트 [m,n] 뒤집기 | $O(n)$ | $O(1)$ |

## 참고 문헌

- [Introduction to Algorithms (CLRS), Chapter 10](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)


## 연습문제

**연습문제 1.**
뒤집기에 대해 삽입, 삭제, 탐색, 접근 연산의 시간 복잡도를 진술하라.

??? success "연습문제 1 풀이"
    복잡도는 구체적인 구현(배열 기반이냐 연결 기반이냐)에 달려 있다. 배열 기반은 접근이 $O(1)$이고 임의의 위치에서의 삽입·삭제가 $O(n)$이다. 연결 기반은 이미 아는 위치에서의 삽입·삭제가 $O(1)$이고 탐색·접근이 $O(n)$이다. 어떤 연산이 주를 이루느냐에 따라 선택이 갈린다.

---

**연습문제 2.**
원소 6개로 뒤집기을(를) 따라가며 각 연산 후의 자료구조 상태를 보여라.

??? success "연습문제 2 풀이"
    구조에 삽입, 접근, 삭제를 차례로 수행하라. 각 단계마다 (연결 구조라면) 포인터를, (배열 기반이라면) 배열의 내용을 보이며 구조가 불변식을 어떻게 유지하는지 나타내라.

---

**연습문제 3.**
뒤집기이(가) PyTorch의 텐서 저장과 어떻게 관련되는지 설명하라. 자료구조의 선택이 메모리 배치와 캐시 성능에 어떤 영향을 주는가?

??? success "연습문제 3 풀이"
    PyTorch 텐서는 캐시에 효율적으로 접근할 수 있도록 연속된 배열로 저장된다. 연결 구조는 autograd 그래프를 훑는 데 내부적으로 쓰인다. 이 선택은 메모리 사용량(배열에는 포인터 부담이 없다)과 접근 양상(캐시 지역성 덕분에 순차적인 배열 접근이 연결 리스트 순회보다 10~100배 빠르다)에 모두 영향을 준다.

---

**연습문제 4.**
반복문 불변식을 사용하여 뒤집기의 주요 연산의 시간 복잡도를 증명하라.

??? success "연습문제 4 풀이"
    알고리즘의 반복문이 유지하는 불변식을 진술하라. 초기화, 유지, 종료를 증명하라. 이 불변식으로부터 반복문이 명시된 횟수 안에 끝남이 따라 나오며, 이로써 복잡도의 상계가 확립된다. $\square$