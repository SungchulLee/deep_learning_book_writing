# 양방향 연결

단일 연결 리스트에서는 각 노드가 다음 노드에 대한 참조만 저장한다. 이 한 방향 설계 때문에 뒤로 가는 순회가 불가능하다. 어떤 노드의 앞 노드에 닿으려면 머리에서부터 리스트 전체를 다시 훑어야 한다. **이중 연결 리스트**는 각 노드에 앞 원소를 되가리키는 두 번째 포인터를 더해 이 한계를 해결한다. 이 절은 이중 연결 노드의 구조를 소개하고 양방향 연결이 어떻게 효율적인 양방향 순회를 가능하게 하는지 설명한다.

## 노드 구조

이중 연결 리스트의 노드는 세 개의 항목을 담는다.

- **data** — 노드에 저장된 값.
- **next** — 뒤따르는 노드에 대한 참조(꼬리라면 `None`).
- **prev** — 앞 노드에 대한 참조(머리라면 `None`).

```python
"""
이중 연결 리스트의 노드와 기본적인 양방향 순회.

prev와 next 포인터를 갖는 노드 구조, 리스트 만들기,
그리고 양방향 순회를 보인다.
"""


# === 노드 정의 ===

class Node:
    """이중 연결 리스트의 노드 하나."""

    def __init__(self, data, prev=None, next_node=None):
        self.data = data
        self.prev = prev
        self.next = next_node


# === 리스트 만들기 ===

def build_list(values):
    """Build a doubly linked list from a Python list of values.

    만들어진 이중 연결 리스트의 머리 노드를 돌려준다.
    """
    if not values:
        return None
    head = Node(values[0])
    current = head
    for val in values[1:]:
        new_node = Node(val, prev=current)
        current.next = new_node
        current = new_node
    return head


# === 순회 ===

def traverse_forward(head):
    """머리에서 꼬리까지 순회하며 값을 모은다."""
    result = []
    current = head
    while current:
        result.append(current.data)
        current = current.next
    return result


def traverse_backward(tail):
    """꼬리에서 머리까지 순회하며 값을 모은다."""
    result = []
    current = tail
    while current:
        result.append(current.data)
        current = current.prev
    return result


def get_tail(head):
    """head에서 시작하는 리스트의 꼬리 노드를 돌려준다."""
    current = head
    while current and current.next:
        current = current.next
    return current


# === 메인 ===

if __name__ == "__main__":
    head = build_list([10, 20, 30, 40])

    print("Forward: ", traverse_forward(head))
    print("Backward:", traverse_backward(get_tail(head)))
```

**출력:**

```
Forward:  [10, 20, 30, 40]
Backward: [40, 30, 20, 10]
```

## 포인터 사이의 관계

이웃한 모든 노드 쌍은 두 개의 불변식을 만족한다.

1. 노드 $A$이 `A.next = B`이면 노드 $B$은 반드시 `B.prev = A`이어야 한다.
2. 노드 $B$이 `B.prev = A`이면 노드 $A$은 반드시 `A.next = B`이어야 한다.

이 불변식은 연결이 **대칭**임을 뜻한다. `next`를 따라간 뒤 `prev`를 따라가면 원래 노드로 돌아오고, 그 반대도 마찬가지이다. 형식적으로, 머리도 꼬리도 아닌 임의의 내부 노드 $x$에 대해 다음이 성립한다.

$$
x.\text{next}.\text{prev} = x = x.\text{prev}.\text{next}
$$

삽입과 삭제 중에 이 대칭을 유지하는 것이 이중 연결 리스트 알고리즘의 핵심 과제이며, [삽입과 삭제](operations.md) 페이지에서 다룬다.

## 앞으로 가는 순회와 뒤로 가는 순회

각 노드가 `prev` 포인터를 지니고 있으므로 이중 연결 리스트는 추가 자료구조 없이 양쪽 방향의 순회를 지원한다.

| 순회 | 시작 | 따라갈 것 | 멈추는 때 |
|---|---|---|---|
| 앞으로 | `head` | `current = current.next` | `current is None` |
| 뒤로 | `tail` | `current = current.prev` | `current is None` |

앞으로 가는 순회와 뒤로 가는 순회 모두 모든 노드를 정확히 한 번씩 방문하므로 시간 복잡도는 $O(n)$이고 리스트 자체 외의 공간 복잡도는 $O(1)$이다.

## 단일 연결 노드와의 비교

| 특징 | 단일 연결 | 이중 연결 |
|---|---|---|
| 노드당 포인터 | 1개 (`next`) | 2개 (`prev` + `next`) |
| 노드당 메모리 | 적음 | 많음 (포인터 하나 추가) |
| 앞으로 가는 순회 | $O(n)$ | $O(n)$ |
| 뒤로 가는 순회 | 다시 시작하며 $O(n^2)$ | `prev`로 $O(n)$ |
| 주어진 노드 삭제 | $O(n)$ (앞 노드가 필요하다) | $O(1)$ (앞 노드가 `node.prev`이다) |

핵심 절충은 **메모리와 유연함** 사이에 있다. 이중 연결 노드는 포인터를 하나 더 쓰지만(64비트 시스템에서 보통 8바이트), 이 여분의 포인터가 이미 아는 노드의 $O(1)$ 삭제와 효율적인 뒤로 가는 순회를 가능하게 한다. 단일 연결 노드에서는 비싸거나 아예 불가능한 연산들이다.

## 양방향 연결을 언제 쓸 것인가

다음이 필요한 응용에서는 이중 연결 리스트가 자연스러운 선택이다.

- **뒤로 가는 반복**: 실행 취소 기록, 브라우저의 뒤로 가기 단추, 텍스트 편집기의 커서 이동.
- **이미 아는 노드의 O(1) 삭제**: LRU 캐시가 그렇다. 해시 맵이 리스트 노드에 대한 직접 참조를 담고 있어 선형 훑기 없이 지워야 한다.
- **양방향 탐색**: 정렬된 리스트에서 가운데에서 만나는 알고리즘.

이런 요구가 하나도 없다면 단일 연결 리스트가 메모리를 아끼고 구현하기도 더 간단하다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C.
  *Introduction to Algorithms* (4th ed.), Chapter 10. MIT Press.


## 연습문제

**연습문제 1.**
양방향 연결에 대해 삽입, 삭제, 탐색, 접근 연산의 시간 복잡도를 진술하라.

??? success "연습문제 1 풀이"
    복잡도는 구체적인 구현(배열 기반이냐 연결 기반이냐)에 달려 있다. 배열 기반은 접근이 $O(1)$이고 임의의 위치에서의 삽입·삭제가 $O(n)$이다. 연결 기반은 이미 아는 위치에서의 삽입·삭제가 $O(1)$이고 탐색·접근이 $O(n)$이다. 어떤 연산이 주를 이루느냐에 따라 선택이 갈린다.

---

**연습문제 2.**
원소 6개로 양방향 연결을(를) 따라가며 각 연산 후의 자료구조 상태를 보여라.

??? success "연습문제 2 풀이"
    구조에 삽입, 접근, 삭제를 차례로 수행하라. 각 단계마다 (연결 구조라면) 포인터를, (배열 기반이라면) 배열의 내용을 보이며 구조가 불변식을 어떻게 유지하는지 나타내라.

---

**연습문제 3.**
양방향 연결이(가) PyTorch의 텐서 저장과 어떻게 관련되는지 설명하라. 자료구조의 선택이 메모리 배치와 캐시 성능에 어떤 영향을 주는가?

??? success "연습문제 3 풀이"
    PyTorch 텐서는 캐시에 효율적으로 접근할 수 있도록 연속된 배열로 저장된다. 연결 구조는 autograd 그래프를 훑는 데 내부적으로 쓰인다. 이 선택은 메모리 사용량(배열에는 포인터 부담이 없다)과 접근 양상(캐시 지역성 덕분에 순차적인 배열 접근이 연결 리스트 순회보다 10~100배 빠르다)에 모두 영향을 준다.

---

**연습문제 4.**
반복문 불변식을 사용하여 양방향 연결의 주요 연산의 시간 복잡도를 증명하라.

??? success "연습문제 4 풀이"
    알고리즘의 반복문이 유지하는 불변식을 진술하라. 초기화, 유지, 종료를 증명하라. 이 불변식으로부터 반복문이 명시된 횟수 안에 끝남이 따라 나오며, 이로써 복잡도의 상계가 확립된다. $\square$