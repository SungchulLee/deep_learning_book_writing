# 삭제

단일 연결 리스트에서 노드를 지우는 일은 개념적으로 단순하다. 지울 노드를 건너뛰도록 포인터를 고치면 된다. 다만 세부는 어느 노드를 지우느냐에 달려 있다. 주된 어려움은 단일 연결이라는 구조에서 온다. 노드를 없애려면 그 **앞 노드**에 닿을 수 있어야 앞 노드의 `next` 포인터를 돌려놓을 수 있다. 단일 연결 리스트에는 뒤로 가는 참조가 없으므로 앞 노드를 찾으려면 머리에서부터 순회해야 한다. 그래서 머리에서의 삭제는 $O(1)$이지만 다른 곳에서의 삭제는 최악의 경우 $O(n)$이다.

## 머리에서의 삭제

첫 노드를 없애는 것이 가장 단순한 경우이다. 머리 포인터를 두 번째 노드로 돌리면 옛 머리에는 닿을 수 없게 된다(파이썬에서는 쓰레기 수집된다).

**알고리즘:**

1. 리스트가 비어 있으면 오류를 낸다.
2. (반환하기 위해) 머리의 데이터에 대한 참조를 저장한다.
3. `head = head.next`로 둔다.

**시간 복잡도:** $O(1)$이다. 순회가 필요 없다.

## 꼬리에서의 삭제

마지막 노드를 없애려면 리스트 전체를 훑어 끝에서 두 번째 노드를 찾은 뒤 그 `next`를 `None`으로 두어야 한다.

**알고리즘:**

1. 리스트가 비어 있으면 오류를 낸다.
2. 리스트에 노드가 하나뿐이면 `head = None`으로 둔다.
3. 그렇지 않으면 `next.next`가 `None`인 노드(끝에서 두 번째 노드)까지 순회한다.
4. 그 노드의 `next`를 `None`으로 둔다.

**시간 복잡도:** $O(n)$이다. 노드 $n - 1$개를 지나가야 한다.

## 값으로 삭제하기

특정 값을 담은 첫 노드를 지우려면 앞 노드를 추적하며 리스트를 순회한다.

**알고리즘:**

1. 머리가 목표 값을 담고 있으면 머리 삭제를 수행한다.
2. 그렇지 않으면 `current`보다 한 걸음 뒤에 있는 `prev` 포인터를 유지하며 머리에서부터 순회한다.
3. `current.data == target`이면 `prev.next = current.next`로 둔다.
4. 일치하는 노드가 없으면 리스트는 그대로이다.

**시간 복잡도:** $O(n)$이다. 리스트 전체를 훑어야 할 수 있다.

## 위치 k에서의 삭제

0에서 시작하는 인덱스 $k$의 노드를 지우려면 머리에서 노드 $k$개를 지나간다.

**알고리즘:**

1. $k = 0$이면 머리 삭제를 수행한다.
2. 그렇지 않으면 $k - 1$걸음을 지나 앞 노드에 이른다.
3. `predecessor.next = predecessor.next.next`로 둔다.

**시간 복잡도:** $O(k)$이다. 노드 $k$개를 지나간다.

!!! warning "앞 노드 문제"

    단일 연결 리스트에서 (앞 노드가 아니라) 그 노드에 대한 참조만 있을 때 노드를 지우는 것은 까다롭다. 흔히 쓰는 우회책은 다음 노드의 데이터를 현재 노드로 복사한 뒤 다음 노드를 지우는 것이다. 이 요령은 복사해 올 뒤 노드가 없는 꼬리 노드에서는 통하지 않는다. (다음 절에서 다루는) 이중 연결 리스트는 뒤로 가는 포인터로 이를 우아하게 해결한다.

## 구현

```python
"""단일 연결 리스트의 삭제 연산."""


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


# === 삭제 연산 ===
def delete_head(head):
    """첫 노드를 지운다. (새 머리, 지운 값)을 돌려준다."""
    if head is None:
        raise IndexError("Cannot delete from an empty list")
    return head.next, head.data


def delete_tail(head):
    """마지막 노드를 지운다. (새 머리, 지운 값)을 돌려준다."""
    if head is None:
        raise IndexError("Cannot delete from an empty list")
    if head.next is None:
        return None, head.data
    current = head
    while current.next.next is not None:
        current = current.next
    deleted_value = current.next.data
    current.next = None
    return head, deleted_value


def delete_by_value(head, target):
    """data == target인 첫 노드를 지운다. 새 머리를 돌려준다."""
    if head is None:
        return None
    if head.data == target:
        return head.next
    current = head
    while current.next is not None:
        if current.next.data == target:
            current.next = current.next.next
            return head
        current = current.next
    return head  # 목표를 찾지 못함


def delete_at_position(head, k):
    """0부터 세는 인덱스 k의 노드를 지운다. 새 머리를 돌려준다."""
    if head is None:
        raise IndexError("Cannot delete from an empty list")
    if k == 0:
        return head.next
    current = head
    for _ in range(k - 1):
        if current.next is None:
            raise IndexError(f"Position {k} out of range")
        current = current.next
    if current.next is None:
        raise IndexError(f"Position {k} out of range")
    current.next = current.next.next
    return head


# === 시연 ===
if __name__ == "__main__":
    # 리스트 만들기: 10 -> 20 -> 30 -> 40 -> 50
    head = build_list([10, 20, 30, 40, 50])
    print(f"Original:          {to_list(head)}")

    # 머리 삭제
    head, val = delete_head(head)
    print(f"After delete head: {to_list(head)}  (removed {val})")

    # 꼬리 삭제
    head, val = delete_tail(head)
    print(f"After delete tail: {to_list(head)}  (removed {val})")

    # 값으로 삭제 (30)
    head = delete_by_value(head, 30)
    print(f"After delete 30:   {to_list(head)}")

    # 다시 만들고 위치 2에서 삭제
    head = build_list([1, 2, 3, 4, 5])
    head = delete_at_position(head, 2)
    print(f"Delete at pos 2:   {to_list(head)}")
```

**출력:**
```
Original:          [10, 20, 30, 40, 50]
After delete head: [20, 30, 40, 50]  (removed 10)
After delete tail: [20, 30, 40]  (removed 50)
After delete 30:   [20, 40]
Delete at pos 2:   [1, 2, 4, 5]
```

## 복잡도 요약

| 연산            | 시간       | 공간  |
|----------------------|------------|--------|
| 머리 삭제          | $O(1)$     | $O(1)$ |
| 꼬리 삭제          | $O(n)$     | $O(1)$ |
| 값으로 삭제      | $O(n)$     | $O(1)$ |
| 위치 $k$에서 삭제 | $O(k)$  | $O(1)$ |

모든 삭제 연산은 새 노드를 할당하지 않고 포인터만 고치므로 보조 공간이 $O(1)$이다.

## 참고 문헌

- [Introduction to Algorithms (CLRS), Chapter 10](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)


## 연습문제

**연습문제 1.**
삭제에 대해 삽입, 삭제, 탐색, 접근 연산의 시간 복잡도를 진술하라.

??? success "연습문제 1 풀이"
    복잡도는 구체적인 구현(배열 기반이냐 연결 기반이냐)에 달려 있다. 배열 기반은 접근이 $O(1)$이고 임의의 위치에서의 삽입·삭제가 $O(n)$이다. 연결 기반은 이미 아는 위치에서의 삽입·삭제가 $O(1)$이고 탐색·접근이 $O(n)$이다. 어떤 연산이 주를 이루느냐에 따라 선택이 갈린다.

---

**연습문제 2.**
원소 6개로 삭제을(를) 따라가며 각 연산 후의 자료구조 상태를 보여라.

??? success "연습문제 2 풀이"
    구조에 삽입, 접근, 삭제를 차례로 수행하라. 각 단계마다 (연결 구조라면) 포인터를, (배열 기반이라면) 배열의 내용을 보이며 구조가 불변식을 어떻게 유지하는지 나타내라.

---

**연습문제 3.**
삭제이(가) PyTorch의 텐서 저장과 어떻게 관련되는지 설명하라. 자료구조의 선택이 메모리 배치와 캐시 성능에 어떤 영향을 주는가?

??? success "연습문제 3 풀이"
    PyTorch 텐서는 캐시에 효율적으로 접근할 수 있도록 연속된 배열로 저장된다. 연결 구조는 autograd 그래프를 훑는 데 내부적으로 쓰인다. 이 선택은 메모리 사용량(배열에는 포인터 부담이 없다)과 접근 양상(캐시 지역성 덕분에 순차적인 배열 접근이 연결 리스트 순회보다 10~100배 빠르다)에 모두 영향을 준다.

---

**연습문제 4.**
반복문 불변식을 사용하여 삭제의 주요 연산의 시간 복잡도를 증명하라.

??? success "연습문제 4 풀이"
    알고리즘의 반복문이 유지하는 불변식을 진술하라. 초기화, 유지, 종료를 증명하라. 이 불변식으로부터 반복문이 명시된 횟수 안에 끝남이 따라 나오며, 이로써 복잡도의 상계가 확립된다. $\square$