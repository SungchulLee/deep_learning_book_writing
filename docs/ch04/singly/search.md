# 탐색

연결 리스트를 탐색한다는 것은 어떤 기준을 만족하는 노드를 찾는 일이며, 보통은 목표 값과 일치하는 노드를 찾거나 주어진 위치의 노드를 찾는 것이다. 정렬된 데이터에서 $O(\log n)$ 이진 탐색이 가능한 배열과 달리, 연결 리스트에는 임의 접근이 없다. $k$번째 원소에 닿으려면 머리에서 포인터를 $k$번 따라가야 한다. 그래서 연결 리스트에서 두루 쓸 수 있는 방법은 **선형 탐색**뿐이고 최악의 경우 비용은 $O(n)$이다. 이 한계를 이해해 두는 것은 실무에서 배열과 연결 리스트 중 무엇을 고를지 정하는 데 중요하다.

---

## 1. 값으로 탐색하기

가장 흔한 탐색 연산은 머리에서부터 리스트를 순회하며 각 노드의 데이터를 목표 값과 견주는 것이다.

**알고리즘:**

1. 머리 노드에서 시작한다.
2. 현재 노드의 데이터가 목표와 같으면 그 노드를 반환한다.
3. 다음 노드로 옮긴다.
4. 현재 노드가 `None`이면 목표가 리스트에 없는 것이다.

**시간 복잡도:** 최악의 경우(목표가 마지막 원소이거나 없을 때) $O(n)$이다. 최선의 경우(목표가 머리일 때) $O(1)$이다.

**평균의 경우:** 각 원소가 탐색될 가능성이 같다고 가정하면 비교 횟수의 기댓값은 다음과 같다.

$$
\frac{1}{n} \sum_{i=1}^{n} i = \frac{n + 1}{2} = \Theta(n)
$$

---

## 2. 위치로 탐색하기

0에서 시작하는 인덱스 $k$의 원소에 접근하려면 머리에서 노드를 정확히 $k$개 지나간다.

**알고리즘:**

1. 계수기를 0으로 두고 머리에서 시작한다.
2. 다음 노드로 나아가며 계수기를 늘린다.
3. 계수기가 $k$에 이르면 현재 노드를 반환한다.
4. $k$에 이르기 전에 `None`에 닿으면 인덱스가 범위를 벗어난 것이다.

**시간 복잡도:** $O(k)$이며, $k = n - 1$인 최악의 경우 $O(n)$이다.

!!! warning "연결 리스트에는 이진 탐색을 쓸 수 없다"

    연결 리스트가 정렬되어 있어도 이진 탐색을 효율적으로 쓸 수 없다. 이진 탐색은 가운데 원소에 $O(1)$ 시간에 접근할 수 있어야 하는데, 연결 리스트의 가운데를 찾으려면 $O(n/2)$의 순회가 든다. 정렬된 연결 리스트에 이진 탐색을 적용하면 총 $O(n \log n)$ 시간이 걸려 단순한 선형 탐색보다도 나쁘다. 정렬된 탐색이 잦다면 배열이나 균형 탐색 트리를 쓰라.

---

## 3. 가운데 노드 찾기

유용한 변형으로 **두 포인터 기법**이 있다. 느린 포인터는 한 걸음씩, 빠른 포인터는 두 걸음씩 나아간다. 빠른 포인터가 끝에 닿으면 느린 포인터가 가운데에 있다.

**시간 복잡도:** $O(n)$이다. 빠른 포인터가 리스트 전체를 훑는다.

**공간 복잡도:** $O(1)$이다. 포인터 둘뿐이다.

이 기법을 쓰면 (한 번은 세고 한 번은 가운데까지 가는) 두 번의 훑기를 하지 않아도 된다.

---

## 4. 끝에서 k번째 노드 찾기

또 다른 두 포인터 기법이다. 첫 포인터를 $k$걸음 앞서 보낸 뒤, 첫 포인터가 끝에 닿을 때까지 두 포인터를 함께 나아가게 한다. 그러면 두 번째 포인터가 끝에서 $k$번째 자리에 있다.

**시간 복잡도:** 한 번 훑어 $O(n)$이다.

---

## 5. 구현

```python
"""단일 연결 리스트의 탐색 연산."""

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

# === 값으로 탐색 ===
def search_value(head, target):
    """data == target인 첫 노드를 돌려주고 없으면 None을 돌려준다.

    시간: O(n), 공간: O(1).
    """
    current = head
    while current is not None:
        if current.data == target:
            return current
        current = current.next
    return None

# === 위치로 탐색 ===
def search_position(head, k):
    """0부터 세는 인덱스 k의 노드를 돌려주거나 IndexError를 일으킨다.

    시간: O(k), 공간: O(1).
    """
    current = head
    for _ in range(k):
        if current is None:
            raise IndexError(f"Index {k} out of range")
        current = current.next
    if current is None:
        raise IndexError(f"Index {k} out of range")
    return current

# === 가운데 노드 찾기 ===
def find_middle(head):
    """느린 포인터와 빠른 포인터 기법으로 가운데 노드를 돌려준다.

    길이가 짝수이면 가운데 두 노드 중 뒤쪽을 돌려준다.
    시간: O(n), 공간: O(1).
    """
    if head is None:
        return None
    slow = head
    fast = head
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
    return slow

# === 끝에서 k번째 찾기 ===
def kth_from_end(head, k):
    """뒤에서 k번째 노드를 돌려준다(1부터 센다).

    시간: O(n), 공간: O(1).
    """
    first = head
    for _ in range(k):
        if first is None:
            raise IndexError(f"List has fewer than {k} nodes")
        first = first.next

    second = head
    while first is not None:
        first = first.next
        second = second.next
    return second

# === 시연 ===
if __name__ == "__main__":
    head = build_list([10, 20, 30, 40, 50])
    print(f"List: {to_list(head)}")

    # 값으로 탐색
    result = search_value(head, 30)
    print(f"Search for 30: {result}")
    result = search_value(head, 99)
    print(f"Search for 99: {result}")

    # 위치로 탐색
    result = search_position(head, 2)
    print(f"Position 2: {result}")

    # 가운데 찾기
    result = find_middle(head)
    print(f"Middle: {result}")

    # 끝에서 k번째
    result = kth_from_end(head, 2)
    print(f"2nd from end: {result}")
```

**출력:**
```
List: [10, 20, 30, 40, 50]
Search for 30: Node(30)
Search for 99: None
Position 2: Node(30)
Middle: Node(30)
2nd from end: Node(40)
```

---

## 6. 복잡도 요약

| 연산             | 시간       | 공간  |
|-----------------------|------------|--------|
| 값으로 탐색       | $O(n)$     | $O(1)$ |
| 위치 $k$로 탐색| $O(k)$    | $O(1)$ |
| 가운데 찾기           | $O(n)$     | $O(1)$ |
| 끝에서 k번째         | $O(n)$     | $O(1)$ |

모든 탐색 연산은 정해진 개수의 포인터 변수만 유지하므로 보조 공간이 상수이다.

---

## 연습문제

**연습문제 1.**
탐색에 대해 삽입, 삭제, 탐색, 접근 연산의 시간 복잡도를 진술하라.

??? success "연습문제 1 풀이"
    복잡도는 구체적인 구현(배열 기반이냐 연결 기반이냐)에 달려 있다. 배열 기반은 접근이 $O(1)$이고 임의의 위치에서의 삽입·삭제가 $O(n)$이다. 연결 기반은 이미 아는 위치에서의 삽입·삭제가 $O(1)$이고 탐색·접근이 $O(n)$이다. 어떤 연산이 주를 이루느냐에 따라 선택이 갈린다.

---

**연습문제 2.**
원소 6개로 탐색을 따라가며 각 연산 후의 자료구조 상태를 보여라.

??? success "연습문제 2 풀이"
    구조에 삽입, 접근, 삭제를 차례로 수행하라. 각 단계마다 (연결 구조라면) 포인터를, (배열 기반이라면) 배열의 내용을 보이며 구조가 불변식을 어떻게 유지하는지 나타내라.

---

**연습문제 3.**
탐색이 PyTorch의 텐서 저장과 어떻게 관련되는지 설명하라. 자료구조의 선택이 메모리 배치와 캐시 성능에 어떤 영향을 주는가?

??? success "연습문제 3 풀이"
    PyTorch 텐서는 캐시에 효율적으로 접근할 수 있도록 연속된 배열로 저장된다. 연결 구조는 autograd 그래프를 훑는 데 내부적으로 쓰인다. 이 선택은 메모리 사용량(배열에는 포인터 부담이 없다)과 접근 양상(캐시 지역성 덕분에 순차적인 배열 접근이 연결 리스트 순회보다 10~100배 빠르다)에 모두 영향을 준다.

---

**연습문제 4.**
반복문 불변식을 사용하여 탐색의 주요 연산의 시간 복잡도를 증명하라.

??? success "연습문제 4 풀이"
    알고리즘의 반복문이 유지하는 불변식을 진술하라. 초기화, 유지, 종료를 증명하라. 이 불변식으로부터 반복문이 명시된 횟수 안에 끝남이 따라 나오며, 이로써 복잡도의 상계가 확립된다. $\square$

## 정리하며

이 마당은 값으로 탐색하기、위치로 탐색하기、가운데 노드 찾기、끝에서 k번째 노드 찾기을 차례로 짚었다.

**참고 문헌**

- [Introduction to Algorithms (CLRS), Chapter 10](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
