# 감시 노드

머리에 삽입하기, 꼬리 지우기, 임의의 노드 없애기 등 이중 연결 리스트의 모든 연산은 "리스트가 비었는가?", "이것이 머리인가?", "이것이 꼬리인가?" 같은 경계 사례를 다뤄야 한다. 이런 검사는 코드를 어수선하게 만들고 버그가 생길 틈을 준다. **감시 노드**(더미 노드라고도 한다)는 리스트의 경계에 놓여 진짜 데이터를 담지 않는 특별한 노드로, 모든 진짜 노드가 언제나 앞 노드와 뒤 노드를 함께 갖도록 보장하는 것만이 그 목적이다. 이 절은 감시 노드가 모든 `None` 검사를 없애 이중 연결 리스트 코드를 어떻게 간단하게 만드는지 보여준다.

---

## 1. 감시 노드가 없을 때의 문제

보통의 이중 연결 리스트에서 주어진 노드 `x` 뒤에 새 노드를 넣는 경우를 생각하자. 감시 노드가 없으면 코드가 두 가지 특별한 경우를 다뤄야 한다.

```python
# 보초가 없으면 -- 경계 검사가 필요하다
def insert_after(x, new_node):
    new_node.prev = x
    new_node.next = x.next
    if x.next is not None:        # 특수한 경우: x가 꼬리일 때
        x.next.prev = new_node
    x.next = new_node
```

마찬가지로 삭제할 때도 지울 노드가 머리인지 꼬리인지 확인해야 한다. 모든 연산이 이런 부담을 진다.

---

## 2. 감시 노드를 쓰는 설계

감시 노드를 쓰는 이중 연결 리스트는 리스트의 시작과 끝을 함께 나타내는 감시 노드 `s` 하나를 둔다. 이 감시 노드는 원형으로 이어진다.

- `s.next`는 첫 번째 진짜 노드를 가리킨다(리스트가 비어 있으면 `s` 자신을 되가리킨다).
- `s.prev`는 마지막 진짜 노드를 가리킨다(리스트가 비어 있으면 `s` 자신을 되가리킨다).

빈 리스트는 그저 감시 노드가 양쪽 방향으로 자기 자신을 가리키는 것이다.

$$
s.\text{next} = s \quad \text{and} \quad s.\text{prev} = s
$$

감시 노드는 **결코 지워지지 않으며** **사용자 데이터를 담지도 않는다**. 오직 구조를 위한 요소로만 존재한다.

---

## 3. 구현

```python
"""
보초 노드를 갖는 이중 연결 리스트.

보초는 경계 조건에 대한 None 검사를 모두 없애어
삽입과 삭제 코드를 한결같고 간결하게 만든다.
"""

# === 노드 정의 ===

class Node:
    """보초를 쓰는 이중 연결 리스트의 노드."""

    def __init__(self, data=None):
        self.data = data
        self.prev = None
        self.next = None

# === 보초를 쓰는 이중 연결 리스트 ===

class SentinelDLL:
    """보초 노드를 쓰는 이중 연결 리스트."""

    def __init__(self):
        self.sentinel = Node()          # 허수아비 노드, 실제 데이터 없음
        # 자기 자신을 양쪽으로 가리키게 해 고리를 만든다. 이 두 줄이
        # 이 페이지의 전부라 해도 좋다. 리스트가 비어 있든 차 있든
        # 모든 노드에 앞과 뒤가 반드시 있게 되므로, prev나 next가
        # None인 경우가 아예 생기지 않는다.
        # 앞 절의 DoublyLinkedList가 머리와 꼬리를 따로 검사하다가
        # 빠뜨리곤 했던 자리들이 여기서는 통째로 사라진다
        self.sentinel.next = self.sentinel
        self.sentinel.prev = self.sentinel

    def is_empty(self):
        # 보초가 자기 자신을 가리키고 있으면 비어 있다는 뜻이다.
        # head is None으로 묻던 것이 이렇게 바뀐다
        return self.sentinel.next is self.sentinel

    def insert_after(self, x, data):
        """'data'를 담은 새 노드를 노드 x 바로 뒤에 넣는다."""
        new_node = Node(data)
        # 이음선 넷을 고치는 것은 앞 절과 같다. 다른 것은 조건문이
        # 하나도 없다는 점이다. x가 어디에 있든, 리스트가 비어 있든
        # 이 네 줄이 그대로 맞는다
        new_node.prev = x
        new_node.next = x.next
        # x.next는 결코 None이 아니다. 마지막 노드라면 보초를 가리킨다
        x.next.prev = new_node         # None 검사가 필요 없다
        # 순서에 주의하라. x.next를 먼저 덮어쓰면 바로 윗줄에서 쓸
        # 옛 다음 노드를 잃는다
        x.next = new_node
        return new_node

    def insert_front(self, data):
        """리스트의 맨 앞(보초 뒤)에 넣는다."""
        return self.insert_after(self.sentinel, data)

    def insert_back(self, data):
        """리스트의 맨 뒤(보초 앞)에 넣는다."""
        # sentinel.prev가 곧 꼬리다. 고리이므로 보초의 앞이 마지막
        # 노드이기 때문이며, 덕분에 끝까지 걸어가지 않고 O(1)에 끝난다.
        # 앞 절의 insert_at_end가 꼬리를 찾느라 O(n)이었던 것과 견주어 보라.
        # 앞뒤 어느 쪽에든 O(1)로 넣고 뺄 수 있으므로 이 구조가
        # 데크(deque)의 뼈대가 된다
        return self.insert_after(self.sentinel.prev, data)

    def delete(self, x):
        """리스트에서 노드 x를 없앤다. x는 보초여서는 안 된다."""
        # 두 줄이면 끝난다. 앞 절의 remove가 index==0을 따로 다루고
        # 마지막 노드에서 None을 만나 터지던 것과 견주어 보라.
        # x가 첫 노드면 x.prev가 보초이고 마지막 노드면 x.next가
        # 보초이므로, 어느 경우든 검사 없이 그대로 통한다.
        # 지울 노드를 이미 손에 쥐고 있으므로 O(1)이다. 인덱스로
        # 지운다면 그 자리를 찾는 데 O(n)이 든다
        x.prev.next = x.next           # None 검사가 필요 없다
        x.next.prev = x.prev
        # 떼어 낸 노드의 이음선을 끊어 둔다. 없어도 동작에는 지장이
        # 없지만, 남겨 두면 지운 노드를 들고 리스트를 훑는 실수를
        # 저지를 수 있고 참조가 남아 메모리도 붙들린다
        x.prev = None
        x.next = None
        return x.data

    def to_list(self):
        """모든 데이터 값을 파이썬 리스트로 돌려준다(정방향)."""
        result = []
        # 보초 다음이 첫 데이터 노드다. 멈추는 조건도 None이 아니라
        # "보초로 되돌아왔는가"이다. 고리이므로 이 조건을 빠뜨리면
        # 영영 돌게 된다.
        # 보초 자신은 데이터가 없으므로 결과에 들어가지 않는다
        current = self.sentinel.next
        while current is not self.sentinel:
            result.append(current.data)
            current = current.next
        return result

    def to_list_reverse(self):
        """모든 데이터 값을 역순으로 돌려준다."""
        result = []
        current = self.sentinel.prev
        while current is not self.sentinel:
            result.append(current.data)
            current = current.prev
        return result

# === 메인 ===

if __name__ == "__main__":
    dll = SentinelDLL()

    # 원소 삽입
    dll.insert_back(10)
    dll.insert_back(20)
    dll.insert_back(30)
    print("Forward: ", dll.to_list())
    print("Backward:", dll.to_list_reverse())

    # 앞에 삽입
    dll.insert_front(5)
    print("After insert_front(5):", dll.to_list())

    # 두 번째 원소(값 10) 삭제
    node_10 = dll.sentinel.next.next   # 5 -> 10 -> 20 -> 30
    dll.delete(node_10)
    print("After deleting 10:    ", dll.to_list())
```

**출력:**

```
Forward:  [10, 20, 30]
Backward: [30, 20, 10]
After insert_front(5): [5, 10, 20, 30]
After deleting 10:     [5, 20, 30]
```

---

## 4. 감시 노드가 통하는 이유

감시 노드는 핵심 불변식 하나를 보장한다. **모든 진짜 노드는 유효한 `Node` 객체인 앞 노드와 뒤 노드를 갖는다.** 더 정확히 말해 모든 진짜 노드 $x$에 대해 다음이 성립한다.

- $x.\text{prev}$은 또 다른 진짜 노드이거나 감시 노드이다.
- $x.\text{next}$은 또 다른 진짜 노드이거나 감시 노드이다.

감시 노드는 (`None`이 아니라) 실제 `Node` 객체이므로 `x.next.prev = ...`과 `x.prev.next = ...` 같은 대입이 언제나 안전하다. 이로써 삽입과 삭제가 여러 갈래의 조건문에서 포인터를 네 번 고치는 정해진 순서로 바뀐다.

---

## 5. 복잡도

감시 노드는 어떤 연산의 점근적 복잡도도 바꾸지 않는다. 다만 상수 시간의 조건 분기를 없앤다.

| 연산 | 감시 노드 없이 | 감시 노드와 함께 |
|---|---|---|
| 노드 뒤에 삽입 | 분기 있는 $O(1)$ | 분기 없는 $O(1)$ |
| 주어진 노드 삭제 | 분기 있는 $O(1)$ | 분기 없는 $O(1)$ |
| 탐색 | $O(n)$ | $O(n)$ |
| 추가 공간 | 0 | 노드 1개 |

실질적인 이득은 점근적으로 더 빠른 성능이 아니라 더 간단하고 실수가 덜한 코드이다.

!!! tip "감시 노드를 언제 쓸 것인가"
    감시 노드는 삽입과 삭제가 주된 연산이고 리스트가 자주 바뀔 때 빛을 발한다. 읽기가 많은 작업이나 아주 짧은 리스트에서는 여분의 감시 노드가 불필요한 부담이다. CLRS는 연결 리스트를 설명하는 내내 감시 노드를 표준적인 구현 방식으로 쓴다.

---

## 연습문제

**연습문제 1.**
감시 노드에 대해 삽입, 삭제, 탐색, 접근 연산의 시간 복잡도를 진술하라.

??? success "연습문제 1 풀이"
    복잡도는 구체적인 구현(배열 기반이냐 연결 기반이냐)에 달려 있다. 배열 기반은 접근이 $O(1)$이고 임의의 위치에서의 삽입·삭제가 $O(n)$이다. 연결 기반은 이미 아는 위치에서의 삽입·삭제가 $O(1)$이고 탐색·접근이 $O(n)$이다. 어떤 연산이 주를 이루느냐에 따라 선택이 갈린다.

---

**연습문제 2.**
원소 6개로 감시 노드을(를) 따라가며 각 연산 후의 자료구조 상태를 보여라.

??? success "연습문제 2 풀이"
    구조에 삽입, 접근, 삭제를 차례로 수행하라. 각 단계마다 (연결 구조라면) 포인터를, (배열 기반이라면) 배열의 내용을 보이며 구조가 불변식을 어떻게 유지하는지 나타내라.

---

**연습문제 3.**
감시 노드이(가) PyTorch의 텐서 저장과 어떻게 관련되는지 설명하라. 자료구조의 선택이 메모리 배치와 캐시 성능에 어떤 영향을 주는가?

??? success "연습문제 3 풀이"
    PyTorch 텐서는 캐시에 효율적으로 접근할 수 있도록 연속된 배열로 저장된다. 연결 구조는 autograd 그래프를 훑는 데 내부적으로 쓰인다. 이 선택은 메모리 사용량(배열에는 포인터 부담이 없다)과 접근 양상(캐시 지역성 덕분에 순차적인 배열 접근이 연결 리스트 순회보다 10~100배 빠르다)에 모두 영향을 준다.

---

**연습문제 4.**
반복문 불변식을 사용하여 감시 노드의 주요 연산의 시간 복잡도를 증명하라.

??? success "연습문제 4 풀이"
    알고리즘의 반복문이 유지하는 불변식을 진술하라. 초기화, 유지, 종료를 증명하라. 이 불변식으로부터 반복문이 명시된 횟수 안에 끝남이 따라 나오며, 이로써 복잡도의 상계가 확립된다. $\square$

## 정리하며

이 마당은 감시 노드가 없을 때의 문제、감시 노드를 쓰는 설계、구현、감시 노드가 통하는 이유을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C.
  *Introduction to Algorithms* (4th ed.), Chapter 10.3. MIT Press.
