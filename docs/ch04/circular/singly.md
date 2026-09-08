# 원형 단일 연결 리스트

보통의 단일 연결 리스트에서는 마지막 노드의 `next` 포인터가 `None`이어서 사슬의 끝을 알린다. **원형 단일 연결 리스트**는 마지막 노드의 `next` 포인터를 첫 노드로 되돌려 이어 닫힌 고리를 이루도록 설계를 바꾼다. 이 원형 구조는 원소가 되풀이해서 돌아가는 문제를 자연스럽게 나타낸다. 작업 스케줄러, 여러 사람이 하는 게임의 차례, 스트리밍 데이터 버퍼 모두 경계에 부딪히지 않고 리스트를 끝없이 순회할 수 있다는 점에서 이득을 본다.

---

## 1. 구조

노드가 $x_0, x_1, \ldots, x_{n-1}$인 원형 단일 연결 리스트는 다음을 만족한다.

$$
x_i.\text{next} = x_{(i+1) \bmod n} \quad \text{for all } i
$$

특히 마지막 노드 $x_{n-1}$은 $x_{n-1}.\text{next} = x_0$이다. 이 구조에는 `None` 포인터가 없다.

흔한 설계 선택은 머리 대신 **꼬리** 노드에 대한 참조를 유지하는 것이다. `tail.next`로 머리에 $O(1)$에 닿을 수 있어 양쪽 끝에 모두 효율적으로 접근할 수 있기 때문이다.

---

## 2. 구현

```python
"""
원형 단일 연결 리스트의 구현.

마지막 노드의 next 포인터가 머리로 되돌아 이어져
고리를 이룬다. 꼬리 참조가 양 끝에 O(1)로 접근하게 해 준다.
"""

# === 노드 정의 ===

class Node:
    """원형 단일 연결 리스트의 노드."""

    def __init__(self, data):
        self.data = data
        self.next = None

# === 원형 단일 연결 리스트 ===

class CircularSLL:
    """꼬리 참조를 갖는 원형 단일 연결 리스트."""

    def __init__(self):
        self.tail = None
        self.size = 0

    def is_empty(self):
        return self.size == 0

    def insert_front(self, data):
        """리스트 맨 앞에 새 노드를 넣는다.

        새 노드가 tail.next(머리)가 된다.
        """
        new_node = Node(data)
        if self.is_empty():
            new_node.next = new_node   # 노드가 하나면 자기 자신을 가리킨다
            self.tail = new_node
        else:
            new_node.next = self.tail.next   # 새 머리가 옛 머리를 가리킨다
            self.tail.next = new_node        # 꼬리가 새 머리를 가리킨다
        self.size += 1

    def insert_back(self, data):
        """리스트 맨 뒤에 새 노드를 넣는다.

        새 노드가 새 꼬리가 된다.
        """
        self.insert_front(data)
        self.tail = self.tail.next     # 꼬리를 새로 넣은 노드로 옮긴다
        # insert_front가 머리에 넣었으므로, 꼬리로 만들면 회전이 된다

    def delete_front(self):
        """맨 앞(머리) 원소를 빼서 돌려준다."""
        if self.is_empty():
            raise IndexError("delete from empty list")
        head = self.tail.next
        if self.size == 1:
            self.tail = None
        else:
            self.tail.next = head.next
        self.size -= 1
        return head.data

    def rotate(self):
        """꼬리 참조를 한 자리 앞으로 옮긴다.

        이는 머리를 뒤로 옮기는 셈이어서 리스트를 한 칸 돌린다.
        """
        if not self.is_empty():
            self.tail = self.tail.next

    def traverse(self):
        """머리에서 시작하여 모든 값을 차례대로 돌려준다."""
        if self.is_empty():
            return []
        result = []
        current = self.tail.next       # 머리에서 시작
        while True:
            result.append(current.data)
            current = current.next
            if current is self.tail.next:
                break
        return result

    def search(self, target):
        """목표 값을 담은 노드를 찾는다.

        찾으면 노드를, 없으면 None을 돌려준다.
        """
        if self.is_empty():
            return None
        current = self.tail.next
        while True:
            if current.data == target:
                return current
            current = current.next
            if current is self.tail.next:
                return None

# === 메인 ===

if __name__ == "__main__":
    cll = CircularSLL()

    # 리스트 만들기
    cll.insert_back(10)
    cll.insert_back(20)
    cll.insert_back(30)
    print("List:", cll.traverse())

    # 앞에 삽입
    cll.insert_front(5)
    print("After insert_front(5):", cll.traverse())

    # 앞에서 삭제
    removed = cll.delete_front()
    print(f"Removed {removed}:", cll.traverse())

    # 회전
    cll.rotate()
    print("After rotate:", cll.traverse())
```

**출력:**

```
List: [10, 20, 30]
After insert_front(5): [5, 10, 20, 30]
Removed 5: [10, 20, 30]
After rotate: [20, 30, 10]
```

---

## 3. 순회의 종료

원형 리스트를 순회하는 것과 보통의 리스트를 순회하는 것의 근본적인 차이는 멈추는 조건에 있다. 보통의 리스트에서는 `current is None`일 때 순회가 멈춘다. 원형 리스트에는 `None`이 아예 나타나지 않으므로 커서가 출발한 노드로 돌아왔을 때 멈춰야 한다.

!!! warning "무한 반복의 위험"
    출발점으로 돌아왔는지 확인하는 것을 잊는 것이 원형 리스트 코드에서 가장 흔한 버그이다. 언제나 `do-while` 방식을 쓰라. 반복문에 들어가기 전에 출발 노드를 기록해 두고 매 반복의 끝에서 그것과 견주어 보면 된다.

---

## 4. 꼬리 참조를 쓰는 설계

머리 참조 대신 꼬리 참조를 유지하는 것은 실용적인 최적화이다.

| 연산 | 머리 참조 | 꼬리 참조 |
|---|---|---|
| 머리 접근 | $O(1)$ | `tail.next`로 $O(1)$ |
| 꼬리 접근 | $O(n)$ | $O(1)$ |
| 앞에 삽입 | 마지막 노드를 고치는 데 $O(n)$ | $O(1)$ |
| 뒤에 삽입 | 마지막 노드를 찾는 데 $O(n)$ | $O(1)$ |

꼬리 참조를 쓰면 앞쪽 삽입과 뒤쪽 삽입이 모두 상수 시간이 되어, 큐처럼 쓰는 작업에서 원형 단일 연결 리스트가 이중 연결 리스트에 견줄 만해진다.

---

## 5. 복잡도

| 연산 | 시간 |
|---|---|
| 앞에 삽입 | $O(1)$ |
| 뒤에 삽입 | $O(1)$ |
| 앞에서 삭제 | $O(1)$ |
| 임의 위치 삭제 | $O(n)$ (앞 노드가 필요하다) |
| 탐색 | $O(n)$ |
| 순회 | $O(n)$ |
| 회전 | $O(1)$ |

노드 $n$개에 대한 공간 복잡도는 $O(n)$이다. 보통의 단일 연결 리스트와 견주면 구조상 달라지는 것은 포인터 대입 하나(마지막 노드에서 첫 노드로)뿐이므로 추가 공간 부담이 없다.

---

## 연습문제

**연습문제 1.**
원형 단일 연결 리스트에 대해 삽입, 삭제, 탐색, 접근 연산의 시간 복잡도를 진술하라.

??? success "연습문제 1 풀이"
    복잡도는 구체적인 구현(배열 기반이냐 연결 기반이냐)에 달려 있다. 배열 기반은 접근이 $O(1)$이고 임의의 위치에서의 삽입·삭제가 $O(n)$이다. 연결 기반은 이미 아는 위치에서의 삽입·삭제가 $O(1)$이고 탐색·접근이 $O(n)$이다. 어떤 연산이 주를 이루느냐에 따라 선택이 갈린다.

---

**연습문제 2.**
원소 6개로 원형 단일 연결 리스트을(를) 따라가며 각 연산 후의 자료구조 상태를 보여라.

??? success "연습문제 2 풀이"
    구조에 삽입, 접근, 삭제를 차례로 수행하라. 각 단계마다 (연결 구조라면) 포인터를, (배열 기반이라면) 배열의 내용을 보이며 구조가 불변식을 어떻게 유지하는지 나타내라.

---

**연습문제 3.**
원형 단일 연결 리스트이(가) PyTorch의 텐서 저장과 어떻게 관련되는지 설명하라. 자료구조의 선택이 메모리 배치와 캐시 성능에 어떤 영향을 주는가?

??? success "연습문제 3 풀이"
    PyTorch 텐서는 캐시에 효율적으로 접근할 수 있도록 연속된 배열로 저장된다. 연결 구조는 autograd 그래프를 훑는 데 내부적으로 쓰인다. 이 선택은 메모리 사용량(배열에는 포인터 부담이 없다)과 접근 양상(캐시 지역성 덕분에 순차적인 배열 접근이 연결 리스트 순회보다 10~100배 빠르다)에 모두 영향을 준다.

---

**연습문제 4.**
반복문 불변식을 사용하여 원형 단일 연결 리스트의 주요 연산의 시간 복잡도를 증명하라.

??? success "연습문제 4 풀이"
    알고리즘의 반복문이 유지하는 불변식을 진술하라. 초기화, 유지, 종료를 증명하라. 이 불변식으로부터 반복문이 명시된 횟수 안에 끝남이 따라 나오며, 이로써 복잡도의 상계가 확립된다. $\square$

## 정리하며

이 마당은 구조、구현、순회의 종료、꼬리 참조를 쓰는 설계을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C.
  *Introduction to Algorithms* (4th ed.), Chapter 10. MIT Press.
- Goodrich, M. T., Tamassia, R., & Goldwasser, M. H.
  *Data Structures and Algorithms in Python*, Section 7.2. Wiley.
