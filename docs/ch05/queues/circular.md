# 원형 큐

소박한 배열 기반 큐는 배열의 앞쪽을 빼는 끝으로 쓴다. 뺄 때마다 남은 원소를 모두 앞으로 옮겨야 하므로 $O(n)$ 시간이 든다. **원형 큐**(링 버퍼라고도 한다)는 바탕 배열을 원처럼 다루어 이를 해결한다. 뒤 포인터가 배열의 끝에 이르면 처음으로 돌아간다. 그러면 원소를 옮길 필요가 없으므로 넣기와 빼기 모두 $O(1)$ 시간이다. 이 기법은 운영체제의 입출력 버퍼, 음향 처리, 네트워크 패킷 큐에 널리 쓰인다. 이 쪽은 돌아 감는 방식을 설명하고, 가득 참과 비어 있음의 모호함을 다루며, 완전한 구현을 제시한다.

## 돌아 감기 문제

선형 배열에서는 넣기와 빼기를 여러 번 하면 앞 포인터와 뒤 포인터가 모두 배열의 끝 쪽으로 나아간다. 배열 앞쪽의 자리가 비었는데도 다시 쓸 수 없다. 용량 대부분이 낭비된 채 배열이 가득 찬 것처럼 보인다.

원형 큐는 **나머지 연산**으로 포인터를 감아 이를 바로잡는다.

$$
\text{next}(i) = (i + 1) \bmod C
$$

여기서 $C$은 바탕 배열의 용량이다. 뒤 포인터가 위치 $C-1$에 이르면 다음 넣기는 (비어 있다면) 위치 0에 원소를 놓아 앞서 빼기로 비운 자리를 다시 쓴다.

## 가득 참과 비어 있음 가려내기

가득 찬 상태와 빈 상태는 `front == rear`이라는 같은 증상을 보인다. 이를 가려내는 흔한 해결책이 두 가지 있다.

1. **한 칸을 비워 두기**: 배열의 한 칸을 언제나 비워 둔다. `(rear + 1) mod C == front`이면 큐가 가득 찬 것이다. 저장할 수 있는 원소는 최대 $C - 1$개이다.
2. **개수를 세기**: 따로 `size` 변수를 둔다. `size == C`이면 가득 찬 것이고 `size == 0`이면 빈 것이다.

여기서는 명확함을 위해 개수를 세는 방식을 쓴다.

## 구현

```python
"""
원형 큐 — 원형 배열을 쓰는 크기 고정 큐.

앞과 뒤 포인터를 배열의 끝에서 감아 나머지 연산으로 넣기와 빼기를
O(1)에 한다.
"""


# === 원형 큐 ===========================================================

class CircularQueue:
    """순환 배열로 구현한 고정 용량 큐.

    가득 참과 비어 있음을 가르려고 크기 세개를 쓴다.
    모든 연산이 최악의 경우에도 O(1) 시간에 돌아간다.
    """

    def __init__(self, capacity):
        self._data = [None] * capacity
        self._capacity = capacity
        self._front = 0
        self._rear = 0
        self._size = 0

    def enqueue(self, x):
        """원소 x를 뒤에 넣는다. 가득 차 있으면 OverflowError를 일으킨다."""
        if self.is_full():
            raise OverflowError("enqueue to full queue")
        self._data[self._rear] = x
        self._rear = (self._rear + 1) % self._capacity
        self._size += 1

    def dequeue(self):
        """앞 원소를 빼서 돌려준다. 비어 있으면 IndexError를 일으킨다."""
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        value = self._data[self._front]
        self._data[self._front] = None  # 쓰레기 수집을 돕는다
        self._front = (self._front + 1) % self._capacity
        self._size -= 1
        return value

    def front(self):
        """앞 원소를 빼지 않고 돌려준다."""
        if self.is_empty():
            raise IndexError("front from empty queue")
        return self._data[self._front]

    def is_empty(self):
        """큐에 원소가 없으면 True를 돌려준다."""
        return self._size == 0

    def is_full(self):
        """큐가 용량에 다다랐으면 True를 돌려준다."""
        return self._size == self._capacity

    def size(self):
        """지금 큐에 든 원소의 개수를 돌려준다."""
        return self._size

    def _snapshot(self):
        """디버깅을 위해 내부 배열의 상태를 돌려준다."""
        return self._data.copy()

    def __repr__(self):
        # 논리적인 차례로 원소 보이기 (앞에서 뒤로)
        if self._size == 0:
            return "CircularQueue([])"
        elements = []
        i = self._front
        for _ in range(self._size):
            elements.append(self._data[i])
            i = (i + 1) % self._capacity
        return f"CircularQueue({elements})"


# === 시연 ============================================================

if __name__ == "__main__":
    cq = CircularQueue(capacity=5)

    print(f"{'Operation':<20s} {'Logical':<25s} {'Internal Array':<25s} {'front':>5s} {'rear':>5s}")
    print("-" * 82)

    def show(label):
        print(f"{label:<20s} {str(cq):<25s} {str(cq._snapshot()):<25s} {cq._front:>5d} {cq._rear:>5d}")

    # 1~4 넣기
    for x in [1, 2, 3, 4]:
        cq.enqueue(x)
        show(f"enqueue({x})")

    # 원소 2개 빼기
    for _ in range(2):
        val = cq.dequeue()
        show(f"dequeue() → {val}")

    # 2개 더 넣기 (돌아 감긴다!)
    for x in [5, 6]:
        cq.enqueue(x)
        show(f"enqueue({x})")

    # 남은 것 빼기
    while not cq.is_empty():
        val = cq.dequeue()
        show(f"dequeue() → {val}")
```

**출력:**
```
Operation            Logical                   Internal Array            front  rear
----------------------------------------------------------------------------------
enqueue(1)           CircularQueue([1])        [1, None, None, None, None]     0     1
enqueue(2)           CircularQueue([1, 2])     [1, 2, None, None, None]     0     2
enqueue(3)           CircularQueue([1, 2, 3])  [1, 2, 3, None, None]     0     3
enqueue(4)           CircularQueue([1, 2, 3, 4]) [1, 2, 3, 4, None]     0     4
dequeue() → 1        CircularQueue([2, 3, 4])  [None, 2, 3, 4, None]     1     4
dequeue() → 2        CircularQueue([3, 4])     [None, None, 3, 4, None]     2     4
enqueue(5)           CircularQueue([3, 4, 5])  [None, None, 3, 4, 5]     2     0
enqueue(6)           CircularQueue([3, 4, 5, 6]) [6, None, 3, 4, 5]     2     1
dequeue() → 3        CircularQueue([4, 5, 6])  [6, None, None, 4, 5]     3     1
dequeue() → 4        CircularQueue([5, 6])     [6, None, None, None, 5]     4     1
dequeue() → 5        CircularQueue([6])        [6, None, None, None, None]     0     1
dequeue() → 6        CircularQueue([])         [None, None, None, None, None]     1     1
```

따라가기는 돌아 감기가 실제로 어떻게 일어나는지 보여 준다. 1~4를 넣고 1~2를 뺀 뒤 뒤 포인터는 위치 4에 있다. 5를 넣으면 위치 4에 들어가고 뒤 포인터는 위치 0으로 감긴다. 6을 넣으면 위치 0에 들어가 앞서 빼기로 비운 자리를 다시 쓴다. 원소가 내부 배열에 흩어져 있어도 논리적인 순서(앞에서 뒤로)는 언제나 선입선출을 따른다.

## 복잡도

| 연산 | 시간 | 공간 |
|-----------|------|-------|
| `enqueue(x)` | $O(1)$ | $O(1)$ |
| `dequeue()` | $O(1)$ | $O(1)$ |
| `front()` | $O(1)$ | $O(1)$ |
| `is_empty()` | $O(1)$ | $O(1)$ |
| `is_full()` | $O(1)$ | $O(1)$ |

자료 구조 전체의 공간은 용량을 $C$이라 할 때 $O(C)$이다. 모든 연산이 최악의 경우에도 $O(1)$이며, 동적 배열과 달리 상각이 필요 없다.

!!! tip "고정 용량과 가변 용량"
    원형 큐는 용량이 고정되어 있다. 크기를 바꾸어야 한다면 배열을 두 배로 늘리고 모든 원소를 새 배열의 위치 0부터 복사해야 한다. 그러면 이따금 넣기가 $O(n)$이 되지만 상각으로는 $O(1)$이 유지된다. 입출력 버퍼처럼 큐의 최대 크기를 미리 아는 응용에서는 고정 용량이 낫다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 10. MIT Press.


## 연습문제

**연습문제 1.**
원형 큐의 추상 자료형이 지원하는 연산을 시간 복잡도와 함께 모두 열거하라. 어느 연산이 병목인가?

??? success "연습문제 1 풀이"
    추상 자료형은 구현과 무관하게 지원하는 연산을 정한다. 무엇이 병목인지는 쓰임새에 달렸다. 실시간 시스템에서는 최악의 복잡도가 중요하고, 일괄 처리에서는 상각 복잡도로 충분하다.

---

**연습문제 2.**
원형 큐을(를) 서로 다른 두 자료 구조로 구현하라. 각각의 절충을 비교하라.

??? success "연습문제 2 풀이"
    구현 1: 배열 기반 — 접근은 상수 시간이지만 크기를 다시 잡아야 할 수 있다. 구현 2: 연결 리스트 기반 — 삽입과 삭제는 상수 시간이지만 접근은 $O(n)$이다. 어느 쪽을 고를지는 응용에서 예상되는 연산의 구성에 달렸다.

---

**연습문제 3.**
원형 큐을(를) 쓰는 딥러닝 응용을 하나 설명하라(예: 그래프 신경망의 너비 우선 탐색, 기호 미분에서의 식 계산, 데이터 적재의 스케줄링).

??? success "연습문제 3 풀이"
    구체적인 응용은 그 추상 자료형의 순서 성질에 달렸다. 선입선출(큐)은 GNN의 너비 우선 그래프 순회에 쓰이고, 후입선출(스택)은 자동 미분 테이프 처리에 쓰이며, 우선순위 순서는 빔 탐색과 예정 표집에 쓰인다.

---

**연습문제 4.**
원형 큐을(를) 원형 배열로 구현하면 모든 연산이 상각 $O(1)$ 시간임을 증명하라.

??? success "연습문제 4 풀이"
    원형 배열은 머리와 꼬리 인덱스를 용량으로 나눈 나머지로 관리한다. 넣기와 빼기는 인덱스를 $O(1)$에 조정한다. 배열이 가득 차면 용량을 두 배로 늘리는 데 $O(n)$이 들지만, 이는 값싼 연산 $O(n)$번 뒤에 한 번 일어나므로 동적 배열과 같은 논법으로 상각 $O(1)$이 된다. $\square$