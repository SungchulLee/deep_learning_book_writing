# 원형 배열

정적 배열로 큐를 구현하면 원소를 뒤에 넣고 앞에서 뺀다. 소박하게 만들면 뺄 때마다 남은 원소를 모두 왼쪽으로 밀어 $O(n)$ 시간이 들거나, 아니면 앞 인덱스만 앞으로 나아가다가 대부분의 자리가 비어 있는데도 결국 배열을 다 써 버린다. **원형 배열**(링 버퍼라고도 한다)은 배열이 빙 둘러 이어진 것처럼 다루어 두 문제를 함께 푼다. 인덱스가 마지막 자리를 지나면 첫 자리로 돌아간다. 이로써 공간을 낭비하지 않으면서 $O(1)$의 넣기와 빼기를 얻는다.

---

## 1. 나머지 연산을 이용한 인덱스 계산

빙 둘러 이어지는 동작은 나머지 연산자로 구현한다. 용량이 $c$인 배열에서 임의의 인덱스 $i$은 다음을 통해 물리적 위치로 대응된다.

$$
\text{pos}(i) = i \bmod c
$$

즉 인덱스 $c$은 위치 0으로, 인덱스 $c + 1$은 위치 1로 돌아가는 식이다. 원형 버퍼가 유지하는 핵심 포인터는 둘이다.

- **front**: 첫 번째(가장 오래된) 원소의 인덱스.
- **rear**: 다음에 삽입할 수 있는 빈자리의 인덱스.

넣을 때마다 rear는 다음과 같이 나아간다.

$$
\text{rear} \leftarrow (\text{rear} + 1) \bmod c
$$

뺄 때마다 front는 다음과 같이 나아간다.

$$
\text{front} \leftarrow (\text{front} + 1) \bmod c
$$

현재 원소의 개수는 다음과 같다.

$$
\text{size} = (\text{rear} - \text{front}) \bmod c
$$

---

## 2. 가득 참과 비어 있음의 구분

미묘한 문제가 하나 생긴다. `front == rear`일 때 버퍼가 완전히 비었을 수도, 완전히 찼을 수도 있다. 흔한 해결책이 둘 있다.

1. **자리 하나를 비워 둔다**: 자리 하나를 늘 비워 두어 $(\text{rear} + 1) \bmod c = \text{front}$일 때 버퍼가 가득 찬 것으로 본다. 쓸 수 있는 용량이 $c - 1$로 줄어든다.
2. **개수를 따로 센다**: 원소의 개수를 따로 추적하여 $c$개의 자리를 모두 쓴다.

아래 구현은 개수를 세는 방식을 쓴다.

---

## 3. 연산과 복잡도

| 연산 | 시간 복잡도 | 설명                                 |
|-----------|-----------------|---------------------------------------------|
| 넣기   | $O(1)$          | rear에 쓰고 rear 포인터를 나아가게 한다         |
| 빼기   | $O(1)$          | front에서 읽고 front 포인터를 나아가게 한다        |
| 들여다보기      | $O(1)$          | 나아가지 않고 front에서 읽는다             |
| 비었는가  | $O(1)$          | 개수가 0인지 확인한다                  |
| 가득 찼는가   | $O(1)$          | 개수가 용량과 같은지 확인한다              |

모든 연산이 상각이 아니라 최악의 경우에도 $O(1)$이다. 원소를 밀거나 다시 할당하는 일이 없다.

---

## 4. 구현

```python
"""고정 크기 배열로 구현한 원형 버퍼(링 버퍼)."""

# === 원형 버퍼 클래스 ===
class CircularBuffer:
    """O(1) 넣기와 빼기를 지원하는 고정 용량 원형 버퍼."""

    def __init__(self, capacity: int):
        # 배열은 한 번 잡아 두고 다시 잡지 않는다. 앞뒤 표시만 옮겨 다니며
        # 같은 칸을 돌려쓰는 것이 이 자료 구조의 요령이다.
        # 파이썬 리스트로 큐를 만들면 pop(0)이 뒤의 원소를 모두 한 칸씩
        # 당겨 O(n)이 드는데, 이쪽은 넣기와 빼기가 모두 O(1)이다
        self._data = [None] * capacity
        self._capacity = capacity
        self._front = 0
        self._rear = 0
        # 크기를 따로 센다. front와 rear만으로는 가득 찬 상태와 빈 상태를
        # 가릴 수 없기 때문이다. 둘 다 front == rear가 되어 버린다.
        # 한 칸을 비워 두어 구별하는 구현도 있지만 이쪽이 알아보기 쉽다
        self._size = 0

    def is_empty(self) -> bool:
        return self._size == 0

    def is_full(self) -> bool:
        return self._size == self._capacity

    def enqueue(self, value) -> None:
        if self.is_full():
            raise OverflowError("Buffer is full")
        self._data[self._rear] = value
        # 나머지 연산이 고리를 만든다. 끝에 닿으면 0으로 돌아가므로
        # 배열의 마지막 칸 다음이 첫 칸이 된다. "원형"이라는 이름이
        # 여기서 나오며, 실제로 도는 것은 자료가 아니라 이 표시다
        self._rear = (self._rear + 1) % self._capacity
        self._size += 1

    def dequeue(self):
        if self.is_empty():
            raise IndexError("Buffer is empty")
        value = self._data[self._front]
        # 값을 지우지 않아도 _size 덕에 논리적으로는 빠진 것이지만,
        # 배열이 참조를 붙들고 있으면 그 객체가 메모리에 남는다.
        # None으로 덮어써야 실제로 놓아 준다
        self._data[self._front] = None  # 쓰레기 수집을 돕는다
        # 앞 표시만 옮긴다. 원소를 밀지 않으므로 O(1)이다
        self._front = (self._front + 1) % self._capacity
        self._size -= 1
        return value

    def peek(self):
        if self.is_empty():
            raise IndexError("Buffer is empty")
        return self._data[self._front]

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        items = []
        # 배열의 0번 칸이 아니라 _front부터, 그것도 _size개만 훑는다.
        # 자료가 배열 끝에서 앞으로 감겨 있을 수 있고 빈 칸도 섞여
        # 있으므로, 저장된 차례가 아니라 논리적 차례로 읽어야 한다
        idx = self._front
        for _ in range(self._size):
            items.append(repr(self._data[idx]))
            idx = (idx + 1) % self._capacity
        return f"CircularBuffer([{', '.join(items)}])"

# === 시연 ===
if __name__ == "__main__":
    buf = CircularBuffer(4)
    buf.enqueue("A")
    buf.enqueue("B")
    buf.enqueue("C")
    print(f"After 3 enqueues: {buf}")

    print(f"Dequeue: {buf.dequeue()}")
    print(f"Dequeue: {buf.dequeue()}")
    print(f"After 2 dequeues: {buf}")

    buf.enqueue("D")
    buf.enqueue("E")
    print(f"After 2 more enqueues: {buf}")
    print(f"Internal array: {buf._data}")
```

**출력:**
```
After 3 enqueues: CircularBuffer(['A', 'B', 'C'])
Dequeue: A
Dequeue: B
After 2 dequeues: CircularBuffer(['C'])
After 2 more enqueues: CircularBuffer(['C', 'D', 'E'])
Internal array: ['E', None, 'C', 'D']
```

내부 배열을 보면 빙 둘러 이어지는 효과가 드러난다. 위치 0과 1에서 A와 B를 빼낸 뒤, 새 원소 D와 E가 위치 3을 채우고 다시 위치 0으로 돌아간다.

??? example "빙 둘러 이어지는 과정 따라가기"

    용량 4, `front = 0`, `rear = 0`, `size = 0`에서 시작한다.

    | 연산   | front | rear | size | 배열 상태              |
    |-------------|-------|------|------|--------------------------|
    | enqueue(A)  | 0     | 1    | 1    | `[A, _, _, _]`           |
    | enqueue(B)  | 0     | 2    | 2    | `[A, B, _, _]`           |
    | enqueue(C)  | 0     | 3    | 3    | `[A, B, C, _]`           |
    | dequeue→A   | 1     | 3    | 2    | `[_, B, C, _]`           |
    | dequeue→B   | 2     | 3    | 1    | `[_, _, C, _]`           |
    | enqueue(D)  | 2     | 0    | 2    | `[_, _, C, D]`           |
    | enqueue(E)  | 2     | 1    | 3    | `[E, _, C, D]`           |

    `rear`가 인덱스 3을 지나 나아가면 $(3 + 1) \bmod 4 = 0$을 통해 인덱스 0으로 돌아간다.

---

## 5. 응용

원형 배열은 몇 가지 중요한 쓰임새에서 표준적인 밑받침 구조이다.

- **큐 구현**: 원형 배열 큐는 공간을 낭비하지 않으면서 $O(1)$ 연산을 제공하며, 5장에서 다루는 배열 기반 큐에 쓰인다.
- **유계인 생산자-소비자 버퍼**: 운영체제와 입출력 시스템은 속도가 다른 생산자와 소비자 사이에서 데이터를 주고받는 데 링 버퍼를 쓴다.
- **스트리밍 데이터**: 음향 처리, 네트워크 패킷 버퍼, 로그 시스템은 가장 최근의 원소 $c$개를 남기고 오래된 것을 자동으로 버리는 데 원형 버퍼를 쓴다.
- **미끄럼창 알고리즘**: 데이터 흐름 위에 고정 크기의 창을 유지하는 일은 원형 버퍼에 그대로 대응된다.

---

## 연습문제

**연습문제 1.**
원형 배열에 대해 삽입, 삭제, 탐색, 접근 연산의 시간 복잡도를 진술하라.

??? success "연습문제 1 풀이"
    복잡도는 구체적인 구현(배열 기반이냐 연결 기반이냐)에 달려 있다. 배열 기반은 접근이 $O(1)$이고 임의의 위치에서의 삽입·삭제가 $O(n)$이다. 연결 기반은 이미 아는 위치에서의 삽입·삭제가 $O(1)$이고 탐색·접근이 $O(n)$이다. 어떤 연산이 주를 이루느냐에 따라 선택이 갈린다.

---

**연습문제 2.**
원소 6개로 원형 배열을(를) 따라가며 각 연산 후의 자료구조 상태를 보여라.

??? success "연습문제 2 풀이"
    구조에 삽입, 접근, 삭제를 차례로 수행하라. 각 단계마다 (연결 구조라면) 포인터를, (배열 기반이라면) 배열의 내용을 보이며 구조가 불변식을 어떻게 유지하는지 나타내라.

---

**연습문제 3.**
원형 배열이(가) PyTorch의 텐서 저장과 어떻게 관련되는지 설명하라. 자료구조의 선택이 메모리 배치와 캐시 성능에 어떤 영향을 주는가?

??? success "연습문제 3 풀이"
    PyTorch 텐서는 캐시에 효율적으로 접근할 수 있도록 연속된 배열로 저장된다. 연결 구조는 autograd 그래프를 훑는 데 내부적으로 쓰인다. 이 선택은 메모리 사용량(배열에는 포인터 부담이 없다)과 접근 양상(캐시 지역성 덕분에 순차적인 배열 접근이 연결 리스트 순회보다 10~100배 빠르다)에 모두 영향을 준다.

---

**연습문제 4.**
반복문 불변식을 사용하여 원형 배열의 주요 연산의 시간 복잡도를 증명하라.

??? success "연습문제 4 풀이"
    알고리즘의 반복문이 유지하는 불변식을 진술하라. 초기화, 유지, 종료를 증명하라. 이 불변식으로부터 반복문이 명시된 횟수 안에 끝남이 따라 나오며, 이로써 복잡도의 상계가 확립된다. $\square$

## 정리하며

이 마당은 나머지 연산을 이용한 인덱스 계산、가득 참과 비어 있음의 구분、연산과 복잡도、구현을 차례로 짚었다.

**참고 문헌**

- [Introduction to Algorithms (CLRS), Chapter 10](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
