# 배열 기반 큐
```python
import numpy as np; np.random.seed(0)

def main():
    q = []
    
    for i in range(10):
        if np.random.uniform() < 0.5:
            enqueue_item = np.random.randint(low=1, high=7)
            q.append(enqueue_item)
            print(f'{i = }, {enqueue_item = :4}, {q = }')
        else:
            try:
                dequeue_item = q.pop(0)
            except IndexError:
                dequeue_item = str(None)
            print(f'{i = }, {dequeue_item = :4}, {q = }')

            
if __name__ == "__main__":
    main()
```

**출력:**
```
i = 0, dequeue_item = None, q = []
i = 1, dequeue_item = None, q = []
i = 2, dequeue_item = None, q = []
i = 3, dequeue_item = None, q = []
i = 4, enqueue_item =    6, q = [6]
i = 5, enqueue_item =    1, q = [6, 1]
i = 6, dequeue_item =    6, q = [1]
i = 7, enqueue_item =    1, q = [1, 1]
i = 8, enqueue_item =    2, q = [1, 1, 2]
i = 9, enqueue_item =    2, q = [1, 1, 2, 2]
```

# 참고 문헌

[Queue - Data Structures & Algorithms Tutorials In Python #8](https://www.youtube.com/watch?v=rUUrmGKYwHw)

## 연습문제

**연습문제 1.**
큐를 구현할 때 `list.pop(0)`이 비효율적인 이유는 무엇인가?

??? success "연습문제 1 풀이"
    `pop(0)`은 첫 원소를 없애고 남은 원소를 모두 왼쪽으로 옮기므로 $O(n)$ 시간이 든다. 빼기를 $n$번 하면 전체 비용이 $O(n^2)$이다. `popleft`이 $O(1)$인 `collections.deque`을 쓰라.

---

**연습문제 2.**
용량이 고정된 원형 버퍼 큐를 구현하라.

??? success "연습문제 2 풀이"
    ```python
    class CircularQueue:
        def __init__(self, k):
            self.data = [None] * k
            self.head = self.tail = 0
            self.size = 0
        def enqueue(self, val):
            self.data[self.tail] = val
            self.tail = (self.tail + 1) % len(self.data)
            self.size += 1
        def dequeue(self):
            val = self.data[self.head]
            self.head = (self.head + 1) % len(self.data)
            self.size -= 1
            return val
    ```

---

**연습문제 3.**
파이썬 리스트로 하는 넣기와 빼기의 상각 시간 복잡도는 얼마인가?

??? success "연습문제 3 풀이"
    넣기(`append`)는 상각 $O(1)$이고, 빼기(`pop(0)`)는 옮기기 때문에 $O(n)$이다. 이 비대칭 때문에 보통의 리스트는 큐로 알맞지 않다. 덱은 둘 다 $O(1)$에 한다.

---

**연습문제 4.**
생산자-소비자 방식을 설명하고 거기에 큐가 꼭 필요한 이유를 말하라.

??? success "연습문제 4 풀이"
    생산자는 큐에 항목을 넣고 소비자는 그것을 뺀다. 큐가 생산과 소비의 속도를 떼어 놓는다. 여러 스레드를 쓰는 프로그램에서는 스레드에 안전한 큐(`queue.Queue`)가 경쟁 상태를 막는다. 선입선출 순서 덕분에 항목이 도착한 순서대로 처리된다.
