# 연결 리스트 큐
```python
import numpy as np; np.random.seed(0)
from collections import deque

class Queue:   
    def __init__(self):
        self.queue = deque()
    def enqueue(self, val):
        self.queue.append(val)    
    def dequeue(self):
        if len(self.queue) > 0:
            return self.queue.popleft()
        else:
            return None
    def peek(self):
        if len(self.queue) > 0:
            return self.queue[0]
        else:
            return None
    def is_empty(self):
        return len(self.queue)==0
    def size(self):
        return len(self.queue)
    def __str__(self):
        return str(self.queue)
    
    
def main():
    q = Queue()

    for i in range(10):
        if np.random.uniform() < 0.5:
            enqueue_item = np.random.randint(low=1, high=7)
            q.enqueue(enqueue_item)
            print(f'{i = }, {enqueue_item = :>9}, {str(q) = }')
        else:
            dequeue_item = q.dequeue()
            print(f'{i = }, {str(dequeue_item) = :>4}, {str(q) = }')
            
            
if __name__ == "__main__":
    main()
```

**출력:**
```
i = 0, str(dequeue_item) = None, str(q) = 'deque([])'
i = 1, str(dequeue_item) = None, str(q) = 'deque([])'
i = 2, str(dequeue_item) = None, str(q) = 'deque([])'
i = 3, str(dequeue_item) = None, str(q) = 'deque([])'
i = 4, enqueue_item =         6, str(q) = 'deque([6])'
i = 5, enqueue_item =         1, str(q) = 'deque([6, 1])'
i = 6, str(dequeue_item) =    6, str(q) = 'deque([1])'
i = 7, enqueue_item =         1, str(q) = 'deque([1, 1])'
i = 8, enqueue_item =         2, str(q) = 'deque([1, 1, 2])'
i = 9, enqueue_item =         2, str(q) = 'deque([1, 1, 2, 2])'
```

# 참고 문헌

[Queue - Data Structures & Algorithms Tutorials In Python #8](https://www.youtube.com/watch?v=rUUrmGKYwHw)

## 연습문제

**연습문제 1.**
머리 포인터와 꼬리 포인터를 갖는 단일 연결 리스트로 큐를 구현하라.

??? success "연습문제 1 풀이"
    ```python
    class LinkedQueue:
        def __init__(self):
            self.head = self.tail = None
        def enqueue(self, val):
            node = Node(val)
            if self.tail:
                self.tail.next = node
            self.tail = node
            if not self.head:
                self.head = node
        def dequeue(self):
            val = self.head.val
            self.head = self.head.next
            if not self.head:
                self.tail = None
            return val
    ```

---

**연습문제 2.**
연결 리스트 큐에서 넣기와 빼기의 시간 복잡도는 얼마인가?

??? success "연습문제 2 풀이"
    둘 다 $O(1)$이다. 넣기는 꼬리에 덧붙이고(포인터 갱신), 빼기는 머리에서 없앤다(포인터 갱신). 옮기거나 크기를 다시 잡을 필요가 없다.

---

**연습문제 3.**
머리 포인터와 꼬리 포인터가 모두 필요한 이유는 무엇인가?

??? success "연습문제 3 풀이"
    꼬리 포인터가 없으면 넣을 때 끝을 찾으려고 리스트 전체를 훑어야 하므로 $O(n)$이 든다. 꼬리 포인터를 하나 더 관리하는 대가로 넣기를 $O(1)$에 할 수 있다.

---

**연습문제 4.**
원소가 $n$개일 때 연결 리스트 큐와 원형 배열 큐의 메모리 사용을 비교하라.

??? success "연습문제 4 풀이"
    연결 리스트는 노드가 $n$개이고 각각 데이터와 포인터를 가지므로 $n(d + 8)$바이트이다. 원형 배열은 용량이 $c \geq n$인 이어진 블록 하나로 $cd$바이트에 $O(1)$의 메타데이터를 더 쓴다. 크기가 정해진 자료형에서는 배열이 메모리 효율과 캐시 친화성 모두에서 낫다.
