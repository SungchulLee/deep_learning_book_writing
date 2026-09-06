# 연결 리스트 스택
```python
import numpy as np; np.random.seed(0)
from collections import deque

class Stack:
    def __init__(self):
        self.stack = deque()   
    def push(self,val):
        self.stack.append(val)       
    def pop(self):
        if len(self.stack) > 0:
            return self.stack.pop()
        else:
            return None
    def peek(self):
        if len(self.stack) > 0:
            return self.stack[-1]
        else:
            return None
    def is_empty(self):
        return len(self.stack)==0
    def size(self):
        return len(self.stack)
    def __str__(self):
        return str(self.stack)

def main():
    s = Stack()

    for i in range(10):
        if np.random.uniform() < 0.5:
            push_item = np.random.randint(low=1, high=7)
            s.push(push_item)
            print(f'{i = }, {push_item = :>8}, {str(s) = }')
        else:
            pop_item = s.pop()    
            print(f'{i = }, {str(pop_item) = :>4}, {str(s) = }')
            
            
if __name__ == "__main__":
    main()
```

**출력:**
```
i = 0, str(pop_item) = None, str(s) = 'deque([])'
i = 1, str(pop_item) = None, str(s) = 'deque([])'
i = 2, str(pop_item) = None, str(s) = 'deque([])'
i = 3, str(pop_item) = None, str(s) = 'deque([])'
i = 4, push_item =        6, str(s) = 'deque([6])'
i = 5, push_item =        1, str(s) = 'deque([6, 1])'
i = 6, str(pop_item) =    1, str(s) = 'deque([6])'
i = 7, push_item =        1, str(s) = 'deque([6, 1])'
i = 8, push_item =        2, str(s) = 'deque([6, 1, 2])'
i = 9, push_item =        2, str(s) = 'deque([6, 1, 2, 2])'
```

# 참고 문헌

[Python STACKs | example implementing a stack using Lists](https://www.youtube.com/watch?v=jwcnSv9tGdY&list=PLj8W7XIvO93qsmdxbaDpIvM1KCyNO1K_c&index=2)

## 연습문제

**연습문제 1.**
push, pop, peek 연산을 갖춘 스택을 단일 연결 리스트로 구현하라.

??? success "연습문제 1 풀이"
    ```python
    class Node:
        def __init__(self, val, next=None):
            self.val = val
            self.next = next

    class LinkedStack:
        def __init__(self):
            self.top = None
        def push(self, val):
            self.top = Node(val, self.top)
        def pop(self):
            val = self.top.val
            self.top = self.top.next
            return val
    ```

---

**연습문제 2.**
연결 리스트 스택이 배열 기반 스택보다 나은 점은 무엇인가?

??? success "연습문제 2 풀이"
    크기를 다시 잡을 필요가 없다. push가 (상각이 아니라) 최악의 경우에도 $O(1)$이다. 메모리를 원소마다 할당하므로 남아도는 용량이 없다. 단점은 캐시 지역성이 나쁘고 원소마다 메모리 부담(포인터 저장)이 크다는 것이다.

---

**연습문제 3.**
연결 리스트 스택도 넘칠 수 있는가? 어떤 뜻에서 그런가?

??? success "연습문제 3 풀이"
    고정 크기가 없으므로 용량 한계로 넘치지는 않지만, 원소를 너무 많이 넣으면 힙 메모리가 바닥날 수 있다. 파이썬에서는 각 노드가 별도의 객체이고 부담이 상당하므로(객체와 포인터를 합해 약 56바이트) 메모리가 동날 수 있다.

---

**연습문제 4.**
배열 기반, 연결 리스트, 덱 기반 스택에서 push와 pop의 최악의 경우 시간 복잡도를 비교하라.

??? success "연습문제 4 풀이"
    배열: push는 최악의 경우 $O(n)$(크기 재조정), 상각 $O(1)$이고 pop은 $O(1)$이다. 연결 리스트: push가 최악의 경우에도 $O(1)$이고 pop도 $O(1)$이다. 덱: push는 상각 $O(1)$, pop은 $O(1)$이다. 연결 리스트가 최악의 경우 보장은 가장 좋지만 캐시 성능은 가장 나쁘다.
