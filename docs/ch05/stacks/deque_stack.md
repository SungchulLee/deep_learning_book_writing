# 스택으로 쓰는 덱
```python
import numpy as np; np.random.seed(0)
from collections import deque

def main():
    s = []

    for i in range(10):
        if np.random.uniform() < 0.5:
            push_item = np.random.randint(low=1, high=7)
            s.append(push_item)
            print(f'{i = }, {push_item = :3}, {s = }')
        else:
            try:
                pop_item = s.pop()
            except IndexError:
                pop_item = str(None)
            print(f'{i = }, {pop_item = :4}, {s = }')
        
        
if __name__ == "__main__":
    main()
```

**출력:**
```
i = 0, pop_item = None, s = []
i = 1, pop_item = None, s = []
i = 2, pop_item = None, s = []
i = 3, pop_item = None, s = []
i = 4, push_item =   6, s = [6]
i = 5, push_item =   1, s = [6, 1]
i = 6, pop_item =    1, s = [6]
i = 7, push_item =   1, s = [6, 1]
i = 8, push_item =   2, s = [6, 1, 2]
i = 9, push_item =   2, s = [6, 1, 2, 2]
```

# 참고 문헌

[Stack - Data Structures & Algorithms Tutorial In Python #7](https://www.youtube.com/watch?v=zwb3GmNAtFk)

## 연습문제

**연습문제 1.**
파이썬에서 스택을 구현할 때 `collections.deque`이 좋은 선택인 이유는 무엇인가?

??? success "연습문제 1 풀이"
    `deque`은 `list`처럼 오른쪽 끝에서 `append`과 `pop`을 $O(1)$에 하면서, `appendleft`과 `popleft`도 $O(1)$에 한다. 고정 크기 블록의 이중 연결 리스트로 구현되어 캐시 성능도 좋다.

---

**연습문제 2.**
스택 연산에서 `deque.pop()`과 `list.pop()`의 성능을 비교하라.

??? success "연습문제 2 풀이"
    둘 다 상각 $O(1)$이다. 순수한 스택 연산만 한다면 메모리 배치가 단순한 `list`이 조금 더 빠를 수 있다. 양 끝에서 모두 빠른 연산이 필요할 때(예: 스택과 큐의 동작을 함께 쓸 때)에는 `deque`이 낫다.

---

**연습문제 3.**
`get_min()`도 $O(1)$ 시간에 지원하는 스택을 `deque`으로 구현하라.

??? success "연습문제 3 풀이"
    ```python
    from collections import deque
    class MinStack:
        def __init__(self):
            self.stack = deque()
            self.min_stack = deque()
        def push(self, val):
            self.stack.append(val)
            self.min_stack.append(min(val, self.min_stack[-1] if self.min_stack else val))
        def pop(self):
            self.min_stack.pop()
            return self.stack.pop()
        def get_min(self):
            return self.min_stack[-1]
    ```

---

**연습문제 4.**
스택으로 쓸 때 `deque`은 `list`에 견주어 공간을 얼마나 더 쓰는가?

??? success "연습문제 4 풀이"
    `deque`은 블록에 대한 포인터와 블록 메타데이터를 저장하여 블록마다 약 64바이트를 더 쓴다(블록 하나에 항목이 약 64개 들어간다). 작은 스택(항목 64개 미만)에서는 `list`이 메모리를 덜 쓴다. 큰 스택에서는 차이가 무시할 만하다.
