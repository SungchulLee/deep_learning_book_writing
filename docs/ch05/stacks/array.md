# 배열 기반 스택
```python
import numpy as np; np.random.seed(0)

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

[Python STACKs | example implementing a stack using Lists](https://www.youtube.com/watch?v=jwcnSv9tGdY&list=PLj8W7XIvO93qsmdxbaDpIvM1KCyNO1K_c&index=2)

## 연습문제

**연습문제 1.**
배열 기반 스택에서 push, pop, peek의 시간 복잡도는 얼마인가?

??? success "연습문제 1 풀이"
    세 연산 모두 상각 $O(1)$이다. push와 pop은 배열의 마지막 원소에 작용한다. 배열의 크기를 다시 잡는 드문 경우에 push는 최악의 경우 $O(n)$이지만 상각으로는 $O(1)$이다.

---

**연습문제 2.**
최대 용량이 있고 넘치면 오류를 일으키는 배열 기반 스택을 구현하라.

??? success "연습문제 2 풀이"
    ```python
    class BoundedStack:
        def __init__(self, capacity):
            self._data = [None] * capacity
            self._top = -1
        def push(self, val):
            if self._top == len(self._data) - 1:
                raise OverflowError('Stack full')
            self._top += 1
            self._data[self._top] = val
        def pop(self):
            if self._top == -1:
                raise IndexError('Stack empty')
            val = self._data[self._top]
            self._top -= 1
            return val
    ```

---

**연습문제 3.**
파이썬의 `list`가 스택으로는 잘 통하지만 큐로는 그렇지 못한 이유는 무엇인가?

??? success "연습문제 3 풀이"
    `list.append()`과 `list.pop()`은 (끝에서 하는 스택 연산이므로) $O(1)$이다. 그러나 `list.pop(0)`은 남은 원소를 모두 왼쪽으로 옮겨야 하므로 $O(n)$이다. 큐에는 양 끝에서 $O(1)$ 연산을 제공하는 `collections.deque`을 쓰라.

---

**연습문제 4.**
배열로 구현한 스택과 연결 리스트로 구현한 스택의 메모리 배치를 비교하라.

??? success "연습문제 4 풀이"
    배열은 메모리가 이어져 있어 캐시에 친화적이고 임의 접근이 $O(1)$이지만, 크기를 다시 잡을 때 모든 원소를 복사한다. 연결 리스트는 메모리가 흩어져 있어 캐시에 불리하고 임의 접근이 안 되지만 크기를 다시 잡을 필요가 없다. 대부분의 쓰임에서는 캐시 지역성 덕분에 배열 기반 스택이 더 빠르다.
