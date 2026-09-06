# 큐로 쓰는 덱
```python
import numpy as np; np.random.seed(0)
from collections import deque

def main():
    q = deque()

    for i in range(10):
        if np.random.uniform() < 0.5:
            enqueue_item = np.random.randint(low=1, high=7)
            q.append(enqueue_item)
            print(f'{i = }, {enqueue_item = :4}, {q = }')
        else:
            try:
                dequeue_item = q.popleft()
            except IndexError:
                dequeue_item = str(None)
            print(f'{i = }, {dequeue_item = :4}, {q = }')
            
            
if __name__ == "__main__":
    main()
```

**출력:**
```
i = 0, dequeue_item = None, q = deque([])
i = 1, dequeue_item = None, q = deque([])
i = 2, dequeue_item = None, q = deque([])
i = 3, dequeue_item = None, q = deque([])
i = 4, enqueue_item =    6, q = deque([6])
i = 5, enqueue_item =    1, q = deque([6, 1])
i = 6, dequeue_item =    6, q = deque([1])
i = 7, enqueue_item =    1, q = deque([1, 1])
i = 8, enqueue_item =    2, q = deque([1, 1, 2])
i = 9, enqueue_item =    2, q = deque([1, 1, 2, 2])
```

# 참고 문헌

[Python QUEUEs | Queue implementation example](https://www.youtube.com/watch?v=XLXWidXVRJk&list=PLj8W7XIvO93qsmdxbaDpIvM1KCyNO1K_c&index=1)

## 연습문제

**연습문제 1.**
큐 연산에서 `list`보다 `collections.deque`을 선호하는 이유는 무엇인가?

??? success "연습문제 1 풀이"
    `deque`은 `append`과 `popleft`을 $O(1)$에 하므로 넣기와 빼기 모두 $O(1)$이다. `list.pop(0)`은 원소를 옮기므로 $O(n)$이다.

---

**연습문제 2.**
`maxlen` 매개변수를 쓰는 `deque`으로 크기가 제한된 큐를 구현하라.

??? success "연습문제 2 풀이"
    ```python
    from collections import deque
    q = deque(maxlen=5)
    for i in range(10):
        q.append(i)  # 가장 오래된 원소가 저절로 버려진다
    print(q)  # deque([5, 6, 7, 8, 9], maxlen=5)
    ```

---

**연습문제 3.**
여러 스레드를 쓰는 응용에서 `deque`과 `queue.Queue`을 비교하라.

??? success "연습문제 3 풀이"
    `deque`의 연산은 CPython에서 (GIL 덕분에) 원자적이지만 형식적으로 스레드에 안전하지는 않다. `queue.Queue`은 명시적인 잠금과 시간 제한이 있는 대기형 `get`/`put`을 제공하며 생산자-소비자 방식을 위해 설계되었다. 여러 스레드를 쓰는 코드에는 `Queue`을, 단일 스레드의 성능에는 `deque`을 쓰라.

---

**연습문제 4.**
파이썬 `deque`의 내부 자료 구조는 무엇인가?

??? success "연습문제 4 풀이"
    고정 크기 블록(하나에 항목 64개)의 이중 연결 리스트이다. 양 끝에서 $O(1)$ 연산을 제공하면서 블록 안에서는 알맞은 캐시 지역성을 유지한다. 임의 접근은 최악의 경우 $O(n)$이다.
