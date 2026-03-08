# Deque as Queue

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

**Output:**
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

# Reference

[Python QUEUEs | Queue implementation example](https://www.youtube.com/watch?v=XLXWidXVRJk&list=PLj8W7XIvO93qsmdxbaDpIvM1KCyNO1K_c&index=1)
