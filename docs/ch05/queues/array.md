# Array-Based Queue

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

**Output:**
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

# Reference

[Queue - Data Structures & Algorithms Tutorials In Python #8](https://www.youtube.com/watch?v=rUUrmGKYwHw)
