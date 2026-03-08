# Array-Based Stack

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

**Output:**
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

# Reference

[Python STACKs | example implementing a stack using Lists](https://www.youtube.com/watch?v=jwcnSv9tGdY&list=PLj8W7XIvO93qsmdxbaDpIvM1KCyNO1K_c&index=2)
