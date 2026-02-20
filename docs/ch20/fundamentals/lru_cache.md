# LRU Cache


$$\begin{array}{llll}
\text{Recursion}&&\text{No Dynamic Programming (No Memoization or Tabulation)}\\
\text{Top Down}&&\text{Dynamic Programming (Memoization)}\\
\text{Bottom Up}&&\text{Dynamic Programming (Tabulation)}
\end{array}$$


```python
import numpy as np
import time
import matplotlib.pyplot as plt
from functools import lru_cache as cache


@cache(maxsize=5)
def a(n):
    
    # Base Case
    if n <= 1:
        return n
    
    # Recursive Case
    return a(n-1) + a(n-2)


def main():
    time_record = []
    for n in range(1,25):
        tic = time.time()
        a(n)
        toc = time.time()
        time_record.append(toc-tic)
    plt.plot(np.arange(1,25), time_record) 
    plt.show()

    
if __name__ == "__main__":
    main()
```

**Output:**
```
<Figure size 432x288 with 1 Axes>
```


# Reference

[캐싱으로 Python 코드 가속화](https://www.youtube.com/watch?v=0_ievUF0L_E)

[python-practice-book](https://anandology.com/python-practice-book/functional-programming.html#recursion)

Abdul_Bari [youtube](https://www.youtube.com/watch?v=5dRGRueKU3M&list=PLDN4rrl48XKpZkf03iYFl-O29szjTrs_O&index=46)

[Caching in Python Using the LRU Cache Strategy](https://realpython.com/lru-cache-python/)
