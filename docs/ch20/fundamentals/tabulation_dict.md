# Tabulation with Dict


$$\begin{array}{llll}
\text{Recursion}&&\text{No Dynamic Programming (No Memoization or Tabulation)}\\
\text{Top Down}&&\text{Dynamic Programming (Memoization)}\\
\text{Bottom Up}&&\text{Dynamic Programming (Tabulation)}
\end{array}$$


```python
import matplotlib.pyplot as plt
import numpy as np
import time


def fib_bottom_up(n):
    memo = {}
    for i in range(n+1):
        if i <= 1:
            memo[i] = i
        else:
            memo[i] = memo[i-1] + memo[i-2]
    return memo[n]


def main():    
    time_record = []
    for n in range(1,100):
        tic = time.time()
        fib_bottom_up(n)
        toc = time.time()
        time_record.append(toc-tic)

    plt.plot(np.arange(1,100), time_record) 
    plt.show()

    
if __name__ == "__main__":
    main()
```

**Output:**
```
<Figure size 432x288 with 1 Axes>
```


# Reference

CS_Dojo 
[youtube](https://www.youtube.com/watch?v=B0NtAFf4bvU&index=6&list=PLBZBJbE_rGRV8D7XZ08LK6z-4zPoWzu5H)
[youtube](https://www.youtube.com/watch?v=5o-kdjv7FD0)

Telusko 
[youtube](https://www.youtube.com/watch?v=gfhtaP5Wq7M&index=39&list=PLsyeobzWxl7poL9JTVyndKe62ieoN-MZ3)
[youtube](https://www.youtube.com/watch?v=XkL3SUioNvo&list=PLsyeobzWxl7poL9JTVyndKe62ieoN-MZ3&index=40)
[youtube](https://www.youtube.com/watch?v=TqqQld6m6A0&list=PLsyeobzWxl7poL9JTVyndKe62ieoN-MZ3&index=41)

Abdul_Bari [youtube](https://www.youtube.com/watch?v=5dRGRueKU3M&list=PLDN4rrl48XKpZkf03iYFl-O29szjTrs_O&index=46)
