# Memoization (Top-Down)

$$\begin{array}{llll}
\text{Recursion}&&\text{No Dynamic Programming (No Memoization or Tabulation)}\\
\text{Top Down}&&\text{Dynamic Programming (Memoization)}\\
\text{Bottom Up}&&\text{Dynamic Programming (Tabulation)}
\end{array}$$

$$

a_n=a_{n-1}+a_{n-2}

$$

where

$$

a_0=0,\quad a_1=1

$$

```python
import matplotlib.pyplot as plt
import time
```

```python
def fib(n):
    if n <= 1: 
        return n
    else:
        return fib(n-1) + fib(n-2)
```

```python
def memoize(f):
    memo = {} # dict
    def g(x):
        if x not in memo:
            memo[x] = f(x)
        return memo[x]
    return g
```

```python
fib = memoize(fib)
```

```python
for i in range(1,10):
    print(fib(i), end='\t')
```

**Output:**
```
1	2	3	5	8	13	21	34	55
```

```python
time_record = []
for i in range(200):
    tic = time.time()
    fib(i)
    toc = time.time()
    time_record.append(toc-tic)
    
plt.plot(time_record);
```

**Output:**
```
<Figure size 432x288 with 1 Axes>
```

# Reference

[python-practice-book](https://anandology.com/python-practice-book/functional-programming.html#recursion)

Abdul_Bari [youtube](https://www.youtube.com/watch?v=5dRGRueKU3M&list=PLDN4rrl48XKpZkf03iYFl-O29szjTrs_O&index=46)

[894. All Possible Full Binary Trees](https://leetcode.com/problems/all-possible-full-binary-trees/)
