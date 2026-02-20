"""
Memoization (Top-Down)
"""

import matplotlib.pyplot as plt
import time


def fib(n):
    if n <= 1: 
        return n
    else:
        return fib(n-1) + fib(n-2)


def memoize(f):
    memo = {} # dict
    def g(x):
        if x not in memo:
            memo[x] = f(x)
        return memo[x]
    return g


fib = memoize(fib)


for i in range(1,10):
    print(fib(i), end='\t')


time_record = []
for i in range(200):
    tic = time.time()
    fib(i)
    toc = time.time()
    time_record.append(toc-tic)
    
plt.plot(time_record);
