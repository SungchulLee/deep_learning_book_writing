"""
Efficient Memory Usage
"""

import matplotlib.pyplot as plt
import time


def fib(n):
    if n <= 1:
        return n
    a = 0
    b = 1
    for _ in range(2, n+1):
        c = a + b
        a, b = b, c
    return c


print(fib(10000))


time_record = []
for i in range(1000):
    tic = time.time()
    fib(i+1)
    toc = time.time()
    time_record.append(toc-tic)
    
plt.plot(time_record);
