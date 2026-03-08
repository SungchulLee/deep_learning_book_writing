"""
Tabulation (Bottom-Up)
"""

import matplotlib.pyplot as plt
import numpy as np
import time


# === Functions ===

def fib_bottom_up(n):
    
    # Base Case
    if n <= 1:
        return n
    
    # Recursive Case
    bottom_up = [None] * (n+1)
    bottom_up[0] = 0
    bottom_up[1] = 1
    for i in range(2, n+1):
        bottom_up[i] = bottom_up[i-1] + bottom_up[i-2]
    return bottom_up[n]


def main():    
    time_record = []
    for n in range(1,100):
        tic = time.time()
        fib_bottom_up(n)
        toc = time.time()
        time_record.append(toc-tic)

    plt.plot(np.arange(1,100), time_record) 
    plt.show()

    

# === Main ===
if __name__ == "__main__":
    main()
