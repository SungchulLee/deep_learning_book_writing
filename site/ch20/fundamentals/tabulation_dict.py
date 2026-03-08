"""
Tabulation with Dict
"""

import matplotlib.pyplot as plt
import numpy as np
import time


# === Functions ===

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

    

# === Main ===
if __name__ == "__main__":
    main()
