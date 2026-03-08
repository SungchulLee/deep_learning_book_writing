"""
Exponential Computation Time
"""

import matplotlib.pyplot as plt
import numpy as np
import time


# === Functions ===

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
    
    

# === Main ===
if __name__ == "__main__":
    main()
