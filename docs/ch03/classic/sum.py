"""
Sum
"""

import numpy as np


def compute_sum_using_recursion(n):
    if n == 0:
        return 0
    return n + compute_sum_using_recursion(n-1)


def compute_sum_using_package(n):
    return np.arange(n+1).sum()


def main():
    n = 10
    for i in range(n):
        result_0 = compute_sum_using_recursion(i)
        result_1 = compute_sum_using_package(i)
        print(f"Computation of sum from 0 to {i} using recursion : {result_0}")
        print(f"Computation of sum from 0 to {i} using package   : {result_1}")
        print()
        
        
if __name__ == "__main__":
    main()
