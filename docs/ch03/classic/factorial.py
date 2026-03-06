"""
Factorial
"""

import math


# === Functions ===

def compute_factorial_using_recursion(n):
    if n == 0:
        return 1
    return n * compute_factorial_using_recursion(n-1)


def compute_factorial_using_package(n):
    return math.factorial(n)


def main():
    n = 10
    for i in range(n):
        result_0 = compute_factorial_using_recursion(i)
        result_1 = compute_factorial_using_package(i)
        print(f"Computation of {i}! using recursion : {result_0}")
        print(f"Computation of {i}! using package   : {result_1}")
        print()
        
        

# === Main ===
if __name__ == "__main__":
    main()
