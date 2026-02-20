"""
Greatest Common Divisor
"""

import math


def compute_gcd_using_recursion(i, j):
    if j == 0:
        return i
    return compute_gcd_using_recursion(j, i%j)


def compute_gcd_using_while_loop(i, j):
    while j:
        i, j = j, i%j
    return i


def compute_gcd_using_package(i, j):
    return math.gcd(i, j)


def main():
    i_ = (6, 10, 25, 30, 364, 45246, 465344)
    j_ = (9, 10, 70, 12, 162, 41312, 413116)
    for i, j in zip(i_, j_):
        result_0 = compute_gcd_using_recursion(i, j)
        result_1 = compute_gcd_using_while_loop(i, j)
        result_2 = compute_gcd_using_package(i, j)
        print(f"Computation of gcd({i},{j}) using recursion  : {result_0}")
        print(f"Computation of gcd({i},{j}) using while loop : {result_1}")
        print(f"Computation of gcd({i},{j}) using package    : {result_2}")
        print()

    
if __name__ == "__main__":
    main()
