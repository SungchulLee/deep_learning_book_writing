"""
Power Computation
"""

def compute_power_using_recursion(x, n):
    if n == 0:
        return 1
    return x * compute_power_using_recursion(x, n-1)


def compute_power_using_python(x, n):
    return x ** n


def main():
    x = 2
    n = 10
    for i in range(n):
        result_0 = compute_power_using_recursion(x, i)
        result_1 = compute_power_using_python(x, i)
        print(f"Computation of {x}^{i} using recursion : {result_0}")
        print(f"Computation of {x}^{i} using python    : {result_1}")
        print()
        
        
if __name__ == "__main__":
    main()
