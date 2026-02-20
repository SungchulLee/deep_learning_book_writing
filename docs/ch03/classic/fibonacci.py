"""
Fibonacci
"""

def compute_fibonacci_using_recursion(n):
    if n <= 1:
        return n
    return compute_fibonacci_using_recursion(n-1) + compute_fibonacci_using_recursion(n-2)


def main():
    n = 10
    
    print(f"fibonacci({0}) = {0}")
    print(f"fibonacci({1}) = {1}")
    for i in range(2,n):
        result = compute_fibonacci_using_recursion(i)
        print(f"fibonacci({i}) = {result}")
        
        
if __name__ == "__main__":
    main()
