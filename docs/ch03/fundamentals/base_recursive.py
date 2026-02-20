"""
Base Case and Recursive Case
"""

def f(n):
    # Base Case
    if n==0:
        return
    
    # Recursion Case
    print(f"{n}    H")
    f(n-1)


def main():
    n = 9
    f(n)
    
    
if __name__ == "__main__":
    main()
