"""
Call Stack
"""

def f(n):
    # Base Case
    if n==0:
        return
    
    # Recursion Case
    print("H", end="")
    f(n+1)


def main():
    n = 9
    f(n)
    
    
if __name__ == "__main__":
    main()
