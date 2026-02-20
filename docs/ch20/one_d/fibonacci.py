"""
Fibonacci
"""

def a(n):
    
    # Base Case
    if n <= 1:
        return n
    
    # Recursive Case
    return a(n-1) + a(n-2)


def main():
    for n in range(1,10):
        print(a(n), end='\t')
        
        
if __name__ == "__main__":
    main()
