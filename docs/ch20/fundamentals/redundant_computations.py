"""
Redundant Computations
"""

count = 0

def a(n):
    
    global count
    
    # Base Case
    if n <= 1:
        print(f'a({n}) : {n}')
        count += 1
        return n
    
    # Recursive Case
    print(f'a({n}) : {a(n-1) + a(n-2)}')
    count += 1
    return a(n-1) + a(n-2)


def main():
    n = 5
    a(n)
    print(f'Number of Function Calls : {count}')


# === Main ===
if __name__ == "__main__":
    main()
