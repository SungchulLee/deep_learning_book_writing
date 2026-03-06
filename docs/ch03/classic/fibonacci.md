# Fibonacci

<img src="img/Screen Shot 2022-05-01 at 7.22.49 PM.png" width=50%>

```python
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
```

**Output:**
```
fibonacci(0) = 0
fibonacci(1) = 1
fibonacci(2) = 1
fibonacci(3) = 2
fibonacci(4) = 3
fibonacci(5) = 5
fibonacci(6) = 8
fibonacci(7) = 13
fibonacci(8) = 21
fibonacci(9) = 34
```

# Reference

[[알고리즘] 제1-1강 Recursion의 개념과 기본 예제들 (1/3)](https://www.youtube.com/watch?v=ln7AfppN7mY&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=1)
