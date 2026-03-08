# Base Case and Recursive Case

<img src="img/Screen Shot 2022-05-01 at 9.52.06 PM.png" width=50%>

```python
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
```

**Output:**
```
9    H
8    H
7    H
6    H
5    H
4    H
3    H
2    H
1    H
```

# Reference

[[알고리즘] 제1-1강 Recursion의 개념과 기본 예제들 (1/3)](https://www.youtube.com/watch?v=ln7AfppN7mY&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=1)
