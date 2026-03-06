# Redundant Computations

$$\begin{array}{llll}
\text{Recursion}&&\text{No Dynamic Programming (No Memoization or Tabulation)}\\
\text{Top Down}&&\text{Dynamic Programming (Memoization)}\\
\text{Bottom Up}&&\text{Dynamic Programming (Tabulation)}
\end{array}$$

```python
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

if __name__ == "__main__":
    main()
```

**Output:**
```
a(1) : 1
a(0) : 0
a(2) : 1
a(1) : 1
a(0) : 0
a(1) : 1
a(3) : 2
a(1) : 1
a(0) : 0
a(2) : 1
a(1) : 1
a(0) : 0
a(1) : 1
a(1) : 1
a(0) : 0
a(2) : 1
a(1) : 1
a(0) : 0
a(4) : 3
a(1) : 1
a(0) : 0
a(2) : 1
a(1) : 1
a(0) : 0
a(1) : 1
a(3) : 2
a(1) : 1
a(0) : 0
a(2) : 1
a(1) : 1
a(0) : 0
a(1) : 1
a(1) : 1
a(0) : 0
a(2) : 1
a(1) : 1
a(0) : 0
a(1) : 1
a(0) : 0
a(2) : 1
a(1) : 1
a(0) : 0
a(1) : 1
a(3) : 2
a(1) : 1
a(0) : 0
a(2) : 1
a(1) : 1
a(0) : 0
a(1) : 1
a(5) : 5
a(1) : 1
a(0) : 0
a(2) : 1
a(1) : 1
a(0) : 0
a(1) : 1
a(3) : 2
a(1) : 1
a(0) : 0
a(2) : 1
a(1) : 1
a(0) : 0
a(1) : 1
a(1) : 1
a(0) : 0
a(2) : 1
a(1) : 1
a(0) : 0
a(4) : 3
a(1) : 1
a(0) : 0
a(2) : 1
a(1) : 1
a(0) : 0
a(1) : 1
a(3) : 2
a(1) : 1
a(0) : 0
a(2) : 1
a(1) : 1
a(0) : 0
a(1) : 1
a(1) : 1
a(0) : 0
a(2) : 1
a(1) : 1
a(0) : 0
a(1) : 1
a(0) : 0
a(2) : 1
a(1) : 1
a(0) : 0
a(1) : 1
a(3) : 2
a(1) : 1
a(0) : 0
a(2) : 1
a(1) : 1
a(0) : 0
a(1) : 1
Number of Function Calls : 101
```

# Reference

CS_Dojo 
[youtube](https://www.youtube.com/watch?v=B0NtAFf4bvU&index=6&list=PLBZBJbE_rGRV8D7XZ08LK6z-4zPoWzu5H)
[youtube](https://www.youtube.com/watch?v=5o-kdjv7FD0)
Telusko 
[youtube](https://www.youtube.com/watch?v=gfhtaP5Wq7M&index=39&list=PLsyeobzWxl7poL9JTVyndKe62ieoN-MZ3)
[youtube](https://www.youtube.com/watch?v=XkL3SUioNvo&list=PLsyeobzWxl7poL9JTVyndKe62ieoN-MZ3&index=40)
[youtube](https://www.youtube.com/watch?v=TqqQld6m6A0&list=PLsyeobzWxl7poL9JTVyndKe62ieoN-MZ3&index=41)
