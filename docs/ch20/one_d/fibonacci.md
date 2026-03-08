# Fibonacci

$$\begin{array}{llll}
\text{Recursion}&&\text{No Dynamic Programming (No Memoization or Tabulation)}\\
\text{Top Down}&&\text{Dynamic Programming (Memoization)}\\
\text{Bottom Up}&&\text{Dynamic Programming (Tabulation)}
\end{array}$$

$$

a_n=a_{n-1}+a_{n-2},\quad
a_0=0,\quad a_1=1

$$

<div align="center"><img src="https://i.stack.imgur.com/CLwKE.jpg" width="60%"></div>

[Fibonacci Numbers](https://riptutorial.com/algorithm/example/31799/fibonacci-numbers)

```python
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
```

# Reference

CS_Dojo [youtube](https://www.youtube.com/watch?v=vYquumk4nWw) [code](https://www.csdojo.io/dpcode)
[youtube](https://www.youtube.com/watch?v=nqlNzOcnCfs) [code](https://www.csdojo.io/dpcode)
[youtube](https://www.youtube.com/watch?v=D6xkbGLQesk&list=PLBZBJbE_rGRV8D7XZ08LK6z-4zPoWzu5H&index=7)

Telusko
[youtube](https://www.youtube.com/watch?v=gfhtaP5Wq7M&index=39&list=PLsyeobzWxl7poL9JTVyndKe62ieoN-MZ3)
[youtube](https://www.youtube.com/watch?v=XkL3SUioNvo&list=PLsyeobzWxl7poL9JTVyndKe62ieoN-MZ3&index=40)
[youtube](https://www.youtube.com/watch?v=TqqQld6m6A0&list=PLsyeobzWxl7poL9JTVyndKe62ieoN-MZ3&index=41)

Abdul_Bari [youtube](https://www.youtube.com/watch?v=5dRGRueKU3M&list=PLDN4rrl48XKpZkf03iYFl-O29szjTrs_O&index=46)

[Fibonacci Numbers](https://riptutorial.com/algorithm/example/31799/fibonacci-numbers)
