# Read N Integers

<img src="img/Screen Shot 2022-05-01 at 8.58.12 PM.png" width=50%>

```python
class Read_N_Integers_Using_Recursion:    
    def __init__(self, n, scanner_in_generator):
        self.n = n
        self.generator = scanner_in_generator
        self.data = [None] * (n + 1)
    def read_n_integers_using_recursion(self, k):
        if k == 0:
            return 
        self.read_n_integers_using_recursion(k-1)
        self.data[k] = next(self.generator)    
    def run(self):
        self.read_n_integers_using_recursion(self.n)
        
        
def range_generator(n):
    for i in range(n):
        yield i

def main():
    n = 10
    scanner_in_generator = range_generator(n)
    
    obj = Read_N_Integers_Using_Recursion(n, scanner_in_generator)
    obj.run()
    print(obj.data)
    
        
if __name__ == "__main__":
    main()
```

**Output:**
```
[None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

# Reference

[[알고리즘] 제1-2강 Recursion의 개념과 기본 예제들 (2/3)](https://www.youtube.com/watch?v=tuzf1yLPgRI&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=2)
