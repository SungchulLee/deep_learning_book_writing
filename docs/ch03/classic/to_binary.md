# Change to Binary

<img src="img/Screen Shot 2022-05-01 at 8.26.20 PM.png" width=70%>

```python
def change_to_binary_and_print_using_recursion(num):
    if num <= 1:
        print(num, end='') 
        return
    
    change_to_binary_and_print_using_recursion(num//2)
    print(num%2, end='') 

def change_to_binary_and_print_using_python(num):
    print(bin(num)[2:])

def main():
    nums = (
    0,
    10,
    2,
    25,
    1256
    )
    for num in nums:
        print(f"Print number as binary using recursion : ", end='')
        change_to_binary_and_print_using_recursion(num)
        print()
        
        print(f"Print number as binary using python    : ", end='')
        change_to_binary_and_print_using_python(num)
        print()

        
if __name__ == "__main__":
    main()
```

**Output:**
```
Print number as binary using recursion : 0
Print number as binary using python    : 0

Print number as binary using recursion : 1010
Print number as binary using python    : 1010

Print number as binary using recursion : 10
Print number as binary using python    : 10

Print number as binary using recursion : 11001
Print number as binary using python    : 11001

Print number as binary using recursion : 10011101000
Print number as binary using python    : 10011101000
```

# Reference

[[알고리즘] 제1-2강 Recursion의 개념과 기본 예제들 (2/3)](https://www.youtube.com/watch?v=tuzf1yLPgRI&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=2)
