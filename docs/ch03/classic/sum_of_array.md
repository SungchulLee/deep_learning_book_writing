# Sum of Array


<img src="img/Screen Shot 2022-05-01 at 8.42.09 PM.png" width=70%>


```python
def compute_sum_of_array_using_recursion(lst):
    if len(lst) == 0:
        return 0    
    return compute_sum_of_array_using_recursion(lst[:-1]) + lst[-1]


def compute_sum_of_array_using_python(lst):
    return sum(lst)


def main():
    lsts = (
    [],
    [0],
    [2,0,],
    [2,1,-1],
    [2,5,-2,4],
    [1,2,3,4,5,6,7,8,9,10]
    )
    for lst in lsts:
        result_0 = compute_sum_of_array_using_recursion(lst)
        result_1 = compute_sum_of_array_using_python(lst)
        print(f"Sum of list {lst} using recursion : {result_0}")
        print(f"Sum of list {lst} using python    : {result_1}")
        print()

        
if __name__ == "__main__":
    main()
```

**Output:**
```
Sum of list [] using recursion : 0
Sum of list [] using python    : 0

Sum of list [0] using recursion : 0
Sum of list [0] using python    : 0

Sum of list [2, 0] using recursion : 2
Sum of list [2, 0] using python    : 2

Sum of list [2, 1, -1] using recursion : 2
Sum of list [2, 1, -1] using python    : 2

Sum of list [2, 5, -2, 4] using recursion : 9
Sum of list [2, 5, -2, 4] using python    : 9

Sum of list [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] using recursion : 55
Sum of list [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] using python    : 55
```


# Reference

[[알고리즘] 제1-2강 Recursion의 개념과 기본 예제들 (2/3)](https://www.youtube.com/watch?v=tuzf1yLPgRI&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=2)
