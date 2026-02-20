# Shell Sort


<div align="center"><img src="http://stoimen.com/wp-content/uploads/2012/02/Shell-Sort.png" width="50%"></div>

[Computer Algorithms: Shell Sort](http://stoimen.com/2012/02/27/computer-algorithms-shell-sort/)


Shell Sort는 고정된 쉘 겝에 주어지는 다양한 쉘을 정렬하고, 겝을 점차적으로 1까지 줄이는 알고리즘이다.


```python
def insert(item, lst):
    pointer = 0
    while True:
        if start_value <= pivot_value:
            start_pointer += 1
            try:
                start_value = lst[start_pointer]
            except IndexError:
                start_pointer -= 1
                break       
        else:
            break    
    return start_pointer
```


```python
def end_pointer_move(end_pointer, lst):
    pivot_value = lst[0]
    end_value = lst[end_pointer]
    while True:
        if (end_value >= pivot_value) and (end_pointer>=2):
            end_pointer -= 1
            end_value = lst[end_pointer]      
        else:
            break    
    return end_pointer
```


```python
def swap_move(start_pointer, end_pointer, lst):
    lst[end_pointer], lst[start_pointer] = lst[start_pointer], lst[end_pointer]
    return lst
```


```python
def pivot_move(end_pointer, lst):
    lst[0], lst[end_pointer] = lst[end_pointer], lst[0]
    return lst
```


```python
def partition(lst):
    
    if (len(lst)==2) and (lst[0]<=lst[1]):
        left, center, right = [lst[0]], [lst[1]], []
        return left, center, right 
    if (len(lst)==2) and (lst[0]>lst[1]):
        left, center, right = [lst[1]], [lst[0]], []
        return left, center, right 
        
    start_pointer = 1 
    end_pointer = len(lst) - 1
    while start_pointer < end_pointer:
        start_pointer = start_pointer_move(start_pointer, lst)
        end_pointer = end_pointer_move(end_pointer, lst)
        if start_pointer < end_pointer: 
            lst = swap_move(start_pointer, end_pointer, lst)
        elif (start_pointer==end_pointer) and (start_pointer==1): 
            left, center, right = [], [lst[0]], lst[1:]
            break
        elif (start_pointer==end_pointer) and (start_pointer==len(lst)-1): 
            center = [lst[0]]
            left = lst[1:]
            right = []
            break
        else:
            lst = pivot_move(end_pointer, lst)
            left, center, right = lst[:end_pointer], [lst[end_pointer]], lst[start_pointer:]
            break
    return left, center, right
```


```python
def quick_sort(lst):
    
    if (len(lst)<=1):
        return lst
    if (len(lst)==2) and (lst[0]<=lst[1]):
        return lst
    if (len(lst)==2) and (lst[0]>lst[1]):
        return lst[::-1]
    
    left, center, right = partition(lst)
    if len(left)>=2:
        left = quick_sort(left)
    if len(right)>=2:
        right = quick_sort(right)
    return left + center + right
```


```python
lst = [-1, 3, 8, -5, 7, 4, 10]
print(quick_sort(lst))
```

**Output:**
```
[-5, -1, 3, 4, 7, 8, 10]
```


<div align="center"><img src="https://gmlwjd9405.github.io/images/algorithm-shell-sort/sort-time-complexity.png" width="50%"></div>

[[알고리즘] 셸 정렬(shell sort)이란](https://gmlwjd9405.github.io/2018/05/08/algorithm-shell-sort.html)


# Reference

[Shell Sort - Data Structures & Algorithms Tutorial Python #18](https://www.youtube.com/watch?v=VxNr9Vudp4Y&list=PLeo1K3hjS3uu_n_a__MI_KktGTLYopZ12&index=19)

[[알고리즘] 셸 정렬(shell sort)이란](https://gmlwjd9405.github.io/2018/05/08/algorithm-shell-sort.html)

[Computer Algorithms: Shell Sort](http://stoimen.com/2012/02/27/computer-algorithms-shell-sort/)
