# Bubble Sort with Key


<div align="center"><img src="https://ww.namu.la/s/ee412a864c3bdcb6cf7077f8ef87e01d4353cf53e66d2a5f6b7def49d257d569a46c810b1b36b9924a495a697c60777bb82d25459c2cbb65e4a700c25351af9b23dd8a07eb23a649358fba3b8cf3403c142f52b3fac839bf3cf1733d70787ee1" width="30%"></div>

[정렬 알고리즘](https://namu.wiki/w/정렬%20알고리즘#s-2.3.1)


Bubble Sort는 맨 앞에서부터 차례대로 비교하여 버블(큰수)를 오른쪽으로 밀어서 위로 올리는 알고리즘이다.


```python
def swap(a, b, key=lambda x: x):
    if key(a) < key(b):
        return a, b, True # last return item is flag that input is already sorted
    else:
        return b, a, False # last return item is flag that input is already sorted
```


```python
def first_bubble_up(lst, key=lambda x: x):
    flag = True
    for i in range(len(lst)-1):
        lst[i], lst[i+1], flag_temp = swap(lst[i], lst[i+1], key=key)
        flag = flag and flag_temp
    return lst, flag # last return item is flag that input is already sorted
```


```python
def bubble_sort(lst, key=lambda x: x):
    for i in range(len(lst)-1):
        if i == 0:
            lst, flag = first_bubble_up(lst, key=key)
        else:
            lst[:-i], flag = first_bubble_up(lst[:-i], key=key)
        if flag:
            break
    return lst
```


```python
lst = [3,-5,2,-7,9]
bubble_sort(lst,key=abs)
```

**Output:**
```
[2, 3, -5, -7, 9]
```


<div align="center"><img src="https://gmlwjd9405.github.io/images/algorithm-bubble-sort/sort-time-complexity.png" width="50%"></div>

[[알고리즘] 버블 정렬(Bubble Sort)이란](https://gmlwjd9405.github.io/2018/05/06/algorithm-bubble-sort.html)


# Reference

[Bubble Sort - Data Structures & Algorithms Tutorial Python #14](https://www.youtube.com/watch?v=ppmIOUIz4uI&list=PLeo1K3hjS3uu_n_a__MI_KktGTLYopZ12&index=15)

[Bubble-sort with Hungarian ("Csángó") folk dance](https://www.youtube.com/watch?v=lyZQPjUT5B4)

[[알고리즘] 버블 정렬(Bubble Sort)이란](https://gmlwjd9405.github.io/2018/05/06/algorithm-bubble-sort.html)

[정렬 알고리즘](https://namu.wiki/w/정렬%20알고리즘#s-2.3.1)
