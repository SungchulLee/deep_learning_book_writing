# Bubble Sort for Dictionary

<div align="center"><img src="https://ww.namu.la/s/ee412a864c3bdcb6cf7077f8ef87e01d4353cf53e66d2a5f6b7def49d257d569a46c810b1b36b9924a495a697c60777bb82d25459c2cbb65e4a700c25351af9b23dd8a07eb23a649358fba3b8cf3403c142f52b3fac839bf3cf1733d70787ee1" width="30%"></div>

[정렬 알고리즘](https://namu.wiki/w/정렬%20알고리즘#s-2.3.1)

Bubble Sort는 맨 앞에서부터 차례대로 비교하여 버블(큰수)를 오른쪽으로 밀어서 위로 올리는 알고리즘이다.

```python
def swap(a, b, key):
    if a[key] < b[key]:
        return a, b, True # last return item is flag that input is already sorted
    else:
        return b, a, False # last return item is flag that input is already sorted
```

```python
def first_bubble_up(lst, key):
    flag = True
    for i in range(len(lst)-1):
        lst[i], lst[i+1], flag_temp = swap(lst[i], lst[i+1], key=key)
        flag = flag and flag_temp
    return lst, flag # last return item is flag that input is already sorted
```

```python
def bubble_sort(lst, key):
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
lst = [
        { 'name': 'mona',   'transaction_amount': 1000, 'device': 'iphone-10'},
        { 'name': 'dhaval', 'transaction_amount': 400,  'device': 'google pixel'},
        { 'name': 'kathy',  'transaction_amount': 200,  'device': 'vivo'},
        { 'name': 'aamir',  'transaction_amount': 800,  'device': 'iphone-8'},
    ]
#bubble_sort(lst,key='name')
#bubble_sort(lst,key='transaction_amount')
bubble_sort(lst,key='device')
```

**Output:**
```
[{'name': 'dhaval', 'transaction_amount': 400, 'device': 'google pixel'},
 {'name': 'mona', 'transaction_amount': 1000, 'device': 'iphone-10'},
 {'name': 'aamir', 'transaction_amount': 800, 'device': 'iphone-8'},
 {'name': 'kathy', 'transaction_amount': 200, 'device': 'vivo'}]
```

<div align="center"><img src="https://gmlwjd9405.github.io/images/algorithm-bubble-sort/sort-time-complexity.png" width="50%"></div>

[[알고리즘] 버블 정렬(Bubble Sort)이란](https://gmlwjd9405.github.io/2018/05/06/algorithm-bubble-sort.html)

# Reference

[Bubble Sort - Data Structures & Algorithms Tutorial Python #14](https://www.youtube.com/watch?v=ppmIOUIz4uI&list=PLeo1K3hjS3uu_n_a__MI_KktGTLYopZ12&index=15)

[Bubble-sort with Hungarian ("Csángó") folk dance](https://www.youtube.com/watch?v=lyZQPjUT5B4)

[[알고리즘] 버블 정렬(Bubble Sort)이란](https://gmlwjd9405.github.io/2018/05/06/algorithm-bubble-sort.html)

[정렬 알고리즘](https://namu.wiki/w/정렬%20알고리즘#s-2.3.1)
