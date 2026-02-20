# Insertion Sort


<img src="img/Screen Shot 2022-04-30 at 1.16.53 PM.png" width="50%">
<img src="img/Screen Shot 2022-04-30 at 1.20.24 PM.png" width="50%">


Insertion Sort는 가장 먼저 생각할 수 있는 정렬 알고리즘이다.


```python
def insert(lst, item):
    """
    input  : lst  : sorted list
             item : item to insert into the given sorted list
    output : out  : new sorted list of lst + [item]
    """
    pointer = len(lst) - 1 # 처음에는 포인터가 주어진 리스트의 맨 마지막 아이템을 포인팅한다.
    while True:
        if pointer >= 0:
            if lst[pointer] > item:
                pointer -= 1 # 포인터는 오른쪽에서 왼쪽으로 한칸씩 움직인다.
            else:
                out = lst[:pointer+1] + [item] + lst[pointer+1:]
                return out 
        else:
            out = [item] + lst
            return out
```


```python
def insertion_sort(lst):
    """
    input  : lst : unsorted list
    output : out : sorted list of the given unsorted list
    """ 
    
    # Base Cases
    if (len(lst)<=1):
        out = lst
        return out
    if (len(lst)==2) and (lst[0]<=lst[1]):
        out = lst
        return out
    if (len(lst)==2) and (lst[0]>lst[1]):
        out = lst[::-1]
        return out
    
    # Recursion 
    lst[:-1] = insertion_sort(lst[:-1])
    out = insert(lst[:-1], lst[-1])
    return out
```


```python
lst = [-1, 3, 8, -5, 7, 4, 10]
print(insertion_sort(lst))
```

**Output:**
```
[-5, -1, 3, 4, 7, 8, 10]
```


<div align="center"><img src="https://gmlwjd9405.github.io/images/algorithm-insertion-sort/sort-time-complexity.png" width="50%"></div>

[[알고리즘] 삽입 정렬(Insertion Sort)이란](https://gmlwjd9405.github.io/2018/05/06/algorithm-insertion-sort.html)


# Reference

[[알고리즘] 제3강 기본적인 정렬 알고리즘](https://www.youtube.com/watch?v=0dG7xTt5IfQ&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=9)

[3. Insertion Sort, Merge Sort](https://www.youtube.com/watch?v=Kg4bqzAqRBM&list=PLUl4u3cNGP61Oq3tWYp6V_F-5jb5L2iHb&index=3)

[Insertion Sort - Data Structures & Algorithms Tutorial Python #16](https://www.youtube.com/watch?v=K0zTIF3rm9s&list=PLeo1K3hjS3uu_n_a__MI_KktGTLYopZ12&index=17)

[[알고리즘] 삽입 정렬(Insertion Sort)이란](https://gmlwjd9405.github.io/2018/05/06/algorithm-insertion-sort.html)

[정렬 알고리즘](https://namu.wiki/w/정렬%20알고리즘#s-2.3.1)
