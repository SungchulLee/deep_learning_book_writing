# Top-Down


Merge Sort는 두 개의 정렬된 데이타를 합병하는 작업을 리커시브하게 진행한다.


```python
def merge_sort(arr):
    if len(arr)<=1:
        return
    
    mid = len(arr) // 2
    left = arr[:mid] 
    right = arr[mid:] 

    merge_sort(left)
    merge_sort(right)

    merge_sort_two_sorted_arrays(left, right, arr)
```


```python
def merge_sort_two_sorted_arrays(left, right, arr):
    len_left = len(left) 
    len_right = len(right) 
    i = j = k = 0
    while (i<len_left) and (j<len_right):
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1 
        else:
            arr[k] = right[j]
            j += 1 
        k += 1
        
    while (i<len_left): 
        arr[k] = left[i]
        i += 1
        k += 1
        
    while (j<len_right): 
        arr[k] = right[j]
        j += 1
        k += 1
```


```python
if __name__ == '__main__':
    test_cases = (
        [],
        [3],
        [3,5],
        [5,3],
        [3,5,2],
        [5,3,4],
        [10,3,15,7,8,23,98,29],
        [9,8,7,2],
        [1,2,3,4,5]
    )
    for arr in test_cases:
        merge_sort(arr)
        print(arr)
```

**Output:**
```
[]
[3]
[3, 5]
[3, 5]
[2, 3, 5]
[3, 4, 5]
[3, 7, 8, 10, 15, 23, 29, 98]
[2, 7, 8, 9]
[1, 2, 3, 4, 5]
```


<div align="center"><img src="https://gmlwjd9405.github.io/images/algorithm-merge-sort/sort-time-complexity.png" width="50%"></div>

[[알고리즘] 합병 정렬(merge sort)이란](https://gmlwjd9405.github.io/2018/05/08/algorithm-merge-sort.html)


# Reference

[Merge Sort vs Quick Sort](https://www.youtube.com/watch?v=es2T6KY45cA)

[Merge Sort - Data Structures & Algorithms Tutorial Python #17](https://www.youtube.com/watch?v=nCNfu_zNhyI&list=PLeo1K3hjS3uu_n_a__MI_KktGTLYopZ12&index=18)

[[알고리즘] 합병 정렬(merge sort)이란](https://gmlwjd9405.github.io/2018/05/08/algorithm-merge-sort.html)

[정렬 알고리즘](https://namu.wiki/w/정렬%20알고리즘#s-2.3.1)
