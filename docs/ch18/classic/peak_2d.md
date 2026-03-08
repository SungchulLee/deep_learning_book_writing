# 2D Peak - Recursion

```python
import numpy as np
```

```python
def compare(left, center, right):
    if left > center: # 왼쪽이 중앙보다 크면 왼쪽 영역 (left 포함) 에서 찿는다.
        return 'left'
    elif right > center: # 오른쪽이 중앙보다 크면 오른쪽 영역 (right 포함) 에서 찿는다.
        return 'right'
    else: # 왼쪽과 오른쪽 모두 중앙보다 작거나 같으면 center가 찿고자하는 1D Peak 이다.
        return 'center'
```

```python
def peak_1d_max_peak(np_array_1d):
    """
    max_value : 주어진 리스트의 최대값
    max_index : 주어진 리스트에서 최대값이 발생하는 위치
    """
    lst = list(np_array_1d)
    for i, item in enumerate(lst):
        if i == 0:
            max_index = i
            max_value = item
        else:
            if item > max_value:
                max_index = i
                max_value = item
    return max_value, max_index
```

```python
def peak(np_array_2d):
    
    # Base Cases
    if np_array_2d.shape[1] == 1:
        max_value, _ = peak_1d_max_peak(np_array_2d.reshape(-1,))
        return max_value
    elif np_array_2d.shape[1] == 2:
        max_value_0, _ = peak_1d_max_peak(np_array_2d[:,0])
        max_value_1, _ = peak_1d_max_peak(np_array_2d[:,1])
        return max(max_value_0, max_value_1) 
    
    # Recursion
    i = int(np_array_2d.shape[1]/2) # center index
    max_value_i, max_index_i = peak_1d_max_peak(np_array_2d[:,i])
    left = np_array_2d[max_index_i, i-1]
    center = np_array_2d[max_index_i, i]
    right = np_array_2d[max_index_i, i+1]
    result = compare(left,center, right)
    
    if result == 'left':
        return peak(np_array_2d[:,:i]) 
    elif result == 'right':
        return peak(np_array_2d[:,i:])
    else:
        return center
```

```python
if 0: 
    lst = [
        [1],
        [2],
        [3],
        [4],
        [5]
    ]
elif 0:
    lst = [
        [1,10],
        [8,2],
        [3,7],
        [6,9],
        [4,5]
    ]
elif 0:
    lst = [
        [1,10,15],
        [8,11,2],
        [3,14,7],
        [12,6,9],
        [4,5,13]
    ]
elif 0:
    lst = [
        [16,1,10,15],
        [8,17,11,2],
        [3,19,14,7],
        [20,12,6,9],
        [4,5,13,18]
    ]
elif 0:
    lst = [
        [16,1,10,15,24],
        [8,17,11,21,2],
        [3,19,25,14,7],
        [20,22,12,6,9],
        [23,4,5,13,18]
    ]
elif 0:
    lst = [
        [16,1,10,15,24,26],
        [8,17,11,21,2,27],
        [3,19,25,14,7,28],
        [29,20,22,12,6,9],
        [23,4,5,13,18,30]
    ]
elif 0:
    lst = [
        [16,1,10,31,15,24,26],
        [8,34,17,11,21,2,27],
        [3,19,25,14,7,28,32],
        [29,20,22,12,6,9,35],
        [33,23,4,5,13,18,30]
    ]
elif 1:
    lst = [
        [16,1,10,31,15,24,26,39],
        [8,34,17,11,21,2,27,40],
        [3,19,36,25,14,7,28,32],
        [29,20,37,22,12,6,9,35],
        [33,38,23,4,5,13,18,30]
    ]
```

```python
np_array_2d = np.array(lst)
np_array_2d
```

**Output:**
```
array([[16,  1, 10, 31, 15, 24, 26, 39],
       [ 8, 34, 17, 11, 21,  2, 27, 40],
       [ 3, 19, 36, 25, 14,  7, 28, 32],
       [29, 20, 37, 22, 12,  6,  9, 35],
       [33, 38, 23,  4,  5, 13, 18, 30]])
```

```python
peak(np_array_2d)
```

**Output:**
```
21
```

# Reference

[1. Algorithmic Thinking, Peak Finding](https://www.youtube.com/watch?v=HtSuA80QTyo&list=PLUl4u3cNGP61Oq3tWYp6V_F-5jb5L2iHb&index=1)
