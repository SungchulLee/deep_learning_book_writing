# 1D Peak - Naive


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
def compare_left_edge(center, right):
    if right > center: # 오른쪽이 중앙보다 크면 오른쪽 영역 (right 포함) 에서 찿는다.
        return 'right'
    else: # 왼쪽과 오른쪽 모두 중앙보다 작거나 같으면 center가 찿고자하는 1D Peak 이다.
        return 'center'
```


```python
def compare_right_edge(left, center):
    if left > center: # 왼쪽이 중앙보다 크면 왼쪽 영역 (left 포함) 에서 찿는다.
        return 'left'
    else: # 왼쪽과 오른쪽 모두 중앙보다 작거나 같으면 center가 찿고자하는 1D Peak 이다.
        return 'center'
```


```python
def peak(lst):
    
    # Base Cases
    if len(lst) == 0:
        return None
    elif len(lst) == 1:
        return lst[0] 
    
    # Naive Linear Search
    for i in range(len(lst)):
        if i == 0:
            center = lst[i]
            right = lst[i+1]
            result = compare_left_edge(center, right) 
        elif 0 < i < (len(lst)-1):
            left = lst[i-1]
            center = lst[i]
            right = lst[i+1]
            result = compare(left,center, right)
        else:
            left = lst[i-1]
            center = lst[i]
            result = compare_right_edge(left, center)
            
        if result == 'center':
            return center
```


```python
#lst = [1,2,3,4,5]
#lst = [5,4,3,2,1]
lst = [1,2,3,4,5,4,3,2,1]
peak(lst)
```


# Reference

[1. Algorithmic Thinking, Peak Finding](https://www.youtube.com/watch?v=HtSuA80QTyo&list=PLUl4u3cNGP61Oq3tWYp6V_F-5jb5L2iHb&index=1)
