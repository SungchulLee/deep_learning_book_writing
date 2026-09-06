# 셸 정렬
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

**출력:**
```
[-5, -1, 3, 4, 7, 8, 10]
```

<div align="center"><img src="https://gmlwjd9405.github.io/images/algorithm-shell-sort/sort-time-complexity.png" width="50%"></div>

[[알고리즘] 셸 정렬(shell sort)이란](https://gmlwjd9405.github.io/2018/05/08/algorithm-shell-sort.html)

# 참고 문헌

[Shell Sort - Data Structures & Algorithms Tutorial Python #18](https://www.youtube.com/watch?v=VxNr9Vudp4Y&list=PLeo1K3hjS3uu_n_a__MI_KktGTLYopZ12&index=19)

[[알고리즘] 셸 정렬(shell sort)이란](https://gmlwjd9405.github.io/2018/05/08/algorithm-shell-sort.html)

[Computer Algorithms: Shell Sort](http://stoimen.com/2012/02/27/computer-algorithms-shell-sort/)

## 연습문제

**연습문제 1.**
셸 정렬의 간격 수열과 그것이 끼워넣기 정렬을 어떻게 일반화하는지 설명하라.

??? success "연습문제 1 풀이"
    셸 정렬은 줄어드는 간격 수열이 정하는 서로 엇갈린 부분 수열에 끼워넣기 정렬을 한다. 간격이 $g$이면 자리 $0, g, 2g, \ldots$의 원소를 함께 정렬한다. $g$이 1로 줄어들수록 배열은 점점 더 정렬된다. 마지막 훑기($g=1$)는 거의 정렬된 배열에 대한 보통의 끼워넣기 정렬이다.

---

**연습문제 2.**
여러 간격 수열과 그 최악의 경우 복잡도를 견주어라.

??? success "연습문제 2 풀이"
    셸의 본디 수열 $(n/2, n/4, \ldots, 1)$: $O(n^2)$. 크누스의 수열 $(1, 4, 13, 40, \ldots)$: $O(n^{3/2})$. 세지윅의 수열: $O(n^{4/3})$. 치우라의 실험 수열 $(1, 4, 10, 23, 57, 132, 301, 701)$: 실전에서 가장 빠르다.

---

**연습문제 3.**
크누스의 간격 수열로 셸 정렬을 구현하라.

??? success "연습문제 3 풀이"
    ```python
    def shell_sort(arr):
        n = len(arr)
        gap = 1
        while gap < n // 3:
            gap = gap * 3 + 1
        while gap > 0:
            for i in range(gap, n):
                temp = arr[i]
                j = i
                while j >= gap and arr[j - gap] > temp:
                    arr[j] = arr[j - gap]
                    j -= gap
                arr[j] = temp
            gap //= 3
    ```

---

**연습문제 4.**
셸 정렬의 정확한 복잡도는 왜 아직 풀리지 않은 문제인가?

??? success "연습문제 4 풀이"
    복잡도는 간격 수열에 달렸고, 간격과 입력 순열이 어우러지는 모습이 복잡하다. 대부분의 간격 수열에서는 빈틈없는 한계가 알려져 있지 않다. 알려진 가장 좋은 일반 한계는 특정 수열에 대한 $O(n\log^2 n)$이지만, $O(n\log n)$이 될 수 있는지는 아직 열려 있다.
