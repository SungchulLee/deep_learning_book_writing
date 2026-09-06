# 끼워넣기 정렬
<img src="img/Screen Shot 2022-04-30 at 1.16.53 PM.png" width="50%">
<img src="img/Screen Shot 2022-04-30 at 1.20.24 PM.png" width="50%">

Insertion Sort는 가장 먼저 생각할 수 있는 정렬 알고리즘이다.

```python
def insert(lst, item):
    """
    입력  : lst  : 정렬된 리스트
             item : 주어진 정렬된 리스트에 끼워 넣을 항목
    출력 : out  : lst + [item]의 새 정렬된 리스트
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
    입력  : lst : 정렬되지 않은 리스트
    출력 : out : 주어진 정렬되지 않은 리스트의 정렬된 리스트
    """ 
    
    # 바닥 경우
    if (len(lst)<=1):
        out = lst
        return out
    if (len(lst)==2) and (lst[0]<=lst[1]):
        out = lst
        return out
    if (len(lst)==2) and (lst[0]>lst[1]):
        out = lst[::-1]
        return out
    
    # 되돌이
    lst[:-1] = insertion_sort(lst[:-1])
    out = insert(lst[:-1], lst[-1])
    return out
```

```python
lst = [-1, 3, 8, -5, 7, 4, 10]
print(insertion_sort(lst))
```

**출력:**
```
[-5, -1, 3, 4, 7, 8, 10]
```

<div align="center"><img src="https://gmlwjd9405.github.io/images/algorithm-insertion-sort/sort-time-complexity.png" width="50%"></div>

[[알고리즘] 삽입 정렬(Insertion Sort)이란](https://gmlwjd9405.github.io/2018/05/06/algorithm-insertion-sort.html)

# 참고 문헌

[[알고리즘] 제3강 기본적인 정렬 알고리즘](https://www.youtube.com/watch?v=0dG7xTt5IfQ&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=9)

[3. Insertion Sort, Merge Sort](https://www.youtube.com/watch?v=Kg4bqzAqRBM&list=PLUl4u3cNGP61Oq3tWYp6V_F-5jb5L2iHb&index=3)

[Insertion Sort - Data Structures & Algorithms Tutorial Python #16](https://www.youtube.com/watch?v=K0zTIF3rm9s&list=PLeo1K3hjS3uu_n_a__MI_KktGTLYopZ12&index=17)

[[알고리즘] 삽입 정렬(Insertion Sort)이란](https://gmlwjd9405.github.io/2018/05/06/algorithm-insertion-sort.html)

[정렬 알고리즘](https://namu.wiki/w/정렬%20알고리즘#s-2.3.1)

## 연습문제

**연습문제 1.**
끼워넣기 정렬이 거의 정렬된 배열에서 왜 효율적인지 설명하라.

??? success "연습문제 1 풀이"
    끼워넣기 정렬의 안쪽 되돌이는 맞는 자리를 찾을 때까지 원소를 밀어낸다. 거의 정렬된 데이터에서는 원소마다 밀어내기가 몇 번 안 된다. 원소마다 정렬된 자리에서 많아야 $k$칸 떨어져 있다면 전체 일은 $O(nk)$이고, $k$이 상수이면 $O(n)$이다.

---

**연습문제 2.**
파이썬으로 끼워넣기 정렬을 구현하라.

??? success "연습문제 2 풀이"
    ```python
    def insertion_sort(arr):
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > key:
                arr[j+1] = arr[j]
                j -= 1
            arr[j+1] = key
    ```

---

**연습문제 3.**
팀 정렬 같은 섞은 정렬 알고리즘에서 끼워넣기 정렬을 바닥 경우로 쓰는 까닭은 무엇인가?

??? success "연습문제 3 풀이"
    작은 부분 배열에서는(대개 $n \leq 32$) 끼워넣기 정렬의 낮은 짐과 좋은 캐시 지역성 덕분에, 함수 부르기와 병합에서 상수 인자가 큰 병합 정렬 같은 되돌이 알고리즘보다 빠르다.

---

**연습문제 4.**
끼워넣기 정렬과 고르기 정렬을 견주어라. 어느 쪽이 더 나으며 왜 그런가?

??? success "연습문제 4 풀이"
    둘 다 $O(n^2)$이다. 끼워넣기 정렬: 적응한다(거의 정렬된 데이터에서 빠르다), 안정적이며 최선의 경우 $O(n)$이다. 고르기 정렬: 늘 $O(n^2)$이고 (표준판은) 안정적이지 않으며 맞바꿈을 가장 적게 한다($n-1$번). 쓰기를 줄이는 것이 결정적이지 않다면 대체로 끼워넣기 정렬이 낫다.
