# 사전을 위한 거품 정렬
<div align="center"><img src="https://ww.namu.la/s/ee412a864c3bdcb6cf7077f8ef87e01d4353cf53e66d2a5f6b7def49d257d569a46c810b1b36b9924a495a697c60777bb82d25459c2cbb65e4a700c25351af9b23dd8a07eb23a649358fba3b8cf3403c142f52b3fac839bf3cf1733d70787ee1" width="30%"></div>

[정렬 알고리즘](https://namu.wiki/w/정렬%20알고리즘#s-2.3.1)

Bubble Sort는 맨 앞에서부터 차례대로 비교하여 버블(큰수)를 오른쪽으로 밀어서 위로 올리는 알고리즘이다.

```python
def swap(a, b, key):
    if a[key] < b[key]:
        return a, b, True # 마지막으로 되돌리는 항목은 입력이 이미 정렬되었는지를 알리는 깃발이다
    else:
        return b, a, False # 마지막으로 되돌리는 항목은 입력이 이미 정렬되었는지를 알리는 깃발이다
```

```python
def first_bubble_up(lst, key):
    flag = True
    for i in range(len(lst)-1):
        lst[i], lst[i+1], flag_temp = swap(lst[i], lst[i+1], key=key)
        flag = flag and flag_temp
    return lst, flag # 마지막으로 되돌리는 항목은 입력이 이미 정렬되었는지를 알리는 깃발이다
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

**출력:**
```
[{'name': 'dhaval', 'transaction_amount': 400, 'device': 'google pixel'},
 {'name': 'mona', 'transaction_amount': 1000, 'device': 'iphone-10'},
 {'name': 'aamir', 'transaction_amount': 800, 'device': 'iphone-8'},
 {'name': 'kathy', 'transaction_amount': 200, 'device': 'vivo'}]
```

<div align="center"><img src="https://gmlwjd9405.github.io/images/algorithm-bubble-sort/sort-time-complexity.png" width="50%"></div>

[[알고리즘] 버블 정렬(Bubble Sort)이란](https://gmlwjd9405.github.io/2018/05/06/algorithm-bubble-sort.html)

# 참고 문헌

[Bubble Sort - Data Structures & Algorithms Tutorial Python #14](https://www.youtube.com/watch?v=ppmIOUIz4uI&list=PLeo1K3hjS3uu_n_a__MI_KktGTLYopZ12&index=15)

[Bubble-sort with Hungarian ("Csángó") folk dance](https://www.youtube.com/watch?v=lyZQPjUT5B4)

[[알고리즘] 버블 정렬(Bubble Sort)이란](https://gmlwjd9405.github.io/2018/05/06/algorithm-bubble-sort.html)

[정렬 알고리즘](https://namu.wiki/w/정렬%20알고리즘#s-2.3.1)

## 연습문제

**연습문제 1.**
이 알고리즘의 시간 복잡도는 얼마인가?

??? success "연습문제 1 풀이"
    시간 복잡도는 최악과 평균의 경우 $O(n^2)$, (입력이 이미 정렬되었고 조기 종료를 쓰는) 최선의 경우 $O(n)$이다. 제자리에서 정렬하므로 공간 복잡도는 $O(1)$이다.

---

**연습문제 2.**
입력 `[5, 3, 1, 4, 2]`에서 알고리즘을 따라가며 한 번 훑을 때마다의 상태를 보여라.

??? success "연습문제 2 풀이"
    1회: `[3, 1, 4, 2, 5]`. 2회: `[1, 3, 2, 4, 5]`. 3회: `[1, 2, 3, 4, 5]`. 4회: 자리바꿈이 없어 조기 종료한다.

---

**연습문제 3.**
이 정렬 알고리즘은 안정적인가? 그 까닭을 밝혀라.

??? success "연습문제 3 풀이"
    그렇다. 이 알고리즘은 왼쪽 원소가 오른쪽보다 엄밀히 클 때만 이웃한 원소를 맞바꾸므로 안정적이다. 같은 원소는 결코 맞바꾸지 않아 본디 상대 순서가 지켜진다.

---

**연습문제 4.**
실제로 이 알고리즘을 언제 쓰겠는가?

??? success "연습문제 4 풀이"
    더 복잡한 알고리즘의 짐이 점근적 이점을 넘어서는 아주 작은 배열($n < 20$)에서, 거의 일차에 가까운 성능을 내는 거의 정렬된 데이터에서, 또는 간단해서 가르치는 보기로 쓴다.
