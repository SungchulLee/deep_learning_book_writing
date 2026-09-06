# 끼워넣기 정렬 - 제자리
<div align="center"><img src="https://w.namu.la/s/e2cca975b1e03bd676ae5e11433526429e9cf77953039ca19a2df4b1112eb75c9c45701ca4f75bcb78194f07ec7b60f28040a4bae7ceed58729887ff62fc13f69868eaa547d811e954217aa647befd21da0fdcf9fb1deb7689cd19dde0e9a7f9" width="30%"></div>

[정렬 알고리즘](https://namu.wiki/w/정렬%20알고리즘#s-2.3.1)

Insertion Sort는 가장 먼저 생각할 수 있는 정렬 알고리즘이다.

```python
def insertion_sort(lst):
    for right in range(1,len(lst)): 
        for left in range(right-1,-1,-1): 
            if lst[left] > lst[right]:
                #print(lst, "--->", lst[left], lst[right], "--->", end=' ')
                lst[left], lst[right] = lst[right], lst[left]
                #print(lst)
                right = left   
            else:
                break        
    return lst
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

[3. Insertion Sort, Merge Sort](https://www.youtube.com/watch?v=Kg4bqzAqRBM&list=PLUl4u3cNGP61Oq3tWYp6V_F-5jb5L2iHb&index=3)

[Insertion Sort - Data Structures & Algorithms Tutorial Python #16](https://www.youtube.com/watch?v=K0zTIF3rm9s&list=PLeo1K3hjS3uu_n_a__MI_KktGTLYopZ12&index=17)

[[알고리즘] 삽입 정렬(Insertion Sort)이란](https://gmlwjd9405.github.io/2018/05/06/algorithm-insertion-sort.html)

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
