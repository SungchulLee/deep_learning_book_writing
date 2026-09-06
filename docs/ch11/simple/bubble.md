# 거품 정렬
<img src="img/Screen Shot 2022-04-30 at 11.43.25 AM.png" width="70%">

<img src="img/Screen Shot 2022-04-30 at 12.23.04 PM.png" width="50%">

```python
class Bubble_Sort:
    
    def __init__(self, lst):
        self.lst = lst.copy()
        self.last = len(self.lst) - 1 # 왼쪽의 끝, 왼쪽과 오른쪽의 경계
        self.idx = 0 # self.lst[self.idx]와 self.lst[self.idx+1]을 맞바꿀 자리의 첨자
        
    def move_bubble_once(self):
        if self.lst[self.idx] > self.lst[self.idx+1]:
            self.lst[self.idx], self.lst[self.idx+1] = self.lst[self.idx+1], self.lst[self.idx]
        self.idx += 1

    def move_bubble(self):
        while self.idx < self.last:
            self.move_bubble_once()
        self.last -= 1
        self.idx = 0
        
    def run(self):
        while self.last > 0:
            self.move_bubble()
        return self.lst

    
def main():
    lsts = [
        [],
        [1,2,3,4],
        [4,3,2,1],
        [1,3,2,4],
        [2,4,1,3],
        [1,2,3,4,4],
        [4,3,4,2,1],
        [1,4,3,4,2],
        [2,4,4,1,3],
    ]
    for lst in lsts:
        obj = Bubble_Sort(lst)
        sorted_lst = obj.run()
        print(lst, '--->', sorted_lst)

    
if __name__ == "__main__":
    main()
```

**출력:**
```
[] ---> []
[1, 2, 3, 4] ---> [1, 2, 3, 4]
[4, 3, 2, 1] ---> [1, 2, 3, 4]
[1, 3, 2, 4] ---> [1, 2, 3, 4]
[2, 4, 1, 3] ---> [1, 2, 3, 4]
[1, 2, 3, 4, 4] ---> [1, 2, 3, 4, 4]
[4, 3, 4, 2, 1] ---> [1, 2, 3, 4, 4]
[1, 4, 3, 4, 2] ---> [1, 2, 3, 4, 4]
[2, 4, 4, 1, 3] ---> [1, 2, 3, 4, 4]
```

<div align="center"><img src="https://gmlwjd9405.github.io/images/algorithm-bubble-sort/sort-time-complexity.png" width="50%"></div>

[[알고리즘] 버블 정렬(Bubble Sort)이란](https://gmlwjd9405.github.io/2018/05/06/algorithm-bubble-sort.html)

# 참고 문헌

[[알고리즘] 제3강 기본적인 정렬 알고리즘](https://www.youtube.com/watch?v=0dG7xTt5IfQ&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=9)

[Bubble Sort - Data Structures & Algorithms Tutorial Python #14](https://www.youtube.com/watch?v=ppmIOUIz4uI&list=PLeo1K3hjS3uu_n_a__MI_KktGTLYopZ12&index=15)

[Bubble-sort with Hungarian ("Csángó") folk dance](https://www.youtube.com/watch?v=lyZQPjUT5B4)

[[알고리즘] 버블 정렬(Bubble Sort)이란](https://gmlwjd9405.github.io/2018/05/06/algorithm-bubble-sort.html)

[정렬 알고리즘](https://namu.wiki/w/정렬%20알고리즘#s-2.3.1)

## 연습문제

**연습문제 1.**
거품 정렬의 최선, 평균, 최악의 경우 시간 복잡도는 무엇인가?

??? success "연습문제 1 풀이"
    최선: $O(n)$(이미 정렬되어 있고 일찍 멈추기를 쓸 때). 평균: $O(n^2)$. 최악: $O(n^2)$(거꾸로 정렬). 공간: $O(1)$(제자리).

---

**연습문제 2.**
일찍 멈추기로 다듬은 거품 정렬을 구현하라.

??? success "연습문제 2 풀이"
    ```python
    def bubble_sort(arr):
        n = len(arr)
        for i in range(n):
            swapped = False
            for j in range(n - 1 - i):
                if arr[j] > arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
                    swapped = True
            if not swapped:
                break
    ```

---

**연습문제 3.**
거품 정렬이 안정적임을 증명하라.

??? success "연습문제 3 풀이"
    거품 정렬은 왼쪽이 오른쪽보다 엄밀히 클 때($a[j] > a[j+1]$)만 이웃한 원소를 맞바꾼다. 같은 원소는 결코 맞바뀌지 않으므로 본디 상대 차례가 지켜진다. $\square$

---

**연습문제 4.**
거품 정렬은 단순한데도 왜 실전에서 거의 쓰이지 않는가?

??? success "연습문제 4 풀이"
    평균의 경우 $O(n^2)$은 큰 입력에 너무 느리다. 일찍 멈추기를 써도 아무렇게나 놓인 데이터에서는 성능이 나쁘다. 끼워넣기 정렬은 (마찬가지로 $O(n^2)$이지만) 상수 인자가 더 좋아 작은 배열에서는 그쪽이 낫다. 두루 쓰는 정렬로는 $O(n\log n)$ 알고리즘(병합 정렬, 빠른 정렬, 팀 정렬)이 표준이다.
