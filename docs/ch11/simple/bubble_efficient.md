# 거품 정렬 - 효율판
<img src="img/Screen Shot 2022-04-30 at 11.43.25 AM.png" width="70%">

<img src="img/Screen Shot 2022-04-30 at 12.23.04 PM.png" width="50%">

```python
class Bubble_Sort:
    
    def __init__(self, lst):
        self.lst = lst.copy()
        self.last = len(self.lst) - 1 # 왼쪽의 끝, 왼쪽과 오른쪽의 경계
        self.idx = 0 # self.lst[self.idx]와 self.lst[self.idx+1]을 맞바꿀 자리의 첨자
        
        self.flag_move_bubble = False # 거품이 움직였는지 알리는 깃발
        self.flag_sorted = False # self.lst가 정렬되었는지 알리는 깃발
        
    def move_bubble_once(self):
        if self.lst[self.idx] > self.lst[self.idx+1]:
            self.lst[self.idx], self.lst[self.idx+1] = self.lst[self.idx+1], self.lst[self.idx]
            self.flag_move_bubble = True
        self.idx += 1

    def move_bubble(self):
        while self.idx < self.last:
            self.move_bubble_once()
        if not self.flag_move_bubble: # 거품이 움직이지 않았다면,
            self.flag_sorted = True   # self.lst는 이미 정렬되어 있다
            return                    # move_bubble 메서드에서 빠져나온다
        self.last -= 1
        self.idx = 0
        self.flag_move_bubble = False
        
    def run(self):
        while self.last > 0:
            self.move_bubble()
            if self.flag_sorted: # self.lst가 이미 정렬되어 있다면,
                return self.lst  # self.lst를 되돌린다
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
