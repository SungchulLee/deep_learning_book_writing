# 고르기 정렬
<img src="img/Screen Shot 2022-04-30 at 11.43.25 AM.png" width="70%">

<img src="img/Screen Shot 2022-04-30 at 11.46.02 AM.png" width="50%">

```python
class Selection_Sort:
    
    def __init__(self, lst):
        self.lst = lst.copy()
        self.last = len(self.lst) - 1 # 왼쪽의 끝, 왼쪽과 오른쪽의 경계
        self.max = None # self.lst[:self.last+1]에서 가장 큰 값의 첨자
        
    def find_max(self):
        max_value = max(self.lst[:self.last+1])
        self.max = self.lst.index(max_value)

    def move_max_to_right(self):
        self.lst[self.last], self.lst[self.max] = self.lst[self.max], self.lst[self.last]
        self.last -= 1
        
    def run(self):
        while self.last > 0:
            self.find_max()
            self.move_max_to_right()
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
        obj = Selection_Sort(lst)
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

<div align="center"><img src="https://gmlwjd9405.github.io/images/algorithm-selection-sort/sort-time-complexity.png" width="50%"></div>

[[알고리즘] 선택 정렬(Selection Sort)이란](https://gmlwjd9405.github.io/2018/05/06/algorithm-selection-sort.html)

# 참고 문헌

[[알고리즘] 제3강 기본적인 정렬 알고리즘](https://www.youtube.com/watch?v=0dG7xTt5IfQ&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=9)

[Selection Sort in python - Data Structures & Algorithms Tutorial Python #19](https://www.youtube.com/watch?v=hhkLdjIimlw&list=PLeo1K3hjS3uu_n_a__MI_KktGTLYopZ12&index=19)

[[알고리즘] 선택 정렬(Selection Sort)이란](https://gmlwjd9405.github.io/2018/05/06/algorithm-selection-sort.html)

[정렬 알고리즘](https://namu.wiki/w/정렬%20알고리즘#s-2.3.1)

## 연습문제

**연습문제 1.**
고르기 정렬이 하는 견줌의 정확한 횟수를 끌어내라.

??? success "연습문제 1 풀이"
    바깥 되돌이는 $n-1$번 돈다. 안쪽 되돌이는 원소 $n-i$개에서 가장 작은 것을 찾는다. 모두 합하면 입력 차례와 상관없이 $\sum_{i=1}^{n-1}(n-i) = \frac{n(n-1)}{2}$번 견준다.

---

**연습문제 2.**
고르기 정렬을 구현하라.

??? success "연습문제 2 풀이"
    ```python
    def selection_sort(arr):
        for i in range(len(arr)):
            min_idx = i
            for j in range(i+1, len(arr)):
                if arr[j] < arr[min_idx]:
                    min_idx = j
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
    ```

---

**연습문제 3.**
고르기 정렬은 왜 꼭 $n-1$번 맞바꾸는가?

??? success "연습문제 3 풀이"
    바깥 되돌이를 한 번 돌 때마다 꼭 한 번 맞바꾸어 원소 하나를 마지막 자리에 놓는다(가장 작은 것을 자리 $i$로 맞바꾼다). $n-1$번 돌면 모든 원소가 정렬되므로 맞바꿈은 꼭 $n-1$번이다.

---

**연습문제 4.**
고르기 정렬은 안정적인가? 그렇지 않다면 반례를 들어라.

??? success "연습문제 4 풀이"
    표준 고르기 정렬은 안정적이지 **않다**. 반례: [2a, 2b, 1]. 첫 번째 훑기에서 2a와 1을 맞바꾸어 [1, 2b, 2a]가 된다. 같은 원소 (2a, 2b)의 상대 차례가 뒤집힌다.
