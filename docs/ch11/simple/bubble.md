# Bubble Sort


<img src="img/Screen Shot 2022-04-30 at 11.43.25 AM.png" width="70%">


<img src="img/Screen Shot 2022-04-30 at 12.23.04 PM.png" width="50%">


```python
class Bubble_Sort:
    
    def __init__(self, lst):
        self.lst = lst.copy()
        self.last = len(self.lst) - 1 # last of left, boundary between left and right
        self.idx = 0 # index of swap between self.lst[self.idx] and self.lst[self.idx+1]
        
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

**Output:**
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


# Reference

[[알고리즘] 제3강 기본적인 정렬 알고리즘](https://www.youtube.com/watch?v=0dG7xTt5IfQ&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=9)

[Bubble Sort - Data Structures & Algorithms Tutorial Python #14](https://www.youtube.com/watch?v=ppmIOUIz4uI&list=PLeo1K3hjS3uu_n_a__MI_KktGTLYopZ12&index=15)

[Bubble-sort with Hungarian ("Csángó") folk dance](https://www.youtube.com/watch?v=lyZQPjUT5B4)

[[알고리즘] 버블 정렬(Bubble Sort)이란](https://gmlwjd9405.github.io/2018/05/06/algorithm-bubble-sort.html)

[정렬 알고리즘](https://namu.wiki/w/정렬%20알고리즘#s-2.3.1)
