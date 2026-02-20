# Selection Sort


<img src="img/Screen Shot 2022-04-30 at 11.43.25 AM.png" width="70%">


<img src="img/Screen Shot 2022-04-30 at 11.46.02 AM.png" width="50%">


```python
class Selection_Sort:
    
    def __init__(self, lst):
        self.lst = lst.copy()
        self.last = len(self.lst) - 1 # last of left, boundary between left and right
        self.max = None # index of max in self.lst[:self.last+1]
        
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


<div align="center"><img src="https://gmlwjd9405.github.io/images/algorithm-selection-sort/sort-time-complexity.png" width="50%"></div>

[[알고리즘] 선택 정렬(Selection Sort)이란](https://gmlwjd9405.github.io/2018/05/06/algorithm-selection-sort.html)


# Reference

[[알고리즘] 제3강 기본적인 정렬 알고리즘](https://www.youtube.com/watch?v=0dG7xTt5IfQ&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=9)

[Selection Sort in python - Data Structures & Algorithms Tutorial Python #19](https://www.youtube.com/watch?v=hhkLdjIimlw&list=PLeo1K3hjS3uu_n_a__MI_KktGTLYopZ12&index=19)

[[알고리즘] 선택 정렬(Selection Sort)이란](https://gmlwjd9405.github.io/2018/05/06/algorithm-selection-sort.html)

[정렬 알고리즘](https://namu.wiki/w/정렬%20알고리즘#s-2.3.1)
