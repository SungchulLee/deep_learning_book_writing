# Insertion Sort - In-Place

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

**Output:**
```
[-5, -1, 3, 4, 7, 8, 10]
```

<div align="center"><img src="https://gmlwjd9405.github.io/images/algorithm-insertion-sort/sort-time-complexity.png" width="50%"></div>

[[알고리즘] 삽입 정렬(Insertion Sort)이란](https://gmlwjd9405.github.io/2018/05/06/algorithm-insertion-sort.html)

# Reference

[3. Insertion Sort, Merge Sort](https://www.youtube.com/watch?v=Kg4bqzAqRBM&list=PLUl4u3cNGP61Oq3tWYp6V_F-5jb5L2iHb&index=3)

[Insertion Sort - Data Structures & Algorithms Tutorial Python #16](https://www.youtube.com/watch?v=K0zTIF3rm9s&list=PLeo1K3hjS3uu_n_a__MI_KktGTLYopZ12&index=17)

[[알고리즘] 삽입 정렬(Insertion Sort)이란](https://gmlwjd9405.github.io/2018/05/06/algorithm-insertion-sort.html)

[정렬 알고리즘](https://namu.wiki/w/정렬%20알고리즘#s-2.3.1)
