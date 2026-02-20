# Heap Property


$$\begin{array}{lllllll}
\text{heappush}&:&\text{Insert}&&O(\log n)\\
\text{heappop}&:&\text{Delete (Root)}&&O(\log n)\\
\\
&:&\text{Create Heap from Left to Right}&&O(n\log n)\\
\text{heapify}&:&\text{Heapify or Create Heap from Right to Left}&&O(n)\\
\end{array}$$


$$\begin{array}{lll}
\text{Heat Sort - Step 1 : Creat Heap}&&O(n)\\
\text{Heat Sort - Step 2 : Delete Root}&&O(n\log n)\\
\text{Heat Sort - Step 1 + Step 2}&&O(n\log n)\\
\end{array}$$


```python
from heapq import heapify, heappop, heappush
```


```python
lst = [4,5,1,2,3]
lst
```

**Output:**
```
[4, 5, 1, 2, 3]
```


```python
# min heap is default
# 내부적으로는 인덱스 0에서 시작해 k번째 원소가 항상 자식 원소들(2k+1, 2k+2) 보다 작거나 같은 최소 힙의 형태로 정렬된다.
heapify(lst) 
lst
```

**Output:**
```
[1, 2, 4, 5, 3]
```


```python
a = heappop(lst) 
print(a, lst)
```

**Output:**
```
1 [2, 3, 4, 5]
```


```python
b = heappop(lst) 
print(b, lst)
```

**Output:**
```
2 [3, 5, 4]
```


```python
heappush(lst,1) 
print(lst)
```

**Output:**
```
[1, 3, 4, 5]
```


# Reference

[DOC](https://docs.python.org/3/library/heapq.html)

[[Python] 힙 자료구조 / 힙큐(heapq) / 파이썬에서 heapq 모듈 사용하기](https://littlefoxdiary.tistory.com/3)

[2.6.3 Heap - Heap Sort - Heapify - Priority Queues](https://www.youtube.com/watch?v=HqPJF2L5h9U&list=PLDN4rrl48XKpZkf03iYFl-O29szjTrs_O&index=32)
