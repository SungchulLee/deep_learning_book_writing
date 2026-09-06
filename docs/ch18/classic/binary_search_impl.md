# 이분 찾기 — 짜기
```python
class BinarySearchTreeNode:
    
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
        
    def add_child(self, data):
        
        # 왼쪽 아래나무에 자료 더하기
        if data < self.data: 
            if self.left:
                self.left.add_child(data)
            else:
                self.left = BinarySearchTreeNode(data)
                
        # 오른쪽 아래나무에 자료 더하기
        if data > self.data: 
            if self.right:
                self.right.add_child(data)
            else:
                self.right = BinarySearchTreeNode(data) 

    def in_order_traversal(self):
    # 중위 돌아보기 [왼쪽 뿌리 오른쪽]
    # 전위 돌아보기 [뿌리 왼쪽 오른쪽]
    # 후위 돌아보기 [왼쪽 오른쪽 뿌리]
        
        elements = []
        
        # 왼쪽 덧붙이기
        if self.left:
            elements += self.left.in_order_traversal()

        # 뿌리 덧붙이기
        elements.append(self.data)

        # 오른쪽 덧붙이기
        if self.right:
            elements += self.right.in_order_traversal()

        return elements
    
    def pre_order_traversal(self):
    # 중위 돌아보기 [왼쪽 뿌리 오른쪽]
    # 전위 돌아보기 [뿌리 왼쪽 오른쪽]
    # 후위 돌아보기 [왼쪽 오른쪽 뿌리]
        
        elements = []
        
        # 뿌리 덧붙이기
        elements.append(self.data)
        
        # 왼쪽 덧붙이기
        if self.left:
            elements += self.left.in_order_traversal()

        # 오른쪽 덧붙이기
        if self.right:
            elements += self.right.in_order_traversal()

        return elements
    
    def post_order_traversal(self):
    # 중위 돌아보기 [왼쪽 뿌리 오른쪽]
    # 전위 돌아보기 [뿌리 왼쪽 오른쪽]
    # 후위 돌아보기 [왼쪽 오른쪽 뿌리]
        
        elements = []
        
        # 왼쪽 덧붙이기
        if self.left:
            elements += self.left.in_order_traversal()

        # 오른쪽 덧붙이기
        if self.right:
            elements += self.right.in_order_traversal()
            
        # 뿌리 덧붙이기
        elements.append(self.data)

        return elements
    
    def is_in(self, val):
        
        if val == self.data:
            return True

        if val < self.data:
            if self.left:
                return self.left.is_in(val)
            else:
                return False

        if val > self.data:
            if self.right:
                return self.right.is_in(val)
            else:
                return False
            
    def find_min(self):
        if self.left is None:
            return self.data
        return self.left.find_min()
    
    def find_max(self):
        if self.right is None:
            return self.data
        return self.right.find_max()
    
    def calculate_sum(self):
        return sum(self.in_order_traversal())
    
    def remove(self, val):
        
        # 지울 마디는 (있다면) 왼쪽 나무에 있다
        if val < self.data:
            if self.left:
                self.left = self.left.remove(val)
        
        # 지울 마디는 (있다면) 오른쪽 나무에 있다
        elif val > self.data:
            if self.right:
                self.right = self.right.remove(val)
        
        # 지울 마디가 뿌리이다
        else:
            
            # 오른쪽 나무에 마디가 없으면 self.left가 뿌리가 된다
            if self.right is None:
                return self.left
            
            # 그렇지 않으면 오른쪽 나무의 최솟값 찾기
            # 이 최솟값을 뿌리로 쓰기
            # 오른쪽 나무에서 이것을 없애기
            min_val = self.right.find_min()
            self.data = min_val
            self.right = self.right.remove(min_val)

        return self
```

```python
def build_tree(elements):
    """
    input : list of numbers
    output : root 
    """
    
    for idx, element in enumerate(elements):
        if idx == 0:
            root = BinarySearchTreeNode(element)
        else:
            root.add_child(element)
            
    return root
```

```python
if __name__ == '__main__':
    data = ["India", "Germany", "USA", "China", "India", "UK", "USA"]
    #data = [17, 4, 1, 20, 9, 23, 18, 34, 18, 4]
    tree = build_tree(data)
    print("Membership Test : ", tree.is_in('Korea'))
```

**출력:**
```
Membership Test :  False
```

```python
if __name__ == '__main__':
    data = ["India", "Germany", "USA", "China", "India", "UK", "USA"]
    #data = [17, 4, 1, 20, 9, 23, 18, 34, 18, 4]
    tree = build_tree(data)
    print("In order traversal     : ", tree.in_order_traversal())
    print("Pre order traversal    : ", tree.pre_order_traversal())
    print("Post order traversal   : ", tree.post_order_traversal())
```

**출력:**
```
In order traversal     :  ['China', 'Germany', 'India', 'UK', 'USA']
Pre order traversal    :  ['India', 'China', 'Germany', 'UK', 'USA']
Post order traversal   :  ['China', 'Germany', 'UK', 'USA', 'India']
```

```python
if __name__ == '__main__':
    #data = ["India", "Germany", "USA", "China", "India", "UK", "USA"]
    data = [17, 4, 1, 20, 9, 23, 18, 34, 18, 4]
    tree = build_tree(data)
    print("min : ", tree.find_min())
    print("max : ", tree.find_max())
    print("sum : ", tree.calculate_sum())
```

**출력:**
```
min :  1
max :  34
sum :  126
```

```python
if __name__ == '__main__':
    data = ["India", "Germany", "USA", "China", "India", "UK", "USA"]
    #data = [17, 4, 1, 20, 9, 23, 18, 34, 18, 4]
    tree = build_tree(data)
    print("In order traversal   : ", tree.in_order_traversal())
    tree = tree.remove('Korea')
    print("In order traversal   : ", tree.in_order_traversal())
```

**출력:**
```
In order traversal   :  ['China', 'Germany', 'India', 'UK', 'USA']
In order traversal   :  ['China', 'Germany', 'India', 'UK', 'USA']
```

# 참고 문헌

[Binary Tree Part 1 | BST | Binary Search Tree - Data Structures & Algorithms Tutorials In Python #10](https://www.youtube.com/watch?v=lFq5mYUWEBk) [github](https://github.com/codebasics/data-structures-algorithms-python/blob/master/data_structures/8_Binary_Tree_1/binary_tree_part_1.py)

[Binary Tree Part 2 | BST | Binary Search Tree - Data Structures & Algorithms Tutorials In Python #11](https://www.youtube.com/watch?v=JnrbMQyGLiU) [github](https://github.com/codebasics/data-structures-algorithms-python/blob/master/data_structures/9_Binary_Tree_2/binary_tree_part_2.py)

## 연습문제

**연습문제 1.**
1차원 배열의 봉우리 찾기 문제를 정의하고 봉우리가 늘 있는 까닭을 밝혀라.

??? success "연습문제 1 풀이"
    A peak in a 1D array $A[0..n-1]$ is an element $A[i]$ such that $A[i] \geq A[i-1]$ and $A[i] \geq A[i+1]$ (with boundary conditions $A[-1] = A[n] = -\infty$). A peak always exists because: consider the global maximum element. It is at least as large as both its neighbors, so it is a peak. More formally, any finite array has a maximum, and the maximum satisfies the peak condition.

---

**연습문제 2.**
Explain how binary search finds a peak in $O(\log n)$ time. Why is the midpoint comparison sufficient to decide which half contains a peak?

??? success "연습문제 2 풀이"
    Compare $A[\text{mid}]$ with its neighbors. If $A[\text{mid}] < A[\text{mid}+1]$, the right half (including $\text{mid}+1$) must contain a peak: either $A[\text{mid}+1]$ is a peak, or values keep increasing until they must eventually decrease (at the boundary), creating a peak. Similarly, if $A[\text{mid}] < A[\text{mid}-1]$, the left half contains a peak. If neither neighbor is larger, $A[\text{mid}]$ itself is a peak. Each comparison halves the search space, giving $O(\log n)$ time.

---

**연습문제 3.**
봉우리 찾기 알고리즘을 2차원으로 넓혀라. 나누어 이기기 방식의 시간 복잡도는 무엇인가?

??? success "연습문제 3 풀이"
    For an $n \times m$ matrix, find the maximum in the middle column ($O(n)$ time), then compare with horizontal neighbors. If a neighbor is larger, recurse on that half of the columns. Finding the column max ensures we move toward a 2D peak. Recurrence: $T(n, m) = T(n, m/2) + O(n)$, giving $O(n \log m)$. Alternatively, alternating between row and column searches gives $O(n + m)$ or $O(\max(n,m))$ depending on the strategy.

---

**연습문제 4.**
봉우리 찾기 문제에 이분 찾기를 짜고 모서리 경우(테두리의 봉우리, 평평한 곳)를 올바로 다루는지 확인하여라.

??? success "연습문제 4 풀이"
    ```python
    def find_peak(arr):
        lo, hi = 0, len(arr) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if arr[mid] < arr[mid + 1]:
                lo = mid + 1
            else:
                hi = mid
        return lo

    # 시험
    assert find_peak([1, 3, 2]) == 1
    assert find_peak([5, 4, 3, 2, 1]) == 0  # 왼쪽 테두리의 봉우리
    assert find_peak([1, 2, 3, 4, 5]) == 4  # 오른쪽 테두리의 봉우리
    assert find_peak([1, 2, 2, 2, 1]) in [1, 2, 3]  # 평평한 곳
    ```
    The algorithm correctly handles boundary peaks by the boundary condition $A[-1] = A[n] = -\infty$.
