# 탐색
전위 — 뿌리 먼저

중위 — 뿌리 가운데

후위 — 뿌리 마지막

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.leftChild = None
        self.rightChild = None
    def find(self, data):
        if self.data == data:
            return True # 데이터를 찾음
        elif self.data > data:
            if self.leftChild:
                return self.leftChild.find(data) 
            else:
                return False # 데이터를 못 찾음
        else:
            if self.rightChild:
                return self.rightChild.find(data) 
            else:
                return False # 데이터를 못 찾음
    def insert(self, data):
        if self.data == data:
            return False # 노드를 넣지 않음
        elif self.data > data:
            if self.leftChild:
                return self.leftChild.insert(data) 
            else:
                self.leftChild = Node(data)
                return True # 노드를 넣음
        else:
            if self.rightChild:
                return self.rightChild.insert(data) 
            else:
                self.rightChild = Node(data)
                return True # 노드를 넣음
    def preorder(self):
        if self:
            print(str(self.data)) 
        if self.leftChild:
            self.leftChild.preorder()
        if self.rightChild:
            self.rightChild.preorder()
    def postorder(self):
        if self.leftChild:
            self.leftChild.preorder()
        if self.rightChild:
            self.rightChild.preorder()
        if self:
            print(str(self.data)) 
    def inorder(self):
        if self.leftChild:
            self.leftChild.preorder()
        if self:
            print(str(self.data)) 
        if self.rightChild:
            self.rightChild.preorder()
        
        
class Tree:
    def __init__(self):
        self.root = None
    def find(self, data):
        if self.root:
            return self.root.find(data) 
        else:
            return False # 데이터를 못 찾음
    def insert(self, data):
        if self.root:
            return self.root.insert(data) 
        else:
            self.root = Node(data)
            return True # 노드를 넣음
    def preorder(self):
        print("PreOrder") 
        self.root.preorder()
    def postorder(self):
        print("PostOrder") 
        self.root.postorder()
    def inorder(self):
        print("InOrder") 
        self.root.inorder()
        
        
def main():
    bst = Tree()
    for data in [10,5,2,5,3,20]:
        bst.insert(data)
        
    bst.preorder()
    bst.postorder()
    bst.inorder()
    

if __name__ == "__main__":
    main()
```

**출력:**
```
PreOrder
10
5
2
3
20
PostOrder
5
2
3
20
10
InOrder
5
2
3
10
20
```

# 참고 문헌

[Python: Binary Search Tree - BST](https://www.youtube.com/watch?v=YlgPi75hIBc)

## 연습문제

**연습문제 1.**
열쇠 $n$개를 무작위로 넣은 이진 탐색 트리에서 찾기의 기대 시간 복잡도를 유도하라.

??? success "연습문제 1 풀이"
    무작위 이진 탐색 트리의 기대 높이는 $O(\log n)$이다(더 정확히는 $\sim 2\ln n \approx 1.39 \log_2 n$). 비교할 때마다 한 층씩 내려가므로 기대 탐색 시간은 $O(\log n)$이다. 최악의 경우(정렬된 순서로 넣기)에는 높이가 $n$이므로 찾기가 $O(n)$이다.

---

**연습문제 2.**
반복적인 이진 탐색 트리 찾기를 구현하라.

??? success "연습문제 2 풀이"
    ```python
    def bst_search(root, key):
        while root:
            if key == root.val:
                return root
            elif key < root.val:
                root = root.left
            else:
                root = root.right
        return None
    ```

---

**연습문제 3.**
이진 탐색 트리의 찾기와 정렬된 배열의 이진 탐색을 견주어라.

??? success "연습문제 3 풀이"
    균형 잡힌 이진 탐색 트리와 정렬된 배열 모두 $O(\log n)$이다. 배열은 캐시 지역성이 좋고 포인터 부담이 없지만 삽입이 $O(n)$이다. 이진 탐색 트리는 (균형 잡혔다면) 삽입이 $O(\log n)$이고 동적인 연산을 지원하지만 캐시 성능이 나쁘다. 정적인 데이터에는 배열을, 동적인 데이터에는 이진 탐색 트리를 쓴다.

---

**연습문제 4.**
균형 이진 탐색 트리(AVL, 레드-블랙)가 균형 잡히지 않은 이진 탐색 트리보다 나은 점은 무엇인가?

??? success "연습문제 4 풀이"
    균형 이진 탐색 트리는 높이가 $O(\log n)$임을 보장하므로 최악의 경우에도 찾기·삽입·삭제가 $O(\log n)$이다. 균형 잡히지 않은 이진 탐색 트리는 적대적인 삽입 순서에서 $O(n)$으로 나빠질 수 있다. 대신 균형 이진 탐색 트리는 균형을 지키려고 삽입과 삭제 중에 회전 연산을 해야 한다.
