# 중위 순회
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

```python
# 재귀
class Solution:
    def inorderTraversal(self, root: TreeNode):
        if root is None:
            return []
        return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)
```

```python
# 반복
class Solution:
    def inorderTraversal(self, root: TreeNode):
        if root is None:
            return []
        cur = root
        stack = []
        out = []
        while cur or stack:
            if cur: 
                stack.append(cur) 
                cur = cur.left 
            else: 
                cur = stack.pop() 
                out.append(cur.val) 
                cur = cur.right 
        return out
```

# 참고 문헌

[94. Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/)

[106. Construct Binary Tree from Inorder and Postorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)

[129. Sum Root to Leaf Numbers](https://leetcode.com/problems/sum-root-to-leaf-numbers/)

[230. Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)

[1305. All Elements in Two Binary Search Trees](https://leetcode.com/problems/all-elements-in-two-binary-search-trees/)

[Binary Tree Inorder Traversal - LeetCode 94 Python](https://www.youtube.com/watch?v=RJhh3Jcc9zw&t=605s)

## 연습문제

**연습문제 1.**
중위 순회의 시간 복잡도와 공간 복잡도는 얼마인가?

??? success "연습문제 1 풀이"
    시간은 $O(n)$이다. 노드마다 꼭 한 번씩 들른다. 공간은 재귀 스택에 $O(h)$이며 $h$은 트리의 높이이다. 균형 잡힌 트리에서는 $h = O(\log n)$, 치우친 트리에서는 $h = O(n)$이다.

---

**연습문제 2.**
명시적인 스택을 써서 반복적인 중위 순회를 구현하라.

??? success "연습문제 2 풀이"
    ```python
    def inorder_iterative(root):
        stack, result, curr = [], [], root
        while curr or stack:
            while curr:
                stack.append(curr)
                curr = curr.left
            curr = stack.pop()
            result.append(curr.val)
            curr = curr.right
        return result
    ```

---

**연습문제 3.**
이진 탐색 트리의 중위 순회가 노드를 정렬된 순서로 들름을 증명하라.

??? success "연습문제 3 풀이"
    이진 탐색 트리 성질에 따라 모든 노드에서 왼쪽 부분 트리의 값 < 노드의 값 < 오른쪽 부분 트리의 값이다. 중위 순회는 왼쪽 부분 트리, 노드, 오른쪽 부분 트리 순으로 들른다. 귀납법으로 보면 왼쪽 부분 트리의 값들이 (모두 노드보다 작은) 정렬된 순서로 나오고, 이어 노드, 그다음 오른쪽 부분 트리의 값들이 (모두 노드보다 큰) 정렬된 순서로 나온다. 이어 붙인 결과는 정렬되어 있다. $\square$

---

**연습문제 4.**
중위 순회 결과에서 이진 탐색 트리를 어떻게 되살릴 수 있는가? 그 복원은 유일한가?

??? success "연습문제 4 풀이"
    이진 탐색 트리의 중위 순회는 정렬된 나열이다. 그런데 중위 순회만으로는 이진 탐색 트리가 유일하게 정해지지 않는다(중위 순회가 같은 서로 다른 트리가 많다). 어떤 이진 트리든 유일하게 되살리려면 중위와 전위(또는 중위와 후위)가 함께 있어야 한다.
