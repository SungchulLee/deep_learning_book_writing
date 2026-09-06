# 후위 순회
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
    def postorderTraversal(self, root: TreeNode):
        if root is None:
            return []
        return self.postorderTraversal(root.left) + self.postorderTraversal(root.right) + [root.val]
```

```python
# 반복
class Solution:
    def postorderTraversal(self, root: TreeNode):
        if root is None:
            return []
        stack = [root]
        out = []
        while stack:
            cur = stack.pop()
            out.append(cur.val)
            if cur.left: 
                stack.append(cur.left) 
            if cur.right: 
                stack.append(cur.right) 
        return out[::-1]
```

# 참고 문헌

[106. Construct Binary Tree from Inorder and Postorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)

[145. Binary Tree Postorder Traversal](https://leetcode.com/problems/binary-tree-postorder-traversal/)

## 연습문제

**연습문제 1.**
트리를 지울 때 후위 순회를 쓰는 까닭을 설명하라.

??? success "연습문제 1 풀이"
    후위 순회는 부모보다 자식을 먼저 들르므로, 어떤 노드에 이르렀을 때 그 자식은 이미 처리되어(놓아주어) 있다. 그러면 이미 놓아준 메모리에 닿는 일이 없다. 부모를 먼저 지우면 자식에 대한 참조를 잃게 된다.

---

**연습문제 2.**
스택 두 개를 써서 반복적인 후위 순회를 구현하라.

??? success "연습문제 2 풀이"
    ```python
    def postorder_iterative(root):
        if not root:
            return []
        s1, s2 = [root], []
        while s1:
            node = s1.pop()
            s2.append(node)
            if node.left: s1.append(node.left)
            if node.right: s1.append(node.right)
        return [n.val for n in reversed(s2)]
    ```

---

**연습문제 3.**
전위 순회와 중위 순회가 주어졌을 때 이진 트리를 되살려라.

??? success "연습문제 3 풀이"
    전위 순회의 첫 원소가 뿌리이다. 그 뿌리를 중위 순회에서 찾으면 그 왼쪽이 왼쪽 부분 트리, 오른쪽이 오른쪽 부분 트리이다. 각 부분 트리에 해당하는 전위와 중위 조각으로 같은 일을 재귀적으로 되풀이한다.

---

**연습문제 4.**
후위 순회와 식의 계산 사이에는 어떤 관계가 있는가?

??? success "연습문제 4 풀이"
    식 트리에서 후위 순회는 연산자보다 피연산자를 먼저 계산한다. 왼쪽 자식(첫 피연산자), 오른쪽 자식(둘째 피연산자)을 들른 뒤 뿌리의 연산자를 들른다. 그러면 올바른 계산 순서가 나온다. 후위 나열을 '역폴란드 표기법'이라고도 한다.
