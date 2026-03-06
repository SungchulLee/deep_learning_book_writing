# Inorder

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

```python
# recursion
class Solution:
    def inorderTraversal(self, root: TreeNode):
        if root is None:
            return []
        return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)
```

```python
# iteration
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

# Reference

[94. Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/)

[106. Construct Binary Tree from Inorder and Postorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)

[129. Sum Root to Leaf Numbers](https://leetcode.com/problems/sum-root-to-leaf-numbers/)

[230. Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)

[1305. All Elements in Two Binary Search Trees](https://leetcode.com/problems/all-elements-in-two-binary-search-trees/)

[Binary Tree Inorder Traversal - LeetCode 94 Python](https://www.youtube.com/watch?v=RJhh3Jcc9zw&t=605s)
