"""
Postorder
"""

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# recursion
class Solution:
    def postorderTraversal(self, root: TreeNode):
        if root is None:
            return []
        return self.postorderTraversal(root.left) + self.postorderTraversal(root.right) + [root.val]


# iteration
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
