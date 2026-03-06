"""
Inorder
"""

# ======================================================================

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# recursion
class Solution:
    def inorderTraversal(self, root: TreeNode):
        if root is None:
            return []
        return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)


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



# === Main ===
if __name__ == "__main__":
    pass
