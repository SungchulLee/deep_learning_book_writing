"""
Preorder
"""

# ======================================================================

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# recursion
class Solution:
    def preorderTraversal(self, root: TreeNode):
        if root is None:
            return []
        return [root.val] + self.preorderTraversal(root.left)  + self.preorderTraversal(root.right)


# iteration
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        if root is None:
            return []
        stack = [root]
        out = []
        while stack:
            cur = stack.pop()
            out.append(cur.val) 
            if cur.right:
                stack.append(cur.right) 
            if cur.left:
                stack.append(cur.left)         
        return out



# === Main ===
if __name__ == "__main__":
    pass
