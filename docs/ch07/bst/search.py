"""
Search
"""

class Node:
    def __init__(self, data):
        self.data = data
        self.leftChild = None
        self.rightChild = None
    def find(self, data):
        if self.data == data:
            return True # data found
        elif self.data > data:
            if self.leftChild:
                return self.leftChild.find(data) 
            else:
                return False # data not found
        else:
            if self.rightChild:
                return self.rightChild.find(data) 
            else:
                return False # data not found
    def insert(self, data):
        if self.data == data:
            return False # Node not added 
        elif self.data > data:
            if self.leftChild:
                return self.leftChild.insert(data) 
            else:
                self.leftChild = Node(data)
                return True # Node added
        else:
            if self.rightChild:
                return self.rightChild.insert(data) 
            else:
                self.rightChild = Node(data)
                return True # Node added
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
            return False # data not found
    def insert(self, data):
        if self.root:
            return self.root.insert(data) 
        else:
            self.root = Node(data)
            return True # Node added
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
    


# === Main ===
if __name__ == "__main__":
    main()
