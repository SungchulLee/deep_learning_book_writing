# Binary Search - Implementation


```python
class BinarySearchTreeNode:
    
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
        
    def add_child(self, data):
        
        # add data to left subtree
        if data < self.data: 
            if self.left:
                self.left.add_child(data)
            else:
                self.left = BinarySearchTreeNode(data)
                
        # add data to right subtree
        if data > self.data: 
            if self.right:
                self.right.add_child(data)
            else:
                self.right = BinarySearchTreeNode(data) 

    def in_order_traversal(self):
    # in_order_traversal [left root right]
    # pre_order_traversal [root left right]
    # post_order_traversal [left right root]
        
        elements = []
        
        # append left
        if self.left:
            elements += self.left.in_order_traversal()

        # append root
        elements.append(self.data)

        # append right
        if self.right:
            elements += self.right.in_order_traversal()

        return elements
    
    def pre_order_traversal(self):
    # in_order_traversal [left root right]
    # pre_order_traversal [root left right]
    # post_order_traversal [left right root]
        
        elements = []
        
        # append root
        elements.append(self.data)
        
        # append left
        if self.left:
            elements += self.left.in_order_traversal()

        # append right
        if self.right:
            elements += self.right.in_order_traversal()

        return elements
    
    def post_order_traversal(self):
    # in_order_traversal [left root right]
    # pre_order_traversal [root left right]
    # post_order_traversal [left right root]    
        
        elements = []
        
        # append left
        if self.left:
            elements += self.left.in_order_traversal()

        # append right
        if self.right:
            elements += self.right.in_order_traversal()
            
        # append root
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
        
        # node to delete (if exists) is in left tree 
        if val < self.data:
            if self.left:
                self.left = self.left.remove(val)
        
        # node to delete (if exists) is in right tree 
        elif val > self.data:
            if self.right:
                self.right = self.right.remove(val)
        
        # node to delete is root
        else:
            
            # if there is no node in right tree self.left becomes root
            if self.right is None:
                return self.left
            
            # otherwise, find min of right tree
            # use this min as root
            # remove this from right tree
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

**Output:**
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

**Output:**
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

**Output:**
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

**Output:**
```
In order traversal   :  ['China', 'Germany', 'India', 'UK', 'USA']
In order traversal   :  ['China', 'Germany', 'India', 'UK', 'USA']
```


# Reference

[Binary Tree Part 1 | BST | Binary Search Tree - Data Structures & Algorithms Tutorials In Python #10](https://www.youtube.com/watch?v=lFq5mYUWEBk) [github](https://github.com/codebasics/data-structures-algorithms-python/blob/master/data_structures/8_Binary_Tree_1/binary_tree_part_1.py)

[Binary Tree Part 2 | BST | Binary Search Tree - Data Structures & Algorithms Tutorials In Python #11](https://www.youtube.com/watch?v=JnrbMQyGLiU) [github](https://github.com/codebasics/data-structures-algorithms-python/blob/master/data_structures/9_Binary_Tree_2/binary_tree_part_2.py)
