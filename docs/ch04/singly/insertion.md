# Insertion

<div align="center"><img src="https://cdn-images-1.medium.com/max/1600/1*rEC8Te27eo5TSYCHMA7Ttw.png" width="100%" height="30%"></div>

```python
class Node:
    def __init__(self, data=None, next_data=None):
        self.data = data
        self.next = next_data
```

```python
class LinkedList:
    def __init__(self, data_list=None):
        self.head = None
        if data_list is not None:
            for data in data_list:
                self.insert_at_end(data)
        
    def get_length(self):
        count = 0
        itr = self.head
        while itr:
            count += 1
            itr = itr.next
        return count
    
    def insert_at_start(self, data):
        node = Node(data, self.head)
        self.head = node

    def insert_at_end(self, data):
        if self.head is None:
            self.insert_at_start(data)
            return
        itr = self.head
        while itr.next:
            itr = itr.next
        itr.next = Node(data, None)
        
    def insert(self, index, data):
        if (index < 0) or (index > self.get_length()):
            raise Exception("Invalid Index")
        if index == 0:
            self.insert_at_start(data)
            return
        itr = self.head
        for _ in range(1,index):
            itr = itr.next
        node = Node(data, itr.next)
        itr.next = node
        
    def remove(self, index):
        if (index < 0) or (index >= self.get_length()):
            raise Exception("Invalid Index")
        if index==0:
            self.head = self.head.next
            return
        itr = self.head
        for _ in range(1,index):
            itr = itr.next
        itr.next = itr.next.next
        
    def print_linked_list(self):
        if self.head is None:
            print("Linked list is empty")
            return
        msg = ''
        itr = self.head
        while itr:
            msg += str(itr.data) + ' --> ' if itr.next else str(itr.data)
            itr = itr.next
        print(msg)
```

```python
l = LinkedList()
l.insert_at_start(2)
l.insert_at_start(3)
l.insert_at_start(5)
l.print_linked_list()
```

```python
l = LinkedList()
l.insert_at_end(2)
l.insert_at_end(3)
l.insert_at_end(5)
l.print_linked_list()
```

```python
l = LinkedList()
l.insert_at_end(2)
l.insert_at_end(3)
l.insert_at_end(5)
l.remove(1)
l.print_linked_list()
```

```python
l = LinkedList(["banana","mango","grapes","orange"])
l.print_linked_list()
```

```python
l = LinkedList(["banana","mango","grapes","orange"])
l.insert(1,"blueberry")
l.print_linked_list()
```

# Reference

[Linked List - Data Structures & Algorithms Tutorials in Python #4](https://www.youtube.com/watch?v=qp8u-frRAnU&list=PLeo1K3hjS3uu_n_a__MI_KktGTLYopZ12&index=4)

[109. Convert Sorted List to Binary Search Tree](https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/)

[141. Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/)

[160. Intersection of Two Linked Lists](https://leetcode.com/problems/intersection-of-two-linked-lists/)

[234. Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list/)

[237. Delete Node in a Linked List](https://leetcode.com/problems/delete-node-in-a-linked-list/)
