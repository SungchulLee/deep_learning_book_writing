"""
Linked List Stack
"""

import numpy as np; np.random.seed(0)
from collections import deque


class Stack:
    def __init__(self):
        self.stack = deque()   
    def push(self,val):
        self.stack.append(val)       
    def pop(self):
        if len(self.stack) > 0:
            return self.stack.pop()
        else:
            return None
    def peek(self):
        if len(self.stack) > 0:
            return self.stack[-1]
        else:
            return None
    def is_empty(self):
        return len(self.stack)==0
    def size(self):
        return len(self.stack)
    def __str__(self):
        return str(self.stack)


def main():
    s = Stack()

    for i in range(10):
        if np.random.uniform() < 0.5:
            push_item = np.random.randint(low=1, high=7)
            s.push(push_item)
            print(f'{i = }, {push_item = :>8}, {str(s) = }')
        else:
            pop_item = s.pop()    
            print(f'{i = }, {str(pop_item) = :>4}, {str(s) = }')
            
            
if __name__ == "__main__":
    main()
