"""
Linked List Queue
"""

import numpy as np; np.random.seed(0)
from collections import deque


class Queue:   
    def __init__(self):
        self.queue = deque()
    def enqueue(self, val):
        self.queue.append(val)    
    def dequeue(self):
        if len(self.queue) > 0:
            return self.queue.popleft()
        else:
            return None
    def peek(self):
        if len(self.queue) > 0:
            return self.queue[0]
        else:
            return None
    def is_empty(self):
        return len(self.queue)==0
    def size(self):
        return len(self.queue)
    def __str__(self):
        return str(self.queue)
    
    
def main():
    q = Queue()

    for i in range(10):
        if np.random.uniform() < 0.5:
            enqueue_item = np.random.randint(low=1, high=7)
            q.enqueue(enqueue_item)
            print(f'{i = }, {enqueue_item = :>9}, {str(q) = }')
        else:
            dequeue_item = q.dequeue()
            print(f'{i = }, {str(dequeue_item) = :>4}, {str(q) = }')
            
            
if __name__ == "__main__":
    main()
