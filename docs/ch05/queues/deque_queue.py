"""
Deque as Queue
"""

import numpy as np; np.random.seed(0)
from collections import deque


def main():
    q = deque()

    for i in range(10):
        if np.random.uniform() < 0.5:
            enqueue_item = np.random.randint(low=1, high=7)
            q.append(enqueue_item)
            print(f'{i = }, {enqueue_item = :4}, {q = }')
        else:
            try:
                dequeue_item = q.popleft()
            except IndexError:
                dequeue_item = str(None)
            print(f'{i = }, {dequeue_item = :4}, {q = }')
            
            
if __name__ == "__main__":
    main()
