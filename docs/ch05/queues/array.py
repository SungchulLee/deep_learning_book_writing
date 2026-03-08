"""
Array-Based Queue
"""

import numpy as np; np.random.seed(0)


# === Functions ===
def main():
    q = []
    
    for i in range(10):
        if np.random.uniform() < 0.5:
            enqueue_item = np.random.randint(low=1, high=7)
            q.append(enqueue_item)
            print(f'{i = }, {enqueue_item = :4}, {q = }')
        else:
            try:
                dequeue_item = q.pop(0)
            except IndexError:
                dequeue_item = str(None)
            print(f'{i = }, {dequeue_item = :4}, {q = }')

            

# === Main ===
if __name__ == "__main__":
    main()
