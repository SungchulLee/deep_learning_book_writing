"""
Array-Based Stack
"""

import numpy as np; np.random.seed(0)


# === Functions ===

def main():
    s = []
    
    for i in range(10):
        if np.random.uniform() < 0.5:
            push_item = np.random.randint(low=1, high=7)
            s.append(push_item)
            print(f'{i = }, {push_item = :3}, {s = }')
        else:
            try:
                pop_item = s.pop()
            except IndexError:
                pop_item = str(None)
            print(f'{i = }, {pop_item = :4}, {s = }')
            
            

# === Main ===
if __name__ == "__main__":
    main()
