# Find Path From Top To Bottom

# Infinity Loop

<img src='img/Screen Shot 2022-04-30 at 4.08.01 AM.png' width=50%>

# No Infinity Loop

<img src='img/Screen Shot 2022-04-30 at 4.09.13 AM.png' width=50%>

<img src='img/Screen Shot 2022-04-30 at 4.09.54 AM.png' width=50%>

```python
import matplotlib.pyplot as plt
import numpy as np
from functools import lru_cache as cache

class Maze:

    # deterministic maze
#     maze_size = 8
#     maze = (
#         [0, 0, 0, 0, 0, 0, 0, 1],
#         [0, 1, 1, 0, 1, 1, 0, 1],
#         [0, 0, 0, 1, 0, 0, 0, 1],
#         [0, 1, 0, 0, 1, 1, 0, 0],
#         [0, 1, 1, 1, 0, 0, 0, 1],
#         [0, 1, 0, 0, 0, 1, 0, 1],
#         [0, 0, 0, 1, 0, 0, 0, 1],
#         [0, 1, 1, 1, 0, 1, 0, 0]
#     )

    # random maze
    maze_size = 10
    maze = np.random.binomial(n=1, p=0.19, size=(maze_size,maze_size))
    
    undetermined_color = 0 # yellow
    wall_color = 1    # blue    
    no_path_color = 2 # red 
    path_color = 3    # green
    
    color_dict = {
        0: 'yellow',
        1: 'blue',
        2: 'red',
        3: 'green'
    }
    
    alpha_dict = {
        0: 0.2,
        1: 1,
        2: 0.2,
        3: 0.7
    }
    
    #@cache(maxsize=10)
    def find_maze_path(self, x, y):
        # base case
        if (x<0) or (y<0) or (x>=self.maze_size) or (y>=self.maze_size): # out of maze range 
            self.plot_current_state() 
            return False 
        elif (self.maze[x][y] != self.undetermined_color): # already know answer
            self.plot_current_state() 
            return False 
        elif (x==self.maze_size-1) and (y==self.maze_size-1): # exit location
            self.maze[x][y] = self.path_color
            self.plot_current_state() 
            return True
        
        # recursion
        self.maze[x][y] = self.path_color
        if self.find_maze_path(x+1, y) or self.find_maze_path(x, y+1)\
        or self.find_maze_path(x, y-1) or self.find_maze_path(x-1, y):
            self.plot_current_state() 
            return True
        else:
            self.maze[x][y] = self.no_path_color # dead end
            self.plot_current_state() 
            return False
    
    @classmethod
    def plot_current_state(cls):
        fig, ax = plt.subplots(1,1,figsize=(4,4))
        for y, row_with_fixed_y in enumerate(cls.maze):
            for x, color in enumerate(row_with_fixed_y):
                ax.plot([x,x], [7-y,7-y], 's',
                        markersize=20,
                        color=cls.color_dict[color],
                        alpha=cls.alpha_dict[color])
        ax.axis('off')
        plt.show()
            

def main():
    maze = Maze()
    result = maze.find_maze_path(0, 0)
    if result:
        print('We find a path from the top left corner to bottom right.')
    else:
        print('There is no path from the top left corner to bottom right.')     
    #maze.plot_current_state()    
    

if __name__ == "__main__":
    main()
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
<Figure size 288x288 with 1 Axes>
```

**Output:**
```
We find a path from the top left corner to bottom right.
```

# Reference

[[알고리즘] 제2-1강 Recursion의 응용: 미로찾기](https://www.youtube.com/watch?v=m6lXDsx7oCk&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=4)
