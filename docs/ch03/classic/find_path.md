# 위에서 아래로 가는 경로 찾기
# 무한 루프

<img src='img/Screen Shot 2022-04-30 at 4.08.01 AM.png' width=50%>

# 무한 루프 없음

<img src='img/Screen Shot 2022-04-30 at 4.09.13 AM.png' width=50%>

<img src='img/Screen Shot 2022-04-30 at 4.09.54 AM.png' width=50%>

```python
import matplotlib.pyplot as plt
import numpy as np
from functools import lru_cache as cache

class Maze:

    # 결정적인 미로
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

    # 무작위 미로
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
        # 기저 사례
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
        
        # 재귀
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

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
<Figure size 288x288 with 1 Axes>
```

**출력:**
```
We find a path from the top left corner to bottom right.
```

# 참고 자료

[[알고리즘] 제2-1강 Recursion의 응용: 미로찾기](https://www.youtube.com/watch?v=m6lXDsx7oCk&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=4)

## 연습문제

**연습문제 1.**
위에서 아래로 가는 서로 다른 경로의 총 개수를 세도록 경로 찾기 알고리즘을 수정하라.

??? success "연습문제 1 풀이"
    ```python
    def count_paths(grid, row=0, col=0):
        if row == len(grid) - 1 and col == len(grid[0]) - 1:
            return 1
        count = 0
        if row + 1 < len(grid):
            count += count_paths(grid, row + 1, col)
        if col + 1 < len(grid[0]):
            count += count_paths(grid, row, col + 1)
        return count
    ```

---

**연습문제 2.**
중복되는 부분문제를 다시 계산하지 않도록 메모화를 추가하라.

??? success "연습문제 2 풀이"
    ```python
    from functools import lru_cache

    def count_paths_memo(grid):
        @lru_cache(maxsize=None)
        def helper(row, col):
            if row == len(grid)-1 and col == len(grid[0])-1:
                return 1
            count = 0
            if row + 1 < len(grid):
                count += helper(row + 1, col)
            if col + 1 < len(grid[0]):
                count += helper(row, col + 1)
            return count
        return helper(0, 0)
    ```
    메모화를 쓰면 시간이 $O\binom{m+n}{m}$에서 $O(mn)$으로 줄어든다.

---

**연습문제 3.**
대각선 이동(오른쪽 아래)을 허용하도록 문제를 확장하라. $m \times n$ 격자에서 경로의 개수가 어떻게 달라지는가?

??? success "연습문제 3 풀이"
    방향이 세 가지(아래, 오른쪽, 대각선)이면 점화식이 $f(i,j) = f(i+1,j) + f(i,j+1) + f(i+1,j+1)$이 된다. $3 \times 3$ 격자에서는 경로가 13개이다(아래/오른쪽만 쓸 때의 6개와 대비된다). 이 개수가 델라노이 수 $D(m-1, n-1)$이다.

---

**연습문제 4.**
각 칸에 값이 있는 격자에서 합이 최대인 경로를 찾아라.

??? success "연습문제 4 풀이"
    ```python
    def max_path_sum(grid, row=0, col=0, memo={}):
        if (row, col) in memo:
            return memo[(row, col)]
        if row == len(grid)-1 and col == len(grid[0])-1:
            return grid[row][col]
        best = float('-inf')
        if row + 1 < len(grid):
            best = max(best, max_path_sum(grid, row+1, col, memo))
        if col + 1 < len(grid[0]):
            best = max(best, max_path_sum(grid, row, col+1, memo))
        memo[(row, col)] = grid[row][col] + best
        return memo[(row, col)]
    ```
