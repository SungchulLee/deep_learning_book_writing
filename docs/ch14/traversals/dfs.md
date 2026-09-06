# DFS
$$
\text{Search}
\left\{\begin{array}{lll}
\text{BFS (Queue)}\\
\\
\text{DFS (Stack)}&
\left\{\begin{array}{lll}
\text{DFS}\\
\\
\text{Backtracking}: \text{DFS}+\text{Pruning}
\end{array}\right.\\
\\
\text{BB (Priority Queue)}&
\left\{\begin{array}{lll}
\text{BB}\\
\\
\text{BB} + \text{Extended List}\\
\\
\text{BB} + \text{Admissible Heuristic}\\
\\
\text{A* Search}: \text{BB} + \text{Extended List} + \text{Consistant Heuristic}
\end{array}\right.\\
\end{array}
\right.
$$

# 참고 문헌

[5.1 Graph Traversals - BFS & DFS -Breadth First Search and Depth First Search](https://www.youtube.com/watch?v=pcKY4hjDrxk&list=PLDN4rrl48XKpZkf03iYFl-O29szjTrs_O&index=61)

[BFS DFS - Simplified](https://www.youtube.com/watch?v=kyUpc_5705s&list=PLDN4rrl48XKpZkf03iYFl-O29szjTrs_O&index=81)

John Levine
[Depth First Search](https://www.youtube.com/watch?v=h1RYvCfuoN4)

[39. Combination Sum](https://leetcode.com/problems/combination-sum/)

[133. Clone Graph](https://leetcode.com/problems/clone-graph/)

[695. Max Area of Island](https://leetcode.com/problems/max-area-of-island/)

[851. Loud and Rich](https://leetcode.com/problems/loud-and-rich/)

[1254. Number of Closed Islands](https://leetcode.com/problems/number-of-closed-islands/)

[1255. Maximum Score Words Formed by Letters](https://leetcode.com/problems/maximum-score-words-formed-by-letters/)

[1379. Find a Corresponding Node of a Binary Tree in a Clone of That Tree](https://leetcode.com/problems/find-a-corresponding-node-of-a-binary-tree-in-a-clone-of-that-tree/)

## 연습문제

**연습문제 1.**
꼭짓점 $\{A, B, C, D, E\}$과 변 $\{(A,B), (A,C), (B,D), (C,D), (D,E)\}$인 그래프에서 $A$부터 DFS이 굴러가는 것을 따라가라. 처음 들르는 차례대로 꼭짓점을 적어라.

??? success "연습문제 1 풀이"
    $A$에서 시작해 DFS은 되돌아오기 전에 되도록 깊이 살펴본다. 있을 수 있는 들름 차례 하나(이웃 차례에 따라 달라진다): $A, B, D, C, E$. $A$에서 $B$을 들르고, $B$에서 $D$을 들르고, $D$에서 $C$을 들르고($B$은 이미 들렀다), $D$에서 $E$을 들른 뒤 되돌아온다. 이웃 차례가 다르면 $A, C, D, B, E$도 올바른 차례이다.

---

**연습문제 2.**
이웃 목록으로 나타낸, 꼭짓점 $V$개와 변 $E$개인 그래프에서 DFS의 시간과 공간 복잡도는 얼마인가?

??? success "연습문제 2 풀이"
    **시간:** $O(V + E)$. 꼭짓점마다 꼭 한 번 들르고 변마다 한 번 살핀다(무방향 그래프는 두 번). **공간:** 다녀간 묶음과 되돌이 더미(또는 드러낸 더미)에 $O(V)$. 최악의 경우(길 그래프) 더미의 깊이는 $V$이다.

---

**연습문제 3.**
드러낸 더미를 써서 되풀이 DFS을 구현하고 되돌이 판과 견주어라.

??? success "연습문제 3 풀이"
    ```python
    def dfs_iterative(graph, source):
        visited = set()
        stack = [source]
        order = []
        while stack:
            v = stack.pop()
            if v not in visited:
                visited.add(v)
                order.append(v)
                for u in reversed(graph.get(v, [])):
                    if u not in visited:
                        stack.append(u)
        return order
    ```
    되풀이 판은 파이썬의 되돌이 한도를 피하고 드러낸 더미 공간을 $O(V)$ 쓴다. `reversed`을 쓰면 이웃을 왼쪽에서 오른쪽으로 다룰 때 되돌이 판과 들름 차례가 같아진다.

---

**연습문제 4.**
무방향 그래프에서 DFS이 가로 변을 만들지 않음(나무 변과 되돌이 변만 만듦)을 증명하여라.

??? success "연습문제 4 풀이"
    $u$을 다루는 도중 변 $(u, v)$을 살폈고 $v$은 이미 다녀갔다고 하자. 그래프가 무방향이므로 $v$은 DFS 나무에서 $u$의 조상이어야 한다(아니라면 $v$ 쪽에서 그 변을 다룰 때 $v$이 먼저 $u$을 찾았을 것이다). 조상으로 가는 변은 되돌이 변이다. 무방향 그래프에서는 나무가 아닌 변이 모두 자손과 조상을 잇기 때문에 앞선 변이나 가로 변은 생길 수 없다. $\square$
