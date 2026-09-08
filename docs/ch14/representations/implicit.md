# 속뜻 그래프

모든 그래프가 이웃 목록이나 행렬에 말끔히 들어맞지는 않는다. 중요한 그래프 가운데 많은 것이 드러내 저장하기에는 너무 크거나 아예 끝이 없다. 그래도 그 짜임이 단순한 규칙을 따르므로 필요할 때 이웃을 셈할 수 있다. 칸마다 동서남북 이웃에 이어진 장기판, 상태마다 한 번 돌려 닿는 상태에 이어진 루빅 큐브, 체스의 놀이 나무가 모두 자료가 아니라 규칙으로 정해지는 그래프이다. 이런 것을 속뜻 그래프라 하며, 이를 이해하는 일은 찾기, 계획 세우기, 조합 최적화에 꼭 필요하다.

---

## 1. 정의

**속뜻 그래프** $G = (V, E)$은 꼭짓점과 변을 기억 공간에 드러내 저장하지 않는 그래프이다. 그 대신 **뒤따름 함수**(또는 이웃 함수) $\text{neighbors}(v)$이 필요할 때 $v$에 이웃한 꼭짓점의 묶음을 셈한다.

엄밀하게는 그래프가 다음으로 정해진다:

1. 첫 꼭짓점(또는 꼭짓점 묶음) $s \in V$.
2. 주어진 꼭짓점의 이웃을 되돌리는 함수 $\text{neighbors}: V \to 2^V$.
3. 있어도 되고 없어도 되는 목표 술어 $\text{goal}: V \to \{0, 1\}$.

변의 묶음은 속뜻으로 정해진다:

$$
E = \{(u, v) : v \in \text{neighbors}(u)\}
$$

---

## 2. 속뜻 그래프가 왜 중요한가

상태 공간이 너무 커서 낱낱이 셀 수 없을 때 속뜻 그래프가 나온다:

| 분야 | 꼭짓점 | 이웃 함수 | 상태 공간의 크기 |
|---|---|---|---|
| 격자/미로 | 칸 $(r, c)$ | 벽이 아닌 이웃 칸 | $O(rows \times cols)$ |
| 미끄럼 퍼즐 | 판의 꼴 | 한 번 밀어 닿는 꼴 | $n!$ |
| 루빅 큐브 | 큐브 상태 | 90도 한 번 돌려 나오는 상태 | $\approx 4.3 \times 10^{19}$ |
| 체스 | 판의 자리 | 규칙에 맞는 수 | $\approx 10^{47}$ |
| 낱말 사다리 | 사전의 낱말 | 글자 하나만 다른 낱말 | 사전의 크기 |

루빅 큐브에서 $4.3 \times 10^{19}$개 상태를 모두 드러내 저장하기란 불가능하다. BFS이나 DFS은 시작에서 닿는 상태만 찾아내며 이웃을 게으르게 만든다.

---

## 3. 격자 그래프

알고리즘 문제에서 가장 흔한 속뜻 그래프는 **격자 그래프**이다. $m \times n$ 격자는 $0 \leq r < m$과 $0 \leq c < n$인 정수 자리 $(r, c)$에 꼭짓점을 둔다. 칸마다 동서남북 이웃 4개(위, 아래, 왼쪽, 오른쪽)에 이어지되 가장자리 조건과 걸림돌을 따른다.

$$
\text{neighbors}(r, c) = \{(r', c') : |r - r'| + |c - c'| = 1, \; 0 \leq r' < m, \; 0 \leq c' < n, \; \text{not blocked}\}
$$

```python
"""
속뜻 그래프 보기: 격자 그래프와 낱말 사다리.

드러낸 이웃 짜임 대신 이웃 함수로 그래프를 정하여
크거나 끝없는 상태 공간에서 BFS/DFS을 할 수 있게
하는 법을 보인다.
"""

from collections import deque

# === 격자 그래프 ===

def grid_neighbors(r, c, rows, cols, blocked=None):
    """
    격자 그래프에서 칸 (r, c)의 이웃을 셈한다.

    테두리 안에 있고 막히지 않은 이웃 칸을 되돌린다.
    """
    if blocked is None:
        blocked = set()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    result = []
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in blocked:
            result.append((nr, nc))
    return result

def bfs_grid(rows, cols, start, goal, blocked=None):
    """
    속뜻 격자 그래프에서의 BFS.

    start에서 goal까지 최단 경로의 길이를 되돌리고,
    길이 없으면 -1을 되돌린다.
    """
    if blocked is None:
        blocked = set()
    if start == goal:
        return 0
    visited = {start}
    queue = deque([(start, 0)])

    while queue:
        (r, c), dist = queue.popleft()
        for nr, nc in grid_neighbors(r, c, rows, cols, blocked):
            if (nr, nc) == goal:
                return dist + 1
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), dist + 1))
    return -1

# === 낱말 사다리 ===

def word_neighbors(word, dictionary):
    """
    낱말 사다리 그래프에서 이웃을 셈한다.

    이웃은 입력과 글자 꼭 하나가 다른
    사전의 낱말이다.
    """
    result = []
    for i in range(len(word)):
        for c in 'abcdefghijklmnopqrstuvwxyz':
            if c != word[i]:
                candidate = word[:i] + c + word[i + 1:]
                if candidate in dictionary:
                    result.append(candidate)
    return result

def bfs_word_ladder(start, goal, dictionary):
    """
    속뜻 낱말 사다리 그래프에서의 BFS.

    가장 짧은 바꿈의 길이를 되돌리고, 바꿀 수 없으면
    -1을 되돌린다.
    """
    if start == goal:
        return 0
    dict_set = set(dictionary)
    visited = {start}
    queue = deque([(start, 0)])

    while queue:
        word, dist = queue.popleft()
        for neighbor in word_neighbors(word, dict_set):
            if neighbor == goal:
                return dist + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return -1

# === 메인 ===

if __name__ == "__main__":
    # 격자 BFS: 걸림돌이 있는 5x5 격자
    blocked = {(1, 1), (1, 2), (1, 3), (2, 1)}
    dist = bfs_grid(5, 5, (0, 0), (4, 4), blocked)
    print(f"Grid shortest path (0,0)->(4,4): {dist} steps")

    # 닿을 수 없는 목표
    wall = {(r, 2) for r in range(5)}  # 열 전체가 벽
    dist2 = bfs_grid(5, 5, (0, 0), (0, 4), wall)
    print(f"Grid with wall: {dist2} (unreachable)")

    # 낱말 사다리
    dictionary = ["hit", "hot", "dot", "dog", "lot", "log", "cog"]
    steps = bfs_word_ladder("hit", "cog", dictionary)
    print(f"\nWord ladder 'hit' -> 'cog': {steps} transformations")
```

**출력:**
```
Grid shortest path (0,0)->(4,4): 8 steps
Grid with wall: -1 (unreachable)
Word ladder 'hit' -> 'cog': 4 transformations
```

---

## 4. 복잡도에서 살필 점

속뜻 그래프에서는 표준 복잡도 잣대가 달라진다:

- **공간.** 그래프 전체를 저장하지 않는다. BFS이나 DFS은 다녀간 묶음을 저장하며 그 크기는 $O(|V_{\text{reachable}}|)$까지 자란다.
- **시간.** 꼭짓점마다 한 번 들르고 이웃을 셈하는 데 $T_{\text{neighbors}}$이 든다. 전체 시간은 $O(|V_{\text{reachable}}| \cdot T_{\text{neighbors}})$이다.

격자 그래프에서는 $T_{\text{neighbors}} = O(1)$이므로(이웃이 많아야 4개) BFS은 $V = m \times n$일 때 $O(V)$에 돈다.

길이 $L$의 낱말과 크기 $D$의 사전을 갖는 낱말 사다리 그래프에서는 꼭짓점마다 $T_{\text{neighbors}} = O(26L)$이고, 닿을 수 있는 꼭짓점의 총 개수는 많아야 $D$이다.

!!! warning "끝없는 상태 공간"
    상태 공간이 끝없으면(이를테면 끝없는 격자, 수학 퍼즐) BFS은 최단 경로를 찾음을 보장하지만 풀이가 없으면 끝없이 돌 수 있다. 되풀이 깊이 늘리기 DFS(IDDFS)은 그런 문제에서 BFS의 가장 좋음과 DFS의 공간 효율을 합친다. [되풀이 깊이 늘리기](../traversals/iddfs.md)를 보아라.

---

## 연습문제

**연습문제 1.**
15-퍼즐(4x4 미끄럼 타일 퍼즐)의 속뜻 그래프를 정의하여라. 꼭짓점, 변, 갈라짐 인자는 무엇인가?

??? success "연습문제 1 풀이"
    **꼭짓점**: 꼭짓점마다 자리 16곳의 꼴이다(번호 붙은 타일 15개와 빈칸 하나). 있을 수 있는 꼴은 $16!$가지이지만 주어진 시작에서 닿을 수 있는 것은 그 절반뿐이다.

    **변**: 한 꼴에서 타일 하나를 빈칸으로 밀어 다른 꼴을 얻을 수 있으면 두 꼴이 이어진다. 빈칸은 (가장자리가 아니면) 위, 아래, 왼쪽, 오른쪽으로 움직일 수 있다.

    **갈라짐 인자**: 빈칸이 모서리에 있으면 2가지, 가장자리에 있으면 3가지, 안쪽에 있으면 4가지로 움직인다. 평균 갈라짐 인자는 대략 $\frac{4 \cdot 4 + 8 \cdot 3 + 4 \cdot 2}{16} = \frac{16 + 24 + 8}{16} = 3$이다. $\square$

---

**연습문제 2.**
속뜻 그래프에서 BFS이 왜 $O(b^d)$ 기억 공간을 쓰는지($b$은 갈라짐 인자, $d$은 풀이의 깊이) 설명하고, DFS은 왜 $O(bd)$만 쓰는지 설명하여라.

??? success "연습문제 2 풀이"
    BFS은 지금 앞자락 층의 모든 꼭짓점을 담은 줄을 지킨다. 깊이 $d$에서 앞자락에는 꼭짓점이 $b^d$개까지 들 수 있다. DFS은 뿌리에서 지금 살펴보는 가장 깊은 마디까지의 길 위 꼭짓점과, 층마다 아직 살펴보지 않은 형제만 지킨다. 길의 깊이는 많아야 $d$이고 층마다 살펴보지 않은 형제가 많아야 $b - 1$개이므로 기억 공간은 $O(bd)$이다. $\square$

---

**연습문제 3.**
낱말 사다리 퍼즐은 글자 하나만 다른 낱말을 잇는다. 이를 속뜻 그래프로 본뜨고 "cold"에서 "warm"으로 가는 가장 짧은 바꿈을 찾는 법을 밝혀라.

??? success "연습문제 3 풀이"
    **꼭짓점**: 길이가 같은(여기서는 4글자) 올바른 영어 낱말 전부. **변**: 글자 자리 꼭 하나가 다른 두 낱말은 이웃이다. **알고리즘**: "cold"에서 BFS을 돌리되, 자리 4곳마다 글자 26개를 모두 넣어 보고 사전에 있는 낱말만 남겨 이웃을 만든다. BFS이 최단 경로(바꿈 횟수의 최솟값)를 찾는다. 이를테면 cold $\to$ cord $\to$ card $\to$ ward $\to$ warm(네 걸음)이다. 이웃 만드는 함수가 드러낸 변 저장을 대신한다. $\square$

---

**연습문제 4.**
많은 속뜻 그래프 문제에서 왜 A* 찾기를 BFS보다 낫게 여기는지 이야기하여라. 어림짐작은 어떤 성질을 만족해야 하는가?

??? success "연습문제 4 풀이"
    BFS은 거리 $d + 1$의 꼭짓점을 보기 전에 거리 $d$의 꼭짓점을 모두 살펴보며 꼭짓점 $O(b^d)$개를 다녀간다. A*은 $v$에서 목표까지 남은 값을 어림하는 어림짐작 $h(v)$을 써서 $f(v) = g(v) + h(v)$이 작은 꼭짓점을 앞세운다. $h$이 **받아들일 만하고**(참값을 결코 넘겨 어림하지 않고) **한결같으면**(변마다 $h(u) \leq c(u,v) + h(v)$), A*은 가장 좋은 경로를 찾으면서도 보통 BFS보다 훨씬 적은 꼭짓점만 살펴본다. 어림짐작이 찾기를 목표 쪽으로 이끌기 때문이다. $\square$

## 정리하며

이 마당은 정의、속뜻 그래프가 왜 중요한가、격자 그래프、복잡도에서 살필 점을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. 22장.
- Russell, S. J., & Norvig, P. (2021). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson. Chapter 3.
