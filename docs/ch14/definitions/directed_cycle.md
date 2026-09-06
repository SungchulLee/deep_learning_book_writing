# 방향 그래프에서 고리 알아내기

방향 그래프에서 고리를 알아내는 일은 기댐 짜임이 올바른지 가리는 데 꼭 필요하다. 고리가 없는 방향 그래프는 DAG(방향 비순환 그래프)이며, 위상 정렬을 할 수 있고 일정 짜기, 빌드 얼개, 선수 조건 살피기의 바탕이 된다. 표준 알고리즘은 세 빛깔 표시 방식을 쓴 DFS으로 되돌이 변을 가려내며, 되돌이 변마다 고리가 있음을 알린다.

## 방향 고리

방향 그래프 $G = (V, E)$의 **방향 고리**는 모든 $i$에 대해 $(v_i, v_{i+1 \bmod k}) \in E$인, $k \geq 2$이고 서로 다른 꼭짓점의 늘어놓음 $v_0, v_1, \ldots, v_{k-1}$이다. 다시 말해 방향 변이 닫힌 고리를 이룬다.

방향 고리가 없는 방향 그래프를 **방향 비순환 그래프(DAG)**라 한다.

!!! tip "정리: DAG의 성격"
    방향 그래프 $G$이 DAG일 때 그리고 그때만 $G$의 DFS이 되돌이 변을 만들지 않는다.

DFS의 **되돌이 변**은 DFS 나무에서 $v$이 $u$의 조상인 변 $(u, v)$이다. 곧 $v$을 $u$보다 먼저 찾았고 $v$이 마치기 전에 $u$을 찾았다는 뜻이다. 그런 변은 $v$에서 DFS 나무를 타고 $u$까지 내려간 뒤 되돌이 변으로 $v$에 돌아오는 고리를 닫는다.

## 세 빛깔 알고리즘

표준 알고리즘은 꼭짓점마다 빛깔 상태를 지킨다:

- **흰색**: 찾지 않음 — 아직 들르지 않은 꼭짓점.
- **잿빛**: 하는 중 — 지금 DFS 되돌이 더미에 있는 꼭짓점(찾았으나 마치지 않음).
- **검은색**: 마침 — 이 꼭짓점의 자손을 모두 온전히 살펴봄.

DFS이 잿빛 꼭짓점에서 다른 잿빛 꼭짓점으로 가는 변(되돌이 변)을 만날 때 그리고 그때만 고리가 있다. 꼭짓점 $u$이 잿빛인데 $v$도 잿빛인 변 $(u, v)$을 찾으면, DFS 나무에서 $v$에서 $u$까지의 길에 변 $(u, v)$을 이어 방향 고리가 된다.

### 올바름의 논증

DFS 도중 잿빛 꼭짓점은 지금 DFS 나무의 뿌리에서 다루는 꼭짓점까지의 길을 이룬다. 변 $(u, v)$을 만났는데 $v$이 잿빛이면 $v$은 이 길 위에서 $u$의 조상이므로 길 $v \to \cdots \to u \to v$이 방향 고리이다. 거꾸로 꼭짓점 $v_0 \to v_1 \to \cdots \to v_{k-1} \to v_0$의 방향 고리가 있으면, DFS이 그 고리에서 가장 먼저 찾은 꼭짓점은 고리를 닫는 되돌이 변에 이를 때 잿빛일 것이다.

### 복잡도

이 알고리즘은 꼭짓점마다 한 번 들르고 변마다 한 번 살피므로 시간 복잡도가 $O(V + E)$이고, 빛깔 배열에 $O(V)$과 되돌이 더미에 $O(V)$의 공간이 든다.

$$
T(V, E) = O(V + E)
$$

## 구현

```python
"""
세 빛깔 표시를 쓴 DFS으로 방향 고리 알아내기.

깊이 우선 찾기 도중 꼭짓점 상태(흰색, 잿빛, 검은색)를 좇아
방향 그래프에 고리가 있는지 알아낸다.
잿빛 꼭짓점으로 가는 되돌이 변은 고리가 있음을 알린다.
"""


# === 상수 ===

WHITE, GRAY, BLACK = 0, 1, 2


# === 고리 알아내기 ===

def has_cycle_directed(adj, n):
    """
    방향 그래프에 고리가 있는지 알아낸다.

    세 빛깔 표시를 쓴 DFS을 쓴다. 고리가 있으면 True,
    그래프가 DAG이면 False을 되돌린다.
    """
    color = [WHITE] * n

    def dfs(u):
        color[u] = GRAY
        for v in adj[u]:
            if color[v] == GRAY:
                return True  # 되돌이 변 찾음
            if color[v] == WHITE and dfs(v):
                return True
        color[u] = BLACK
        return False

    for u in range(n):
        if color[u] == WHITE:
            if dfs(u):
                return True
    return False


def find_cycle_directed(adj, n):
    """
    방향 고리 하나를 찾아 되돌리고, 없으면 빈 목록을 되돌린다.

    되돌이 변을 찾았을 때 고리 경로를 되살리려고
    DFS 어버이를 좇는다.
    """
    color = [WHITE] * n
    parent = [-1] * n
    cycle = []

    def dfs(u):
        color[u] = GRAY
        for v in adj[u]:
            if color[v] == GRAY:
                # v에서 u까지, 다시 v로 돌아오는 고리 되살리기
                path = [v]
                cur = u
                while cur != v:
                    path.append(cur)
                    cur = parent[cur]
                path.append(v)
                path.reverse()
                cycle.extend(path)
                return True
            if color[v] == WHITE:
                parent[v] = u
                if dfs(v):
                    return True
        color[u] = BLACK
        return False

    for u in range(n):
        if color[u] == WHITE:
            if dfs(u):
                return cycle
    return cycle


# === 메인 ===

if __name__ == "__main__":
    # 고리가 있는 그래프: 0 -> 1 -> 2 -> 0
    adj1 = [[1], [2], [0, 3], []]
    print(f"Graph 1 has cycle: {has_cycle_directed(adj1, 4)}")
    print(f"Cycle found: {find_cycle_directed(adj1, 4)}")

    # DAG: 0 -> 1 -> 3, 0 -> 2 -> 3
    adj2 = [[1, 2], [3], [3], []]
    print(f"\nGraph 2 (DAG) has cycle: {has_cycle_directed(adj2, 4)}")
    print(f"Cycle found: {find_cycle_directed(adj2, 4)}")
```

**출력:**
```
Graph 1 has cycle: True
Cycle found: [0, 1, 2, 0]
Graph 2 (DAG) has cycle: False
Cycle found: []
```

!!! warning "재귀의 깊이"
    되돌이 DFS 구현은 큰 그래프에서 파이썬의 기본 되돌이 한도에 부딪힐 수 있다. 실제 제품에서는 드러낸 더미를 쓴 되풀이 판으로 바꾸거나 `sys.setrecursionlimit`으로 한도를 올려라.

## 위상 정렬과의 이음

방향 그래프가 [위상 차례](../../ch17/topological/dag.md)를 가질 때 그리고 그때만 그것이 DAG이다. 고리 알아내기 알고리즘을 넓혀 위상 정렬을 낼 수 있다. 곧 DFS이 꼭짓점을 마칠 때(검게 칠할 때) 그것을 날 목록의 앞에 붙인다. 되돌이 변을 찾지 못하면 나온 목록이 올바른 위상 차례이다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. 22.4절.

## 연습문제

**연습문제 1.**
무방향 그래프에서는 되는데 왜 방향 그래프에서 고리를 알아내는 데는 합치기-찾기를 쓸 수 없는가?

??? success "연습문제 1 풀이"
    합치기-찾기는 변의 양끝이 이미 같은 이어진 덩이에 있는지 살펴 무방향 그래프의 고리를 알아낸다. 방향 그래프에서는 변의 방향이 중요하다. 곧 $(u, v)$과 $(v, u)$은 뜻이 다른 서로 다른 변이다. 꼭짓점 둘이 방향 고리를 이루지 않고도 같은 약하게 이어진 덩이에 들 수 있다. 이를테면 $0 \to 1 \to 2$은 꼭짓점 셋이 모두 이어져 있지만 고리는 없다. 합치기-찾기는 변의 방향을 무시하므로 이 갈림을 놓친다. 되돌이 더미를 좇아 되돌이 변을 알아내는 세 빛깔 DFS이 필요하다. $\square$

---

**연습문제 2.**
되돌이 세 빛깔 고리 알아내기 알고리즘을 드러낸 더미를 쓴 되풀이 판으로 바꿔라. 유사 코드를 적어라.

??? success "연습문제 2 풀이"
    ```python
    def has_cycle_iterative(adj, n):
        WHITE, GRAY, BLACK = 0, 1, 2
        color = [WHITE] * n
        for s in range(n):
            if color[s] != WHITE:
                continue
            stack = [(s, 0)]  # (꼭짓점, 이웃 첨자)
            color[s] = GRAY
            while stack:
                u, idx = stack.pop()
                if idx < len(adj[u]):
                    stack.append((u, idx + 1))
                    v = adj[u][idx]
                    if color[v] == GRAY:
                        return True
                    if color[v] == WHITE:
                        color[v] = GRAY
                        stack.append((v, 0))
                else:
                    color[u] = BLACK
        return False
    ```

    핵심은 다음 이웃을 살피기 전에 이웃 첨자를 하나 올린 채 지금 꼭짓점을 다시 밀어 넣어 되돌이 더미를 흉내 내는 것이다. $\square$

---

**연습문제 3.**
방향 그래프의 DFS이 되돌이 변을 찾지 못하면 DFS 마친 차례의 거꾸로가 올바른 위상 정렬임을 증명하여라.

??? success "연습문제 3 풀이"
    그래프의 아무 변 $(u, v)$을 생각하자. 되돌이 변이 없으므로 $v$은 DFS 나무에서 $u$의 조상이 아니다. 세 경우가 있다. (1) $v$이 $u$의 자손이므로 $v$이 $u$보다 먼저 마친다. (2) $(u, v)$이 이미 마친 부분 나무로 가는 가로 변이므로 $v$이 $u$보다 먼저 마친다. (3) $(u, v)$이 앞선 변이므로 $v$이 $u$보다 먼저 마친다. 어느 경우든 $v$이 $u$보다 먼저 마치므로 마친 차례의 거꾸로에서 $u$이 $v$보다 앞에 나온다. 이것이 변마다 성립하므로 마친 차례의 거꾸로가 올바른 위상 정렬이다. $\square$

---

**연습문제 4.**
방향 그래프에 꼭짓점 $n$개와 변 $m$개가 있다. DFS 한 번이 만날 수 있는 되돌이 변은 많아야 몇 개인가? 이 최댓값에 이르는 보기를 들어라.

??? success "연습문제 4 풀이"
    나무 변도 앞선 변도 가로 변도 아닌 변은 모두 되돌이 변이다. 꼭짓점 $n$개의 DFS 나무에는 (이어진 덩이 안에서) 나무 변이 꼭 $n - 1$개 있다. 남은 변 $m - (n - 1)$개는 되돌이 변, 앞선 변, 가로 변이 될 수 있다. 되돌이 변을 가장 많게 하려면 그래프를 고리 하나에 뒤로 가리키는 지름길 변을 더한 꼴로 두면 된다. 이를테면 꼭짓점 $n$개의 완전 그래프에서 변을 모두 낮은 첨자에서 높은 첨자로 향하게 하고 $n-1$에서 $0$으로 가는 되돌이 변을 더한다. 긴 고리 하나 $0 \to 1 \to \cdots \to n-1 \to 0$은 변 $n$개, 나무 변 $n-1$개, 되돌이 변 1개를 갖는다. 일반으로 되돌이 변의 최대 개수는 $m - n + 1$이다(나무가 아닌 변이 모두 되돌이 변일 때). $\square$

---

**연습 5.**
고리가 있음이 알려진 방향 그래프가 주어졌을 때 가장 짧은 방향 고리(변이 가장 적은 것)를 찾는 $O(V + E)$ 알고리즘을 밝혀라.

??? success "연습 5의 풀이"
    꼭짓점 $v$마다 BFS을 돌려 $v$에서 자기 자신으로 돌아오는 최단 경로를 찾는다. 꼭짓점 $v$마다 $v$을 지나는 가장 짧은 고리의 길이는 방향 그래프에서 $v$에서 $v$까지의 BFS 거리와 같다. 전체에서 가장 짧은 고리는 모든 시작 꼭짓점에 걸친 최솟값이다. BFS마다 $O(V + E)$이 들고 시작 꼭짓점이 $V$개이므로 전체 시간은 $O(V(V + E))$이다. 무게 없는 그래프에서 참으로 $O(V + E)$인 길을 원한다면 DFS 나무 짜임을 쓸 수 있다. 곧 되돌이 변 $(u, v)$마다 고리 길이가 $\text{depth}(u) - \text{depth}(v) + 1$이고, 모든 되돌이 변에 걸친 최솟값이 가장 짧은 고리를 준다. 이는 DFS 한 번으로 $O(V + E)$에 된다. $\square$
