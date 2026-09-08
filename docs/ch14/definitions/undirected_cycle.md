# 무방향 그래프에서 고리 알아내기

무방향 그래프에서 고리를 알아내는 일은 기본적인 짜임의 물음에 답한다. 곧 이 그래프가 숲(나무의 모음)인가, 아니면 고리가 적어도 하나 있는가? 나무는 이어진 덩이마다 변이 꼭 $|V| - 1$개이고 군더더기 이음이 없는 반면, 고리가 있다는 것은 변이 적어도 하나 "남는다"는 뜻이다. 표준적인 길이 둘 있다. 곧 DFS 기반 어버이 좇기와 합치기-찾기이며, 둘 다 거의 선형 시간에 돈다.

---

## 1. 무방향 그래프에서 고리의 정의

무방향 그래프 $G = (V, E)$의 **고리**는 $k \geq 3$이고 꼭짓점 $v_0, v_1, \ldots, v_{k-1}$이 모두 서로 다른 닫힌 길 $v_0, v_1, \ldots, v_{k} = v_0$이다. 제약 $k \geq 3$은 변 하나를 "고리"로 세지 않게 한다(단순 그래프에서 변 하나를 오갔다고 고리가 되지는 않는다).

!!! tip "정리: 나무의 성격"
    꼭짓점 $n$개의 이어진 무방향 그래프 $G$이 나무일 때 그리고 그때만 $G$의 변이 꼭 $n - 1$개이다. 마찬가지로 $G$이 나무일 때 그리고 그때만 $G$이 이어져 있고 고리가 없다.

나무에 변을 하나 더하면 고리가 꼭 하나 생기고, 나무에서 변을 하나 지우면 끊어진다.

---

## 2. 방법 1: 어버이를 좇는 DFS

무방향 그래프에서 DFS을 하면 변마다 양끝에서 살펴본다. 꼭짓점 $u$을 들러 이웃 $v$을 살필 때 세 경우가 있다:

1. $v$을 아직 들르지 않았다: $u$을 어버이로 삼아 $v$에서 DFS을 이어 간다.
2. $v$을 이미 들렀고 $v$이 $u$의 어버이이다: 우리가 타고 온 바로 그 변이므로 건너뛴다.
3. $v$을 이미 들렀고 $v$이 $u$의 어버이가 **아니다**: 고리가 있다. DFS 나무에서 $v$에서 $u$까지의 길에 변 $\{u, v\}$을 이으면 고리가 된다.

### 복잡도

꼭짓점과 변마다 많아야 두 번 살피므로(양끝에서 한 번씩) 전체 시간은 $O(V + E)$이고, 다녀감 배열과 되돌이 더미에 $O(V)$ 공간이 든다.

$$
T(V, E) = O(V + E)
$$

```python
"""
어버이를 좇는 DFS으로 무방향 그래프에서 고리 알아내기.

DFS이 지금 꼭짓점의 어버이가 아닌, 이미 다녀간 이웃을
만나면 고리를 찾은 것이다.
"""

# === DFS 기반 고리 알아내기 ===

def has_cycle_dfs(adj, n):
    """
    DFS으로 무방향 그래프에서 고리를 알아낸다.

    고리가 하나라도 있으면 True, 아니면 False을 되돌린다.
    모든 덩이를 훑어 끊어진 그래프도 다룬다.
    """
    visited = [False] * n

    def dfs(u, parent):
        visited[u] = True
        for v in adj[u]:
            if not visited[v]:
                if dfs(v, u):
                    return True
            elif v != parent:
                return True  # 어버이가 아닌 곳으로 가는 되돌이 변 => 고리
        return False

    for u in range(n):
        if not visited[u]:
            if dfs(u, -1):
                return True
    return False

# === 메인 ===

if __name__ == "__main__":
    # 고리가 있는 그래프: 0-1-2-0
    adj_cycle = [[1, 2], [0, 2], [0, 1]]
    print(f"Triangle has cycle: {has_cycle_dfs(adj_cycle, 3)}")

    # 나무: 0-1, 1-2, 1-3
    adj_tree = [[1], [0, 2, 3], [1], [1]]
    print(f"Tree has cycle: {has_cycle_dfs(adj_tree, 4)}")

    # 끊어짐: 나무 + 외톨이 꼭짓점
    adj_disconnected = [[1], [0, 2], [1], []]
    print(f"Disconnected tree has cycle: "
          f"{has_cycle_dfs(adj_disconnected, 4)}")
```

**출력:**
```
Triangle has cycle: True
Tree has cycle: False
Disconnected tree has cycle: False
```

---

## 3. 방법 2: 합치기-찾기

합치기-찾기(서로소 집합) 길은 변을 하나씩 다룬다. 변 $\{u, v\}$마다:

- $u$과 $v$이 다른 집합에 있으면 둘을 합친다.
- $u$과 $v$이 이미 같은 집합에 있으면 이 변이 고리를 만든다.

높이로 합치기와 길 누르기를 쓰면 연산마다 고르게 나눠 $O(\alpha(n))$ 시간이 들며, 여기서 $\alpha$은 애커만 함수의 역함수이다. 전체 시간은 $O(E \cdot \alpha(V))$으로 사실상 선형이다.

```python
"""
합치기-찾기로 무방향 그래프에서 고리 알아내기.

변을 하나씩 다룬다. 양끝이 이미 같은 덩이에 있으면
그 변이 고리를 만든다.
"""

# === 합치기-찾기 ===

class UnionFind:
    """높이로 합치기와 길 누르기를 쓴 서로소 집합."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        """길 누르기로 뿌리를 찾는다."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """높이로 합친다. 이미 같은 집합이면(고리이면) False을 되돌린다."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False  # 고리 찾음
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

# === 고리 알아내기 ===

def has_cycle_union_find(n, edges):
    """
    합치기-찾기로 고리를 알아낸다.

    이미 같은 덩이에 있는 꼭짓점 둘을 잇는 변이 하나라도 있으면
    True를 되돌린다.
    """
    uf = UnionFind(n)
    for u, v in edges:
        if not uf.union(u, v):
            return True
    return False

# === 메인 ===

if __name__ == "__main__":
    # 고리가 있는 그래프
    print(f"Triangle has cycle: "
          f"{has_cycle_union_find(3, [(0,1), (1,2), (2,0)])}")

    # 나무
    print(f"Tree has cycle: "
          f"{has_cycle_union_find(4, [(0,1), (1,2), (1,3)])}")
```

**출력:**
```
Triangle has cycle: True
Tree has cycle: False
```

---

## 4. 방법의 견줌

| 결 | DFS | 합치기-찾기 |
|---|---|---|
| 시간 복잡도 | $O(V + E)$ | $O(E \cdot \alpha(V))$ |
| 공간 | $O(V)$ | $O(V)$ |
| 고리 경로를 찾나 | 예(어버이 좇기로) | 아니오(있는지만 알아냄) |
| 끊어진 그래프 다룸 | 예(모든 꼭짓점을 훑는다) | 예(저절로) |
| 온라인(변이 흘러들 때) | 아니오(온전한 이웃 목록이 필요) | 예 |

!!! warning "겹 변과 제 고리"
    DFS 어버이 좇기 길은 단순 그래프를 놓고 한다. 같은 꼭짓점 짝 사이에 겹 변이 있으면 어버이 살피기가 어버이 꼭짓점만이 아니라 특정 변의 첨자를 좇아야 한다. 제 고리는 그 자체로 고리이다.

---

## 5. 나무와 숲과의 이음

고리 알아내기의 결과는 숲과 나무의 성질과 곧바로 이어진다:

- 꼭짓점 $n$개의 이어진 그래프가 나무일 때 그리고 그때만 변이 꼭 $n - 1$개이고 고리가 없다.
- 그래프가 **숲**일 때 그리고 그때만 이어진 덩이마다 나무이다. 마찬가지로 그래프에 고리가 없다.
- [방향 고리 알아내기](directed_cycle.md)에서는 변의 방향이 중요하므로 알고리즘이 다르다. 곧 DFS이 어버이 좇기 대신 세 빛깔 표시를 쓴다.

---

## 연습문제

**연습문제 1.**
이어진 무방향 그래프에 꼭짓점 $n$개와 변 $n$개가 있다. 고리가 꼭 하나 있음을 증명하여라.

??? success "연습문제 1 풀이"
    꼭짓점 $n$개의 이어진 그래프에는 적어도 변 $n - 1$개가 필요하다(뻗음 나무). $n$번째 변은 이미 나무 안에 있는 꼭짓점 둘을 이어 고리를 꼭 하나 만든다. 고리가 하나뿐임은 이렇게 본다. 그 고리 위 아무 변이나 지우면 뻗음 나무(이어져 있고 변 $n - 1$개)가 되고 여기에는 고리가 없다. 만약 서로 다른 둘째 고리가 있었다면 첫 고리에서 변 하나를 지워도 고리가 남아, 뻗음 나무에 고리가 없다는 사실과 어긋난다. $\square$

---

**연습문제 2.**
찾은 고리의 꼭짓점을 실제로 내놓도록 DFS 기반 고리 알아내기 알고리즘을 고치는 법을 밝혀라.

??? success "연습문제 2 풀이"
    DFS 도중 어버이 배열을 지킨다. 되돌이 변 $(u, v)$을 찾으면($v$을 이미 들렀고 $v \neq \text{parent}[u]$일 때) $u$에서 $v$까지 어버이 가리개를 거슬러 올라가며 고리를 되살린다:

    ```python
    def find_cycle(adj, n):
        visited = [False] * n
        parent = [-1] * n
        cycle = []

        def dfs(u):
            visited[u] = True
            for v in adj[u]:
                if not visited[v]:
                    parent[v] = u
                    result = dfs(v)
                    if result:
                        return result
                elif v != parent[u]:
                    # 고리 되살리기
                    path = [v]
                    cur = u
                    while cur != v:
                        path.append(cur)
                        cur = parent[cur]
                    path.append(v)
                    return path
            return None

        for s in range(n):
            if not visited[s]:
                result = dfs(s)
                if result:
                    return result
        return []
    ```
    $\square$

---

**연습문제 3.**
DFS 기반과 합치기-찾기 기반 고리 알아내기가 저마다 어떤 정보를 더 주는지 견주어라.

??? success "연습문제 3 풀이"
    **DFS 기반**: (어버이 거슬러 가기로) 실제 고리의 꼭짓점을 자연스럽게 준다. 또 모든 변을 갈래 나누고(나무 변과 되돌이 변) 모든 고리를 찾을 수 있다. $O(V + E)$ 시간과 되돌이 더미에 $O(V)$ 공간이 든다.

    **합치기-찾기 기반**: 변을 하나씩 다루며 고리를 만드는 첫 변(양끝이 이미 같은 덩이에 있는 변)을 알아낸다. 어느 변이 고리를 닫는지는 가려내지만 고리의 꼭짓점을 곧바로 주지는 않는다. 높이로 합치기와 길 누르기를 쓰면 연산마다 고르게 나눠 $O(\alpha(n))$ 시간이 들어 전체가 $O(E \cdot \alpha(V))$이다. 합치기-찾기는 되풀이로 구현하기가 더 쉽고 되돌이 깊이 문제를 피한다. $\square$

---

**연습문제 4.**
꼭짓점 $n$개에 변이 적어도 $n$개인 무방향 그래프에 반드시 고리가 있음을 증명하여라.

??? success "연습문제 4 풀이"
    꼭짓점 $n$개와 이어진 덩이 $c$개의 고리 없는 무방향 그래프(숲)에는 변이 꼭 $n - c$개 있다. $c \geq 1$이므로 고리 없는 그래프의 변은 많아야 $n - 1$개이다. 그러므로 $|E| \geq n$이면 그 그래프에 고리가 없을 수 없고 적어도 하나 있어야 한다. $\square$

---

**연습 5.**
무방향 그래프에 길이가 홀수인 고리가 있는지 알아내는 알고리즘을 짜라. 이것이 그 그래프에 대해 무엇을 알려 주는가?

??? success "연습 5의 풀이"
    (홀수 고리 성격 정리에 따라) 무방향 그래프에 길이가 홀수인 고리가 있을 때 그리고 그때만 그 그래프가 이분이 아니다. 그러므로 BFS 기반 두 빛깔 칠하기를 돌린다. 곧 층마다 빛깔을 번갈아 준다. 어떤 변이 같은 빛깔의 꼭짓점 둘을 이으면 홀수 고리가 있고 그 그래프는 이분이 아니다. 두 빛깔 칠하기가 잘되면 홀수 고리가 없다. 이 알고리즘은 $O(V + E)$ 시간에 돈다. $\square$

## 정리하며

이 마당은 무방향 그래프에서 고리의 정의、방법 1: 어버이를 좇는 DFS、방법 2: 합치기-찾기、방법의 견줌을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. 22.3절, 21.1-21.3절.
