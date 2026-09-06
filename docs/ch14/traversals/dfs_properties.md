# DFS의 성질

깊이 우선 찾기에는 꼭짓점을 모두 들르는 것을 훌쩍 넘어서는 짜임의 성질이 있다. DFS의 되돌이 성격은 찾은 때와 마친 때에 겹쳐 넣기 짜임을 새기며, 이를 **괄호 정리**가 담는다. **흰 길 정리**와 합치면 이 성질들이 왜 DFS이 고리 알아내기, 위상 정렬, 강하게 이어진 덩이 알고리즘의 엔진인지 알려 준다.

## DFS 숲

(무방향에서) 이어져 있지 않거나 (방향에서) 강하게 이어져 있지 않은 그래프에서 DFS을 돌리면 샘 꼭짓점 하나로 다른 모든 꼭짓점에 닿을 수 없다. 이를 다루려고 DFS은 모든 꼭짓점을 훑으며 다녀가지 않은 꼭짓점마다 새 찾기를 띄운다. 그 결과가 **DFS 숲**이다. 곧 이어진 덩이나 서로 닿을 수 있는 꼭짓점 무리마다 하나씩인 뿌리 있는 나무의 모음이다. 나무 변 $(u, v)$마다 DFS이 $u$에서 $v$을 처음 찾은 그때에 해당한다.

## 괄호 정리

괄호 정리는 DFS에 대한 근본적인 짜임 결과이다. 이는 찾은 때와 마친 때가 왜 DFS 나무의 조상-자손 관계 전체를 담는지 알려 준다.

DFS 도중 꼭짓점 $u$마다 찾은 때 $\text{pre}(u)$과 마친 때 $\text{post}(u)$을 받는다. 괄호 정리는 아무 꼭짓점 둘 $u$과 $v$에 대해 다음 가운데 꼭 하나가 성립한다고 말한다:

1. 구간 $[\text{pre}(u), \text{post}(u)]$과 $[\text{pre}(v), \text{post}(v)]$이 **온통 겹치지 않는다** — DFS 나무에서 어느 쪽도 다른 쪽의 조상이 아니다.
2. $[\text{pre}(u), \text{post}(u)] \subset [\text{pre}(v), \text{post}(v)]$ — 꼭짓점 $u$이 $v$의 자손이다.
3. $[\text{pre}(v), \text{post}(v)] \subset [\text{pre}(u), \text{post}(u)]$ — 꼭짓점 $v$이 $u$의 자손이다.

구간은 결코 얼마쯤만 겹치지 않는다. 이는 식 속의 짝 맞은 괄호와 같다. 곧 여는 괄호마다 짝이 되는 닫는 괄호가 있고, 짝들은 겹쳐 넣어져 있거나 서로 떨어져 있다.

**증명 얼개.** $\text{pre}(u) < \text{pre}(v)$인 꼭짓점 $u$과 $v$을 생각하자. $u$을 아직 살펴보는 도중에(곧 $u$이 마치기 전에) $v$을 찾았다면 $v$은 $u$의 자손이어야 한다. DFS은 $u$으로 돌아오기 전에 $v$을 마치므로 $\text{post}(v) < \text{post}(u)$이고 담김이 된다. 반대로 $u$이 마친 뒤에 $v$을 찾았다면 $\text{post}(u) < \text{pre}(v)$이므로 서로 떨어진다. 다른 경우는 있을 수 없다. $\square$

!!! tip "상수 시간 조상 검정"
    DFS 나무에서 꼭짓점 $u$이 $v$의 조상일 때 그리고 그때만 $\text{pre}(u) \leq \text{pre}(v)$이고 $\text{post}(v) \leq \text{post}(u)$이다. 이로써 $O(V + E)$ DFS 한 번 뒤에 조상 여부를 $O(1)$에 살필 수 있다.

## 흰 길 정리

괄호 정리는 시각 도장의 짜임을 밝히고, 흰 길 정리는 DFS 나무의 짜임을 그래프의 실제 변과 잇는다. 둘을 합치면 어느 꼭짓점이 어느 꼭짓점의 자손이 되는지 온전히 밝혀진다.

DFS 숲에서 꼭짓점 $v$이 $u$의 자손일 때 그리고 그때만, $u$을 찾은 그때에 $u$에서 $v$까지 온통 **흰**(다녀가지 않은) 꼭짓점으로만 이루어진 길이 있다. 이 정리는 고리 알아내기를 위한 되돌이 변의 성격 밝히기처럼 DFS 차례에 기대는 알고리즘의 맞음을 증명하는 데 꼭 필요하다.

## 시간과 공간 복잡도

DFS은 꼭짓점마다 한 번 들르고 변마다 (방향 그래프에서는) 한 번, (무방향 그래프에서는) 두 번 살핀다. 그러므로 시간 복잡도는 다음과 같다

$$
O(V + E)
$$

다녀간 묶음에 드는 공간 복잡도는 $O(V)$이다. 되돌이 더미(되풀이 판에서는 드러낸 더미)는 최악의 경우(길 그래프) $O(V)$까지 자라므로 보조 공간은 통틀어 $O(V)$이다.

## 되돌이 DFS과 되풀이 DFS

되돌이 구현은 수학의 정의를 곧바로 비추고, 되풀이 판은 큰 그래프에서 파이썬의 되돌이 한도에 부딪히지 않도록 드러낸 더미를 쓴다. 둘 다 같은 DFS 숲을 만들지만, 되풀이 판은 이웃 목록을 훑는 방식에 따라 이웃을 다른 차례로 들를 수 있다.

```python
"""
DFS 구현: 되돌이와 되풀이.

앞/뒤 시각 도장으로 괄호 정리를 보이고
되돌이 길과 되풀이 길을 모두 보인다.
"""

# === 시각 도장을 찍는 되돌이 DFS ============================================

def dfs_recursive(graph):
    """되돌이 DFS을 돌리고 앞/뒤 시각 도장을 되돌린다.

    매개변수
    ----------
    graph : dict[int, list[int]]
        방향 그래프의 이웃 목록.

    반환값
    -------
    pre : dict[int, int]
        찾은 때.
    post : dict[int, int]
        마친 때.
    """
    pre = {}
    post = {}
    clock = [0]

    def explore(u):
        clock[0] += 1
        pre[u] = clock[0]
        for v in graph[u]:
            if v not in pre:
                explore(v)
        clock[0] += 1
        post[u] = clock[0]

    for vertex in graph:
        if vertex not in pre:
            explore(vertex)

    return pre, post


# === 되풀이 DFS =============================================================

def dfs_iterative(graph, source):
    """샘 하나에서 시작하는 되풀이 DFS 돌아보기.

    매개변수
    ----------
    graph : dict[int, list[int]]
        이웃 목록.
    source : int
        시작 꼭짓점.

    반환값
    -------
    list[int]
        DFS 들름 차례로 늘어놓은 꼭짓점.
    """
    visited = set()
    stack = [source]
    order = []

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            order.append(node)
            for neighbor in reversed(graph[node]):
                if neighbor not in visited:
                    stack.append(neighbor)

    return order


# === 메인 =====================================================================

if __name__ == "__main__":
    graph = {0: [1, 2], 1: [3], 2: [3], 3: []}

    pre, post = dfs_recursive(graph)
    print("Parenthesis theorem demonstration:")
    for v in sorted(pre):
        print(f"  Vertex {v}: [{pre[v]}, {post[v]}]")

    print(f"\nIterative DFS order: {dfs_iterative(graph, 0)}")
```

**출력:**
```
Parenthesis theorem demonstration:
  Vertex 0: [1, 8]
  Vertex 1: [2, 5]
  Vertex 2: [6, 7]
  Vertex 3: [3, 4]

Iterative DFS order: [0, 1, 3, 2]
```

구간들이 괄호 정리를 확인해 준다. 곧 $[2, 5]$과 $[6, 7]$은 서로 떨어져 있고(꼭짓점 1과 2은 형제), $[3, 4] \subset [2, 5]$이며(꼭짓점 3은 꼭짓점 1의 자손), 모든 구간이 $[1, 8]$ 안에 겹쳐 들어 있다(꼭짓점 0이 뿌리).

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 22. MIT Press.

## 연습문제

**연습문제 1.**
DFS 시각 도장에 대한 괄호 정리를 밝히고 증명하여라.

??? success "연습문제 1 풀이"
    **정리**: DFS의 아무 꼭짓점 둘 $u$과 $v$에 대해 다음 가운데 꼭 하나가 성립한다. (a) $[\text{disc}[u], \text{fin}[u]]$과 $[\text{disc}[v], \text{fin}[v]]$이 서로 떨어져 있다, (b) 한 구간이 다른 구간에 온통 담긴다.

    **증명**: $v$이 마치기 전에 $u$을 찾았다면(곧 $\text{disc}[u] < \text{disc}[v] < \text{fin}[u]$이면) $u$이 잿빛인 동안 $v$을 찾은 것이므로 $v$은 $u$의 자손이다. 그러므로 $v$은 $u$보다 먼저 마쳐야 한다. 곧 $\text{fin}[v] < \text{fin}[u]$이고 담김 $[\text{disc}[v], \text{fin}[v]] \subset [\text{disc}[u], \text{fin}[u]]$이 된다. 어느 구간도 다른 구간 안에서 시작하지 않으면 둘은 서로 떨어져 있어야 한다(얼마쯤만 겹치면 한 꼭짓점이 다른 꼭짓점의 조상이면서 조상이 아니게 되어 있을 수 없다). $\square$

---

**연습문제 2.**
흰 길 정리를 증명하여라. 곧 DFS 숲에서 꼭짓점 $v$이 $u$의 자손일 때 그리고 그때만, $u$을 찾은 그때에 $u$에서 $v$까지 온통 흰(아직 찾지 않은) 꼭짓점으로만 이루어진 길이 있다.

??? success "연습문제 2 풀이"
    $(\Rightarrow)$ $v$이 $u$의 자손이면 $u$에서 $v$까지의 나무 길은 $u$보다 나중에 찾은 꼭짓점으로 이루어진다. 때 $\text{disc}[u]$에 이 꼭짓점들은 모두 아직 희다.

    $(\Leftarrow)$ $u$을 찾을 때 흰 길 $u = w_0, w_1, \ldots, w_k = v$이 있다고 하자. 귀납으로 보인다. $w_1$은 희고 $u$에 이웃하므로 DFS은 $u$을 마치기 전에 $w_1$을 살펴 $w_1$을 자손으로 만든다. $w_1$을 찾을 때 $w_2, \ldots, w_k$은 아직 희다($u$을 찾을 때 희었고 아직 들르지 않았다). 이어 가면 $w_i$마다 $w_{i-1}$의 자손이 되므로 $v$은 $u$의 자손이다. $\square$

---

**연습문제 3.**
최악의 경우 DFS 되돌이 더미의 깊이가 왜 $O(V)$일 수 있는지 설명하여라. 이에 이르는 그래프의 보기를 들어라.

??? success "연습문제 3 풀이"
    꼭짓점 $0, 1, \ldots, n-1$과 변 $(i, i+1)$인 길 그래프 $P_n$이 되돌이 깊이를 가장 크게 만든다. 꼭짓점 0에서 DFS을 하면 1로, 다시 2로 되돌이해 들어가며 되돌아오기 전에 깊이 $n - 1$에 이른다. 되돌이 더미가 꼭짓점 $n$개를 한꺼번에 담으므로 깊이가 $O(V)$이다. 그래서 큰 그래프에는 드러낸 더미를 쓴 되풀이 DFS이 낫다. $\square$

---

**연습문제 4.**
방향 그래프에서 변 $(u, v)$에 대해 $\text{fin}[u] > \text{fin}[v]$이면 그 변이 나무 변, 앞선 변, 가로 변 가운데 하나이고 되돌이 변은 아님을 보여라.

??? success "연습문제 4 풀이"
    되돌이 변 $(u, v)$은 DFS 나무에서 $v$이 $u$의 조상임을 뜻한다. 괄호 정리에 따라 $[\text{disc}[u], \text{fin}[u]] \subset [\text{disc}[v], \text{fin}[v]]$이고 이는 $\text{fin}[u] < \text{fin}[v]$을 뜻한다. 이는 $\text{fin}[u] > \text{fin}[v]$과 어긋난다. 그러므로 $(u, v)$은 되돌이 변일 수 없다. 남는 것은 나무 변($u$에서 $v$을 찾음), 앞선 변($v$이 자손이지만 나무 변을 거치지 않음), 가로 변($v$이 다른 가지에 있고 이미 마쳤음)이다. $\square$
