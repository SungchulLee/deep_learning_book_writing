# 오그린 그래프

아무리 복잡한 방향 그래프라도 그 안에는 드러나기를 기다리는 유향 비순환 그래프 짜임이 숨어 있다. **오그린 그래프**(**강한 이음 조각의 유향 비순환 그래프**라고도 한다)는 [강하게 이어진 조각](definition.md)마다 하나의 큰 꼭짓점으로 뭉개어 얻는다. 그 결과는 늘 유향 비순환 그래프이며, 따라서 위상 정렬, 최단 경로, 동적 계획 같은 유향 비순환 그래프의 힘센 기법을 모두 써서 본디 그래프의 큰 짜임을 살필 수 있다.

## 정의

!!! note "엄밀한 정의"
    $G = (V, E)$을 강한 이음 조각이 $C_1, C_2, \ldots, C_k$인 방향 그래프라 하자. **오그린 그래프** $G^{SCC} = (V^{SCC}, E^{SCC})$을 다음과 같이 정한다:

    - $V^{SCC} = \{C_1, C_2, \ldots, C_k\}$(강한 이음 조각마다 꼭짓점 하나)
    - $i \neq j$이고 $(u, v) \in E$인 꼭짓점 $u \in C_i$, $v \in C_j$이 있을 때 그리고 오직 그때만 $(C_i, C_j) \in E^{SCC}$이다

같은 큰 꼭짓점 짝 사이의 겹치는 변은 하나로 뭉갠다. ($i = j$이어야 하는) 자기 고리는 정의에서 뺀다.

## 오그린 그래프는 유향 비순환 그래프이다

!!! tip "핵심 성질"
    오그린 그래프 $G^{SCC}$은 늘 유향 비순환 그래프이다.

**증명.** 어긋남을 이끌어 내려고 $G^{SCC}$에 방향 순환 $C_{a_1} \to C_{a_2} \to \cdots \to C_{a_m} \to C_{a_1}$이 있다고 하자. 그러면 아무 $u \in C_{a_1}$에서 (조각 사이의 변을 거쳐) $C_{a_2}$의 어떤 꼭짓점에, 그다음 $C_{a_3}$의 어떤 꼭짓점에, 이렇게 나아가 끝내 $C_{a_1}$의 어떤 꼭짓점으로 돌아올 수 있다. 강한 이음 조각 안에서는 모든 꼭짓점이 서로 닿을 수 있으므로 $C_{a_1} \cup C_{a_2} \cup \cdots \cup C_{a_m}$의 모든 꼭짓점이 서로 닿는다. 곧 이들이 모두 한 강한 이음 조각에 든다는 뜻이며, 이는 강한 이음 조각 쪼갬이 가장 크다는 것과 어긋난다. $\square$

## 세우기

오그린 그래프를 세우려면 다음이 필요하다:

1. [타잔 알고리즘](tarjan.md)이나 [코사라주 알고리즘](kosaraju.md)으로 강한 이음 조각을 모두 찾는다. $O(V + E)$이다.
2. 꼭짓점마다 강한 이음 조각 이름표를 붙인다.
3. $G$의 변 $(u, v)$마다 $\text{scc}[u] \neq \text{scc}[v]$이면 변 $(\text{scc}[u], \text{scc}[v])$을 $G^{SCC}$에 더한다(겹치는 것은 없앤다).

세우는 전체 시간은 $O(V + E)$이다.

```python
"""
오그린 그래프 세우기.

타잔 알고리즘으로 강하게 이어진 조각을 셈하고,
그다음 강한 이음 조각의 유향 비순환 그래프(오그린 그래프)를 세운다.
"""


# === 타잔의 강한 이음 조각(도우미) ===
def tarjan_scc(graph, n):
    """강한 이음 조각을 찾아 (조각 목록, 꼭짓점-조각 대응)을 돌려준다."""
    disc = [-1] * n
    low = [0] * n
    on_stack = [False] * n
    stack = []
    timer = [0]
    sccs = []
    scc_id = [0] * n

    def dfs(u):
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        stack.append(u)
        on_stack[u] = True

        for v in graph.get(u, []):
            if disc[v] == -1:
                dfs(v)
                low[u] = min(low[u], low[v])
            elif on_stack[v]:
                low[u] = min(low[u], disc[v])

        if low[u] == disc[u]:
            component = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc_id[w] = len(sccs)
                component.append(w)
                if w == u:
                    break
            sccs.append(component)

    for u in range(n):
        if disc[u] == -1:
            dfs(u)

    return sccs, scc_id


# === 오그린 그래프 세우기 ===
def build_condensation(graph, n):
    """
    방향 그래프의 오그린 그래프(강한 이음 조각의 유향 비순환 그래프)를 세운다.

    매개변수
    ----------
    graph : dict[int, list[int]]
        꼭짓점이 0부터 n-1인 방향 그래프의 이웃 목록.
    n : int
        꼭짓점의 개수.

    반환값
    -------
    tuple
        (sccs, condensation_adj, scc_id). 여기서 condensation_adj은
        조각 번호마다 이웃 조각 번호의 목록에 대응시킨다.
    """
    sccs, scc_id = tarjan_scc(graph, n)
    k = len(sccs)
    cond_edges = set()
    cond_adj = {i: [] for i in range(k)}

    for u in range(n):
        for v in graph.get(u, []):
            if scc_id[u] != scc_id[v]:
                edge = (scc_id[u], scc_id[v])
                if edge not in cond_edges:
                    cond_edges.add(edge)
                    cond_adj[scc_id[u]].append(scc_id[v])

    return sccs, cond_adj, scc_id


# === 메인 ===
if __name__ == "__main__":
    graph = {
        0: [1], 1: [2, 3], 2: [0], 3: [4],
        4: [5], 5: [3], 6: [5, 7], 7: [],
    }
    sccs, cond_adj, scc_id = build_condensation(graph, 8)

    print("SCCs:")
    for i, scc in enumerate(sccs):
        print(f"  SCC {i}: {sorted(scc)}")

    print("\nCondensation graph edges:")
    for u in cond_adj:
        for v in cond_adj[u]:
            print(f"  SCC {u} -> SCC {v}")
```

**출력:**
```
SCCs:
  SCC 0: [3, 4, 5]
  SCC 1: [0, 1, 2]
  SCC 2: [7]
  SCC 3: [6]

Condensation graph edges:
  SCC 1 -> SCC 0
  SCC 3 -> SCC 0
  SCC 3 -> SCC 2
```

## 오그린 그래프의 쓰임새

오그린 그래프가 유향 비순환 그래프이므로 그 알고리즘을 써서 본디 그래프의 문제를 풀 수 있다:

**닿음.** $G$에서 두 꼭짓점 $u$과 $v$이 서로 닿을 수 있는 것은 둘이 같은 강한 이음 조각에 들 때 그리고 오직 그때뿐이다. 한쪽 방향의 닿음은 오그린 유향 비순환 그래프에서의 닿음으로 줄어들며, 위상 정렬을 한 뒤 답할 수 있다.

**모두에 닿는 데 필요한 최소 꼭짓점 모음.** 근원 강한 이음 조각(오그린 그래프에서 들어오는 차수가 0인 것)은 그로부터 다른 모든 꼭짓점에 닿을 수 있는 조각이다. 다른 모두에 닿는 데 필요한 꼭짓점의 최소 개수는 근원 강한 이음 조각의 개수와 같다.

**가장 긴 경로.** (강한 이음 조각 안의 순환을 무시할 때) $G$의 가장 긴 경로는 큰 꼭짓점마다 그 조각의 꼭짓점 수를 무게로 두고 오그린 유향 비순환 그래프에서 가장 긴 경로를 찾아 셈할 수 있다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. 20장.

## 연습문제

**연습문제 1.**
오그린 그래프를 정의하고 그것이 유향 비순환 그래프임을 증명하여라.

??? success "연습문제 1 풀이"
    방향 그래프 $G$의 **오그림**은 강한 이음 조각마다 꼭짓점 하나로 오그린다. 오그린 그래프에서 조각 $C_1$에서 조각 $C_2$으로 가는 변이 있는 것은 $G$의 어떤 변이 $C_1$의 꼭짓점에서 $C_2$의 꼭짓점으로 갈 때 그리고 오직 그때뿐이다. 오그린 꼭짓점 사이에 순환이 있다면 그에 딸린 조각들이 모두 서로 닿아 더 큰 조각 하나를 이루게 되어 가장 큼에 어긋나므로, 오그린 그래프는 유향 비순환 그래프이다. $\square$

---

**연습문제 2.**
본디 그래프의 강한 이음 조각으로 볼 때 오그린 그래프의 꼭짓점과 변은 몇 개까지 될 수 있는가?

??? success "연습문제 2 풀이"
    본디 그래프에 강한 이음 조각이 $k$개 있으면 오그린 그래프의 꼭짓점은 정확히 $k$개다. 변은 많아야 $\min(k(k-1)/2, E)$개다(조각 사이가 이어진 짝마다 변 하나). 실전에서는 본디 변을 모두 살피고 같은 조각 짝을 잇는 것을 없애서 오그린 그래프의 변 수를 $O(V + E)$에 셈할 수 있다. $\square$

---

**연습문제 3.**
달림 살피기에서 오그린 그래프의 쓰임새를 설명하여라.

??? success "연습문제 3 풀이"
    소프트웨어 빌드 체계에서 단원들의 달림은 방향 그래프를 이룬다. 도는 달림(강한 이음 조각)은 함께 컴파일해야 하는 단원을 가리킨다. 오그린 그래프가 컴파일 차례를 보여 준다. 곧 오그린 유향 비순환 그래프를 위상 정렬하고 조각마다 한 덩어리로 컴파일한다. 이러면 따로 컴파일할 수 있는 묶음과 가장 좋은 빌드 차례를 알아낼 수 있다. 마찬가지로 데이터베이스 거래 살피기에서 강한 이음 조각은 교착 묶음을 나타낸다. $\square$

---

**연습문제 4.**
크기가 4, 3, 2인 강한 이음 조각 3개와 조각 사이의 변 5개를 갖는 그래프에서 오그린 그래프를 설명하여라.

??? success "연습문제 4 풀이"
    오그린 그래프의 꼭짓점은 3개다(조각마다 하나). 본디 그래프의 조각 사이 변 5개는 오그린 그래프에서 더 적은 수의 변에 맞대응될 수 있다(같은 조각 짝을 잇는 변 여럿은 하나로 뭉개진다). 오그린 그래프의 변은 많아야 3개다($\binom{3}{2} = 3$가지 방향 짝). 오그린 그래프는 꼭짓점 3개에 변이 많아야 3개인 유향 비순환 그래프이며 순환이 없어야 한다. 가능한 꼴 하나: 사슬 $A \to B \to C$에 변 $A \to C$을 더한 것(본디 조각 사이 변 5개에서 모두 3개). $\square$
