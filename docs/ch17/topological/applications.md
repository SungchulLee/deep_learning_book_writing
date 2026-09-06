# 위상 정렬의 쓰임새

위상 정렬은 그저 그래프 알고리즘 연습거리가 아니다. 달림을 지키는 차례로 무언가를 다뤄야 할 때면 언제나 나타난다. 빌드 체계는 소스 파일을 달림 차례대로 컴파일하고, 일정 짜개는 앞선 조건을 지키며 일꾼에게 일을 맡기며, 컴파일러는 기호가 쓰이기 앞서 그 뜻매김을 푼다. 이 쪽에서는 가장 중요한 쓰임새를 살펴보는데, 모두 알맞은 [유향 비순환 그래프](dag.md) 위에서 위상 차례를 셈하는 일로 줄어든다.

## 일 일정 짜기

위상 정렬의 가장 곧바른 쓰임새는 앞뒤 제약이 있는 일 일정 짜기이다. 달림이 있는 일 모음이 주어지면 위상 차례가 올바른 실행 순서를 준다.

**문제.** 일 $n$개와 "일 $i$이 일 $j$의 시작 앞에 끝나야 한다"를 뜻하는 앞뒤 제약 $(i, j)$의 모음이 주어질 때 올바른 실행 차례를 찾아라.

이는 바로 위상 정렬 문제이다. 곧 일을 꼭짓점으로, 제약을 방향 변으로 삼고 [칸 알고리즘](kahn.md)이나 [깊이 우선 돌아보기에 바탕한 정렬](dfs.md)을 쓴다.

```python
"""
위상 정렬을 쓴 일 일정 짜기.

위상 차례가 올바른 실행 차례를 어떻게 정하는지
앞뒤 제약이 있는 일에 대해 보여 준다.
"""

from collections import deque


# === 일 일정 짜개 ===
def schedule_tasks(tasks, dependencies):
    """
    모든 달림을 지키는 올바른 일 실행 차례를 셈한다.

    매개변수
    ----------
    tasks : list[str]
        일 이름의 목록.
    dependencies : list[tuple[str, str]]
        (a, b)마다 일 a가 일 b보다 먼저 끝나야 함을 뜻한다.

    반환값
    -------
    list[str]
        올바른 실행 차례의 일들. 도는 달림이 있으면 빈 목록.
    """
    idx = {t: i for i, t in enumerate(tasks)}
    n = len(tasks)
    graph = {i: [] for i in range(n)}
    in_degree = [0] * n

    for a, b in dependencies:
        graph[idx[a]].append(idx[b])
        in_degree[idx[b]] += 1

    queue = deque(i for i in range(n) if in_degree[i] == 0)
    order = []

    while queue:
        u = queue.popleft()
        order.append(tasks[u])
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    return order if len(order) == n else []


# === 메인 ===
if __name__ == "__main__":
    tasks = ["design", "implement", "test", "document", "deploy"]
    deps = [
        ("design", "implement"),
        ("implement", "test"),
        ("implement", "document"),
        ("test", "deploy"),
        ("document", "deploy"),
    ]
    result = schedule_tasks(tasks, deps)
    print(f"Execution order: {result}")
```

**출력:**
```
Execution order: ['design', 'implement', 'test', 'document', 'deploy']
```

## 유향 비순환 그래프의 최단 경로와 최장 경로

In a general weighted graph, shortest path algorithms like Dijkstra or Bellman-Ford have complexities of $O((V + E) \log V)$ or $O(VE)$. In a DAG, topological ordering enables a single-pass solution in $O(V + E)$ time, even with negative edge weights.

**알고리즘.** 꼭짓점을 위상 차례로 다룬다. 꼭짓점 $u$마다 나가는 변을 모두 늦춘다:

$$
d[v] = \min(d[v],\ d[u] + w(u, v))
$$

위상 차례에서 $u$이 $v$ 앞에 오므로 변 $(u, v)$을 다룰 때 $d[u]$은 이미 확정되어 있다.

For the **longest path**, simply negate all weights or replace $\min$ with $\max$. The longest path in a general graph is NP-hard, but in a DAG it is solvable in linear time.

```python
"""
위상 정렬로 얻는 유향 비순환 그래프의 최단 경로와 최장 경로.

꼭짓점을 위상 차례로 다뤄 한 번 훑기로 O(V + E)에
전체 O(F * E log V)에 푼다. 변 무게가 음수여도 된다.
"""

from collections import deque


# === 유향 비순환 그래프 최단 경로 ===
def dag_shortest_path(graph, n, source):
    """
    무게 있는 유향 비순환 그래프에서 근원으로부터의 최단 거리를 셈한다.

    매개변수
    ----------
    graph : dict[int, list[tuple[int, float]]]
        (이웃, 무게) 짝을 담은 이웃 목록.
    n : int
        꼭짓점의 개수.
    source : int
        근원 꼭짓점.

    반환값
    -------
    list[float]
        근원에서 꼭짓점마다까지의 최단 거리.
    """
    # 칸 알고리즘으로 위상 정렬
    in_degree = [0] * n
    adj = {i: [] for i in range(n)}
    for u in range(n):
        for v, w in graph.get(u, []):
            adj.setdefault(u, [])
            in_degree[v] += 1

    queue = deque(i for i in range(n) if in_degree[i] == 0)
    topo = []
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v, w in graph.get(u, []):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    # 위상 차례로 변 늦추기
    dist = [float("inf")] * n
    dist[source] = 0
    for u in topo:
        if dist[u] == float("inf"):
            continue
        for v, w in graph.get(u, []):
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w

    return dist


# === 메인 ===
if __name__ == "__main__":
    # 무게 있는 유향 비순환 그래프: 0->1(2), 0->2(4), 1->2(1), 1->3(7), 2->3(3)
    g = {
        0: [(1, 2), (2, 4)],
        1: [(2, 1), (3, 7)],
        2: [(3, 3)],
        3: [],
    }
    dist = dag_shortest_path(g, 4, 0)
    print(f"Shortest distances from 0: {dist}")
```

**출력:**
```
Shortest distances from 0: [0, 2, 3, 6]
```

## 빌드 체계

Make, Gradle, Bazel 같은 빌드 체계는 소스 파일의 달림을 유향 비순환 그래프로 나타낸다. 파일을 위상 차례로 컴파일하면 달린 것이 그것에 달린 파일보다 먼저 빌드됨이 보장된다.

**짜임:**

- 파일이나 단원마다 꼭짓점이다.
- 방향 변 $(A, B)$은 "파일 $A$을 파일 $B$보다 먼저 컴파일해야 한다"를 뜻한다.
- 위상 정렬이 빌드 차례를 정한다.
- 위상 정렬이 어그러지면(내놓은 꼭짓점이 모두의 수보다 적으면) 도는 달림이 있음을 알아챈다.

## 강의 선행 조건

대학은 흔히 강의 선행 조건 짜임을 유향 비순환 그래프로 나타낸다. 위상 차례가 학기별 계획을 올바로 준다. [칸 알고리즘](kahn.md)의 "나란한 켜" 변종은 어떤 강의를 한꺼번에 들을 수 있는지 자연스레 가려낸다. 곧 칸 알고리즘의 같은 판에 있는 근원끼리는 서로 선행 조건이 없다.

## 결정적 경로 방법

기획 살림에서 **결정적 경로**는 일 달림 유향 비순환 그래프를 가로지르는 가장 긴 경로이다. 일마다 걸리는 때가 있고, 결정적 경로가 기획을 마치는 최소 시간을 정한다.

**알고리즘:**

1. 일을 위상 정렬한다.
2. 앞먹임(근원에서의 최장 경로)으로 일마다 가장 이른 시작 때를 셈한다.
3. 뒤먹임으로 가장 늦은 시작 때를 셈한다.
4. 가장 이른 시작과 가장 늦은 시작이 같은 일이 결정적 경로 위에 있다.

전체 시간은 최장 경로 길이와 같으며, 위에서 말한 대로 $O(V + E)$에 셈할 수 있다.

## 자료 다루기 물길

요즘 자료 물길(Apache Airflow, Dagster, Prefect)은 일 흐름의 단계를 유향 비순환 그래프로 나타낸다. 위상 정렬이 실행 차례를 정하고, 칸 알고리즘에서 나오는 켜 짜임이 어떤 단계를 나란히 돌릴 수 있는지 가려낸다.

!!! tip "깊은 배움과의 이음"
    신경망 셈 그래프는 마디가 연산을, 변이 자료 흐름을 나타내는 유향 비순환 그래프이다. PyTorch나 TensorFlow 같은 틀은 위상 차례를 써서 앞먹임과 뒤먹임의 일정을 짠다. 앞먹임의 뒤집은 위상 차례가 뒤로 퍼뜨리기의 올바른 차례를 준다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. 20장.
- Kahn, A. B. (1962). Topological sorting of large networks. *Communications of the ACM*, 5(11), 558-562.

## 연습문제

**연습문제 1.**
소프트웨어 공학에서 위상 정렬의 쓰임새 세 가지를 설명하여라.

??? success "연습문제 1 풀이"
    (1) **Build systems**: compile source files in dependency order (e.g., Makefiles). A file must be compiled before files that depend on it. (2) **Package managers**: install packages in dependency order (e.g., apt, pip). Each package is installed after all its dependencies. (3) **Task scheduling**: execute tasks in a project where some tasks depend on others. Topological sort gives a valid execution order. All three require a DAG; cycles indicate unresolvable dependencies. $\square$

---

**연습문제 2.**
강의 짜임의 선행 조건이 유향 비순환 그래프를 이룬다. 위상 정렬은 올바른 수강 차례를 어떻게 정하는가?

??? success "연습문제 2 풀이"
    Model courses as vertices and prerequisites as directed edges ($A \to B$ means "A is a prerequisite for B"). Run topological sort to get a linear ordering. Any course appears after all its prerequisites in this ordering. Students can take courses in this order and always satisfy prerequisites. If multiple valid orderings exist, they represent different valid course plans. The sort also reveals the minimum number of semesters needed (the longest path in the DAG). $\square$

---

**연습문제 3.**
유향 비순환 그래프의 최단/최장 경로를 셈하는 데 위상 정렬을 어떻게 쓸 수 있는가?

??? success "연습문제 3 풀이"
    Process vertices in topological order. For each vertex $v$, relax all outgoing edges: $d[u] = \min(d[u], d[v] + w(v,u))$ for shortest paths, or $d[u] = \max(d[u], d[v] + w(v,u))$ for longest paths. Since vertices are processed in dependency order, all incoming edges to $v$ have already been relaxed before processing $v$. This gives correct shortest/longest path distances in $O(V + E)$ — faster than Dijkstra or Bellman-Ford for DAGs. $\square$

---

**연습문제 4.**
위상 정렬이 표 계산기의 칸 값매김에 어떻게 쓰이는지 설명하여라.

??? success "연습문제 4 풀이"
    In a spreadsheet, cells may reference other cells in formulas, creating a dependency DAG. Cell A1 depending on B2 means B2 must be evaluated before A1. Topological sort of the dependency graph gives a valid evaluation order. If a cycle exists (e.g., A1 depends on B2 which depends on A1), the spreadsheet reports a circular reference error. This is how Excel and Google Sheets determine cell evaluation order. $\square$
