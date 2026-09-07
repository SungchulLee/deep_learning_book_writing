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

여느 짐 붙은 그래프에서는 데이크스트라나 벨먼-포드 같은 최단 경로 알고리즘의 복잡도가 $O((V + E) \log V)$이나 $O(VE)$이다. 유향 비순환 그래프에서는 위상 차례 덕에 변의 짐이 음수라도 한 번 훑어 $O(V + E)$ 시간에 풀 수 있다.

**알고리즘.** 꼭짓점을 위상 차례로 다룬다. 꼭짓점 $u$마다 나가는 변을 모두 늦춘다:

$$
d[v] = \min(d[v],\ d[u] + w(u, v))
$$

위상 차례에서 $u$이 $v$ 앞에 오므로 변 $(u, v)$을 다룰 때 $d[u]$은 이미 확정되어 있다.

**가장 긴 경로**는 짐의 부호를 모두 뒤집거나 $\min$을 $\max$으로 갈음하면 된다. 여느 그래프에서 가장 긴 경로는 NP-어려움이지만 유향 비순환 그래프에서는 선형 시간에 풀린다.

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
    (1) **빌드 얼개**: 소스 파일을 매인 차례대로 엮는다(메이크파일 따위). 어떤 파일에 매인 파일보다 그 파일을 먼저 엮어야 한다. (2) **꾸러미 관리자**: 꾸러미를 매인 차례대로 깐다(apt, pip 따위). 꾸러미마다 그것이 매인 것을 모두 깐 뒤에 깔린다. (3) **일 일정**: 어떤 일이 다른 일에 매인 프로젝트에서 일을 해 나간다. 위상 정렬이 옳은 실행 차례를 준다. 셋 다 유향 비순환 그래프여야 하며 순환은 풀 수 없는 매임을 뜻한다. $\square$

---

**연습문제 2.**
강의 짜임의 선행 조건이 유향 비순환 그래프를 이룬다. 위상 정렬은 올바른 수강 차례를 어떻게 정하는가?

??? success "연습문제 2 풀이"
    강의를 꼭짓점으로, 선행 조건을 방향 있는 변으로 그린다($A \to B$은 "A이 B의 선행 조건"이라는 뜻이다). 위상 정렬을 돌려 한 줄 차례를 얻는다. 이 차례에서 강의는 모두 그 선행 조건 뒤에 온다. 학생은 이 차례대로 들으면 늘 선행 조건을 채운다. 옳은 차례가 여럿이면 저마다 다른 옳은 수강 계획을 나타낸다. 이 정렬은 가장 적게 드는 학기 수(유향 비순환 그래프의 가장 긴 경로)도 알려 준다. $\square$

---

**연습문제 3.**
유향 비순환 그래프의 최단/최장 경로를 셈하는 데 위상 정렬을 어떻게 쓸 수 있는가?

??? success "연습문제 3 풀이"
    꼭짓점을 위상 차례대로 다룬다. 꼭짓점 $v$마다 나가는 변을 모두 늦춘다. 최단 경로면 $d[u] = \min(d[u], d[v] + w(v,u))$, 가장 긴 경로면 $d[u] = \max(d[u], d[v] + w(v,u))$이다. 꼭짓점을 매인 차례대로 다루므로 $v$을 다루기 앞에 $v$으로 들어오는 변은 모두 이미 늦춰졌다. 그래서 $O(V + E)$에 옳은 최단/최장 경로 거리를 얻는다. 유향 비순환 그래프에서는 데이크스트라나 벨먼-포드보다 빠르다. $\square$

---

**연습문제 4.**
위상 정렬이 표 계산기의 칸 값매김에 어떻게 쓰이는지 설명하여라.

??? success "연습문제 4 풀이"
    표 계산기에서는 칸이 수식으로 다른 칸을 가리킬 수 있어 매임의 유향 비순환 그래프가 생긴다. 칸 A1이 B2에 매였다면 A1보다 B2을 먼저 셈해야 한다. 매임 그래프를 위상 정렬하면 옳은 셈 차례가 나온다. 순환이 있으면(A1이 B2에 매이고 B2이 다시 A1에 매이는 따위) 표 계산기가 순환 참조 오류를 알린다. 엑셀과 구글 스프레드시트가 칸의 셈 차례를 이렇게 정한다. $\square$
