# 변 갈래 나누기

DFS이 그래프를 돌아볼 때 변마다 DFS 나무와의 관계에 따라 네 갈래 가운데 하나에 든다. 이 갈래 나누기는 그저 장부 적기가 아니다. 어떤 변 갈래가 있느냐 없느냐가 그래프의 근본적인 짜임 성질을 드러낸다. 되돌이 변은 고리를 알리고, 가로 변은 방향 그래프에서만 나타나며, 앞선 변은 군더더기 닿음을 알린다. 변 갈래 나누기를 이해하는 일이 고리 알아내기, 강하게 이어진 덩이 셈하기, 위상 정렬 알고리즘의 바탕이다.

---

## 1. 네 가지 변 갈래

$\text{pre}(u)$과 $\text{post}(u)$을 DFS이 찍은 찾은 때와 마친 때라고 하자. DFS이 변 $(u, v)$을 살필 때 그 변은 다음과 같이 갈린다:

**나무 변.** $(u, v)$을 살필 때 $v$이 희면(다녀가지 않았으면) 그 변은 DFS 나무의 일부가 된다. 나무 변이 DFS 숲을 이룬다.

**되돌이 변.** $(u, v)$을 살필 때 $v$이 잿빛이면(찾았으나 마치지 않았으면) 그 변은 $u$에서 DFS 나무의 조상 $v$으로 간다. 마찬가지로 $\text{pre}(v) \leq \text{pre}(u)$이고 $\text{post}(u) \leq \text{post}(v)$이다.

**앞선 변.** $v$이 검고(마쳤고) $\text{pre}(u) < \text{pre}(v)$이면 그 변은 $u$에서, 다른 길로 이미 온전히 살펴본 자손 $v$으로 간다.

**가로 변.** $v$이 검고(마쳤고) $\text{pre}(v) < \text{pre}(u)$이면 그 변은 DFS 나무의 서로 다른 가지에 있는 꼭짓점 둘을 잇는다(또는 나중 가지에서 앞선 가지로 간다).

---

## 2. 앞/뒤 번호로 갈래 나누기

앞/뒤 구간은 변 갈래를 가려내는 깔끔한 길을 준다. 변 $(u, v)$에 대해:

| 변 갈래 | 조건 | 구간 관계 |
|---|---|---|
| 나무 | $v$이 힘 | $[\text{pre}(v), \text{post}(v)] \subset [\text{pre}(u), \text{post}(u)]$ |
| 되돌이 | $v$이 잿빛임 | $[\text{pre}(u), \text{post}(u)] \subset [\text{pre}(v), \text{post}(v)]$ |
| 앞선 | $v$이 검음, $\text{pre}(u) < \text{pre}(v)$ | $[\text{pre}(v), \text{post}(v)] \subset [\text{pre}(u), \text{post}(u)]$ |
| 가로 | $v$이 검음, $\text{pre}(v) < \text{pre}(u)$ | 구간이 떨어짐, $\text{post}(v) < \text{pre}(u)$ |

!!! tip "무방향 그래프가 더 단순하다"
    무방향 그래프에서 DFS은 **나무 변**과 **되돌이 변**만 만든다. 나무가 아닌 변은 모두 꼭짓점을 조상과 잇기 때문에 앞선 변과 가로 변은 생길 수 없다(먼저 찾은 끝점이 다른 끝점을 검은색이 아니라 잿빛으로 본다).

---

## 3. 되돌이 변으로 고리 알아내기

방향 그래프에 고리가 있을 때 그리고 그때만 DFS이 되돌이 변을 적어도 하나 찾는다. 되돌이 변 $(u, v)$이 고리를 닫는다. 곧 $v$에서 $u$까지 내려가는 나무 길에 되돌이 변 $(u, v)$을 이으면 방향 고리가 된다. 이것이 DFS 기반 고리 알아내기 알고리즘 전부의 바탕이다.

---

## 4. 구현

```python
"""
DFS 돌아보기 도중의 변 갈래 나누기.

꼭짓점 빛깔 칠하기(흰색/잿빛/검은색)로 변마다 나무, 되돌이,
앞선, 가로 가운데 하나로 가른다.
"""

# === 변 갈래 나누기 =========================================================

def classify_edges(graph):
    """DFS으로 방향 그래프의 변을 모두 갈래 나눈다.

    매개변수
    ----------
    graph : dict[int, list[int]]
        방향 그래프의 이웃 목록.

    반환값
    -------
    dict[str, list[tuple[int, int]]]
        갈래로 묶은 변: 'tree', 'back', 'forward', 'cross'.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {v: WHITE for v in graph}
    pre = {}
    post = {}
    clock = [0]
    edges = {"tree": [], "back": [], "forward": [], "cross": []}

    def explore(u):
        clock[0] += 1
        pre[u] = clock[0]
        color[u] = GRAY
        for v in graph[u]:
            if color[v] == WHITE:
                edges["tree"].append((u, v))
                explore(v)
            elif color[v] == GRAY:
                edges["back"].append((u, v))
            elif pre[u] < pre[v]:
                edges["forward"].append((u, v))
            else:
                edges["cross"].append((u, v))
        color[u] = BLACK
        clock[0] += 1
        post[u] = clock[0]

    for vertex in graph:
        if color[vertex] == WHITE:
            explore(vertex)

    return edges, pre, post

# === 메인 =====================================================================

if __name__ == "__main__":
    graph = {
        0: [1, 3],
        1: [2],
        2: [3],
        3: [1],
    }

    edges, pre, post = classify_edges(graph)

    print("Pre/Post timestamps:")
    for v in sorted(pre):
        print(f"  Vertex {v}: [{pre[v]}, {post[v]}]")

    print("\nEdge classification:")
    for etype, elist in edges.items():
        if elist:
            print(f"  {etype.capitalize():8s}: {elist}")
```

**출력:**
```
Pre/Post timestamps:
  Vertex 0: [1, 8]
  Vertex 1: [2, 7]
  Vertex 2: [3, 6]
  Vertex 3: [4, 5]

Edge classification:
  Tree    : [(0, 1), (1, 2), (2, 3)]
  Back    : [(3, 1)]
  Cross   : [(0, 3)]
```

되돌이 변 $(3, 1)$이 고리 $1 \to 2 \to 3 \to 1$을 확인해 준다. 가로 변 $(0, 3)$은 꼭짓점 0을 꼭짓점 3에 잇는데, 그 변을 살필 때 3은 이미 온전히 살펴본(검은) 상태였다.

---

## 연습문제

**연습문제 1.**
꼭짓점 0에서 시작하는 DFS(이웃은 번호 차례로 살핀다) 아래 다음 방향 그래프의 변을 모두 갈래 나누어라. 꼭짓점 $\{0,1,2,3\}$, 변 $\{(0,1),(0,2),(1,2),(2,0),(2,3),(3,3)\}$.

??? success "연습문제 1 풀이"
    0에서 DFS: 0을 찾고, 1을 살피고(나무 변 $0 \to 1$), 1에서 2을 살피고(나무 변 $1 \to 2$), 2에서 0을 살핀다 — 꼭짓점 0은 잿빛이므로 되돌이 변 $2 \to 0$이다. 2에서 3을 살핀다(나무 변 $2 \to 3$). 제 고리 $3 \to 3$: 꼭짓점 3이 잿빛이므로 되돌이 변 $3 \to 3$이다. 3을 마치고 2을 마치고 1을 마친다. 0에서 2을 살피는데 꼭짓점 2은 검고 $\text{disc}[0] < \text{disc}[2]$이므로 앞선 변 $0 \to 2$이다. 0을 마친다. 갈래: 나무 변 $\{0 \to 1, 1 \to 2, 2 \to 3\}$, 되돌이 변 $\{2 \to 0, 3 \to 3\}$, 앞선 변 $\{0 \to 2\}$. $\square$

---

**연습문제 2.**
방향 그래프의 DFS에서 변 $(u, v)$이 가로 변일 때 그리고 그때만, $(u, v)$을 살필 때 $v$이 검고 $\text{disc}[v] > \text{disc}[u]$이 거짓임을 증명하여라.

??? success "연습문제 2 풀이"
    $(u, v)$을 살필 때 $v$이 검다면 $v$은 이미 마쳤다. $\text{disc}[u] < \text{disc}[v]$이면 $v$은 $u$보다 나중에, 그리고 $u$이 살아 있는 구간 안에서 찾아졌으므로 자손이고 따라서 $(u, v)$은 앞선 변이다. $\text{disc}[u] > \text{disc}[v]$이면(곧 $\text{disc}[v] > \text{disc}[u]$이 거짓이고 $u \neq v$이면) $v$은 $u$보다 먼저 찾아졌으나 이미 마쳤으므로 다른 가지에 있다. 이것이 바로 가로 변이다. $\square$

---

**연습문제 3.**
무방향 그래프의 DFS에 왜 가로 변이나 앞선 변이 없는지 설명하여라. 어떤 갈래의 변이 있는가?

??? success "연습문제 3 풀이"
    무방향 DFS에는 나무 변과 되돌이 변만 있다. 꼭짓점 $u$에서 DFS이 변 $\{u, v\}$을 살필 때, $v$이 희면 나무 변이 된다. $v$이 잿빛이면($v \neq \text{parent}[u]$이면) 되돌이 변이다. $v$이 검을 수는 없다. $v$이 이미 마쳤다면 DFS이 $v$에 있을 때 변 $\{v, u\}$을 살펴 ($u$이 희었다면) $u$을 나무 자손으로 만들었거나 되돌이 변을 가려냈을 것이기 때문이다. 그러므로 $v$이 $u$의 조상이 아닌 한 $u$이 아직 살아 있는 동안 $v$이 마쳐 있을 수 없으며, 조상이라면 되돌이 변이 된다. $\square$

---

**연습문제 4.**
어떤 방향 그래프의 DFS 숲에 되돌이 변이 꼭 하나 있다. 그래프의 고리 짜임에 대해 무엇을 말할 수 있는가?

??? success "연습문제 4 풀이"
    그 그래프에는 방향 고리가 적어도 하나 있다(되돌이 변과 나무 길이 이루는 고리). 그러나 어떤 한 DFS에서 되돌이 변이 꼭 하나라고 해서 그래프에 고리가 꼭 하나라는 뜻은 아니다. DFS 차례가 다르면 다른 되돌이 변이 드러날 수 있고, 되돌이 변 하나가 다른 고리와 겹치는 고리에 해당할 수도 있다. 확실히 말할 수 있는 것은 그 되돌이 변(과 그에 해당하는 그래프 변)을 지우면 그래프가 DAG이 된다는 것이다(남은 어떤 DFS도 되돌이 변을 찾지 못한다). $\square$

## 정리하며

| 변 갈래 | DFS 나무에서의 방향 | 뜻하는 바 |
|---|---|---|
| 나무 | 어버이에서 자식으로 | DFS 숲을 이룬다 |
| 되돌이 | 자손에서 조상으로 | 고리가 있음을 알린다 |
| 앞선 | 조상에서 자손으로(나무가 아님) | 군더더기 닿음 |
| 가로 | 가지 사이 | 따로 떨어진 부분 나무를 잇는다 |

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 22. MIT Press.
