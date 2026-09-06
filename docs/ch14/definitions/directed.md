# 방향 그래프와 무방향 그래프

변에 방향이 있느냐는 문제를 그래프로 본뜰 때 가장 근본이 되는 짜임의 고름이다. 무방향 그래프는 벗함이나 물리적 이웃함 같은 대칭 관계를 담고, 방향 그래프는 웹 링크, 앞서 들어야 할 과목의 사슬, 일방통행 길 같은 비대칭 관계를 본뜬다. 이 갈림은 길과 이어짐을 어떻게 정의하는가부터 어떤 자료 짜임을 고르는가까지 그래프 알고리즘의 모든 결에 영향을 준다.

## 무방향 그래프

**무방향 그래프** $G = (V, E)$은 끝이 있는 꼭짓점 묶음 $V$과 변 묶음 $E$으로 이루어지며, 변마다 $u, v \in V$이고 (단순 그래프라면) $u \neq v$인 차례 없는 짝 $\{u, v\}$이다. 변 $\{u, v\}$은 $u$과 $v$을 대칭으로 잇는다. 곧 $u$이 $v$에 이웃하면 $v$도 $u$에 이웃한다.

$$
E \subseteq \binom{V}{2} = \{\{u, v\} : u, v \in V, \; u \neq v\}
$$

꼭짓점 $n$개의 단순 무방향 그래프는 변을 많아야 $\binom{n}{2} = n(n-1)/2$개 갖는다.

!!! example "무방향 그래프"
    "벗함"이 서로 오가는 사회 망. 곧 앨리스가 밥의 벗이면 밥도 앨리스의 벗이다. 변 $\{Alice, Bob\}$에는 방향이 없다.

## 방향 그래프

**방향 그래프** $G = (V, E)$의 변은 차례 있는 짝 $(u, v)$이며 $u$에서 $v$으로 가는 방향 이음을 나타낸다. 꼭짓점 $u$은 변의 **꼬리**(또는 샘)이고 $v$은 **머리**(또는 과녁)이다.

$$
E \subseteq V \times V = \{(u, v) : u, v \in V\}
$$

단순 방향 그래프(제 고리 없음)에서는 $u \neq v$을 요구한다. 차례 있는 짝마다 따로 나타날 수 있으므로 변의 최대 개수는 $n(n-1)$이다.

변 $(u, v)$이 있다고 $(v, u)$이 있는 것은 아니다. $(u, v)$과 $(v, u)$이 모두 있으면 그 둘은 서로 다른 변이며 **맞선 짝**을 이룬다.

!!! example "방향 그래프"
    월드 와이드 웹은 방향 그래프이다. 곧 쪽 $u$이 쪽 $v$에 이어져도 $v$이 되돌아 잇지 않을 수 있다. 과목의 선수 조건도 방향 그래프를 이룬다. "미적분학 1은 미적분학 2보다 앞서야 한다"는 방향 있는 관계이다.

## 짜임의 견줌

| 성질 | 무방향 | 방향 |
|---|---|---|
| 변 표기 | $\{u, v\}$ | $(u, v)$ |
| 대칭 | $\{u,v\} = \{v,u\}$ | 일반으로 $(u,v) \neq (v,u)$ |
| 최대 변(단순) | $\binom{n}{2}$ | $n(n-1)$ |
| 차수 | $\deg(v)$ | $\deg^+(v)$, $\deg^-(v)$ |
| 이어짐 | 이어짐 / 끊어짐 | 강하게 / 약하게 이어짐 |
| 고리 알아내기 | 합치기-찾기 또는 DFS | 빛깔 상태를 쓴 DFS |

## 중요한 방향 그래프 갈래

### 방향 비순환 그래프(DAG)

방향 고리가 없는 방향 그래프를 **DAG**이라 한다. DAG은 기댐 짜임을 본뜨며, DAG마다 꼭짓점의 [위상 차례](../../ch17/topological/dag.md)를 가진다. 곧 변 $(u, v)$마다 $u$이 $v$보다 앞에 나오는 줄 세우기이다.

### 토너먼트

**토너먼트**는 완전 그래프 $K_n$의 변마다 방향을 주어 얻는 방향 그래프이다. 서로 다른 꼭짓점 짝 $u, v$마다 $(u, v)$이나 $(v, u)$ 가운데 꼭 하나가 있다. 토너먼트는 돌아가며 겨루는 시합을 본뜬다.

### 바탕 무방향 그래프

방향 그래프 $G$마다 방향 변 $(u, v)$을 무방향 변 $\{u, v\}$으로 바꾸고 겹치는 것을 지워 얻는 **바탕 무방향 그래프** $G'$이 있다. 바탕 무방향 그래프가 이어져 있으면 그 방향 그래프는 **약하게 이어짐**이다.

## 표현끼리 바꾸기

```python
"""
방향 그래프와 무방향 그래프 표현 사이의 바꿈.

방향 이웃 목록과 무방향 이웃 목록을 쌓고, 방향 그래프를
바탕 무방향 그래프로 바꾸고, 맞선 변을 살피는 법을
보인다.
"""


# === 이웃 목록 쌓기 ===

def build_undirected(n, edges):
    """변 짝에서 무방향 이웃 목록을 쌓는다."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def build_directed(n, edges):
    """변 짝에서 방향 이웃 목록을 쌓는다."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
    return adj


# === 바탕 무방향 그래프 ===

def to_undirected(adj, n):
    """방향 그래프를 바탕 무방향 그래프로 바꾼다."""
    edge_set = set()
    for u in range(n):
        for v in adj[u]:
            edge_set.add((min(u, v), max(u, v)))
    undirected = [[] for _ in range(n)]
    for u, v in edge_set:
        undirected[u].append(v)
        undirected[v].append(u)
    return undirected


# === 맞선 변 알아내기 ===

def find_antiparallel(adj, n):
    """방향 그래프에서 맞선 변 짝을 모두 찾는다."""
    edge_set = set()
    for u in range(n):
        for v in adj[u]:
            edge_set.add((u, v))
    pairs = []
    for u, v in edge_set:
        if (v, u) in edge_set and u < v:
            pairs.append((u, v))
    return pairs


# === 메인 ===

if __name__ == "__main__":
    # 방향 그래프
    directed_edges = [(0, 1), (1, 2), (2, 0), (1, 3)]
    adj_dir = build_directed(4, directed_edges)
    print("Directed adjacency list:")
    for v in range(4):
        print(f"  {v} -> {adj_dir[v]}")

    # 무방향으로 바꾸기
    adj_undir = to_undirected(adj_dir, 4)
    print("\nUnderlying undirected graph:")
    for v in range(4):
        print(f"  {v} -- {adj_undir[v]}")

    # 맞선 변 살피기
    # 맞선 짝을 만들려고 거꾸로 변 (1,0)을 더한다
    directed_edges2 = [(0, 1), (1, 0), (1, 2), (2, 0)]
    adj_dir2 = build_directed(3, directed_edges2)
    pairs = find_antiparallel(adj_dir2, 3)
    print(f"\nAntiparallel pairs: {pairs}")
```

**출력:**
```
Directed adjacency list:
  0 -> [1]
  1 -> [2, 3]
  2 -> [0]
  3 -> []
Underlying undirected graph:
  0 -- [1, 2]
  1 -- [0, 2, 3]
  2 -- [0, 1]
  3 -- [1]
Antiparallel pairs: [(0, 1)]
```

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. 22장.
- Diestel, R. (2017). *Graph Theory* (5th ed.). Springer. 1-2장.

## 연습문제

**연습문제 1.**
꼭짓점 $n$개의 방향 그래프가 있을 수 있는 가장 많은 변을 갖는다(제 고리 없음). 변이 몇 개인가? 무방향 단순 그래프의 최댓값과 견주면 어떠한가?

??? success "연습문제 1 풀이"
    단순 방향 그래프는 짝마다 $(u, v)$과 $(v, u)$을 모두 허락하므로 변의 최대 개수는 $n(n-1)$이다. 무방향 단순 그래프의 최댓값은 $\binom{n}{2} = n(n-1)/2$이다. 무방향 변마다 방향 변 둘에 해당하므로 방향의 최댓값은 무방향의 꼭 두 배이다. $\square$

---

**연습문제 2.**
방향 그래프가 주어졌을 때 그것이 약하게 이어짐인지 $O(V + E)$ 시간에 가려내는 법을 밝혀라.

??? success "연습문제 2 풀이"
    방향 변 $(u, v)$마다 무방향 변 $\{u, v\}$으로 바꾸어(겹치는 것은 무시하고) 바탕 무방향 그래프를 짓는다. 그다음 아무 꼭짓점에서 BFS이나 DFS을 돌린다. 모든 꼭짓점을 다녀갔을 때 그리고 그때만 그 그래프가 약하게 이어짐이다. 짓기와 돌아보기 모두 $O(V + E)$ 시간이 든다. $\square$

---

**연습문제 3.**
꼭짓점 $n$개의 토너먼트마다 해밀턴 길(모든 꼭짓점을 꼭 한 번씩 들르는 방향 길)이 있음을 증명하여라.

??? success "연습문제 3 풀이"
    $n$에 대한 귀납을 쓴다. 바탕 경우: $n = 1$은 시시하다. 귀납 걸음: 꼭짓점 $n - 1$개의 토너먼트마다 해밀턴 길이 있다고 놓자. 꼭짓점 $n$개의 토너먼트 $T$을 생각하자. 꼭짓점 $v_n$을 지워 꼭짓점 $n - 1$개의 토너먼트 $T'$을 얻으면 여기에 해밀턴 길 $v_1 \to v_2 \to \cdots \to v_{n-1}$이 있다. $(v_n, v_1)$이 변이면 $v_n$을 앞에 붙인다. 아니면 $(v_n, v_{i+1})$이 변이면서 $(v_i, v_n)$도 변인 첫 $i$을 찾는다($(v_n, v_1)$이 변이 아니면 $(v_{n-1}, v_n)$이 변이어야 하므로 그런 $i$이 있다). $v_i$과 $v_{i+1}$ 사이에 $v_n$을 끼워 넣는다. 그러면 꼭짓점 $n$개 전체의 해밀턴 길이 나온다. $\square$

---

**연습문제 4.**
방향 그래프에서 맞선 변 짝의 개수를 세는 함수를 적어라. 꼭짓점 $n$개의 그래프에서 그런 짝은 많아야 몇 개인가?

??? success "연습문제 4 풀이"
    모든 변을 집합에 담는다. 변 $(u, v)$마다 $(v, u)$도 집합에 있는지 살피고, 두 번 세지 않도록 $u < v$인 짝만 센다:

    ```python
    def count_antiparallel(adj, n):
        edges = set()
        for u in range(n):
            for v in adj[u]:
                edges.add((u, v))
        count = 0
        for u, v in edges:
            if u < v and (v, u) in edges:
                count += 1
        return count
    ```

    맞선 짝의 최대 개수는 $\binom{n}{2} = n(n-1)/2$이며 짝마다 $(u, v)$과 $(v, u)$이 모두 있을 때 이에 이른다. $\square$

---

**연습 5.**
방향 그래프가 강하게 이어져 있어도 오일러 회로가 없을 수 있는 까닭을, 차수가 모두 짝수인 이어진 무방향 그래프에는 늘 있다는 점과 견주어 설명하여라.

??? success "연습 5의 풀이"
    방향 오일러 회로는 (그래프가 강하게 이어져 있는 것만이 아니라) 꼭짓점마다 들어오는 차수와 나가는 차수가 같기를 요구한다. 강하게 이어진 방향 그래프에도 들어오는 차수와 나가는 차수가 다른 꼭짓점이 있을 수 있다. 이를테면 꼭짓점 $\{0, 1, 2\}$과 변 $(0,1), (1,0), (0,2), (2,0), (1,2)$의 그래프는 강하게 이어져 있지만, 꼭짓점 1은 나가는 차수 2과 들어오는 차수 2인 반면 꼭짓점 2은 나가는 차수 1과 들어오는 차수 2이라 오일러 회로가 없다. 무방향 그래프에서는 이어짐과 차수가 모두 짝수라는 것이 오일러 회로의 필요충분조건이다. $\square$
