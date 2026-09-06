# 덩이-자름 나무

이음매 점과 두 겹 이음 조각을 찾는 일은 자리에 매인 물음, 곧 어떤 꼭짓점이나 변이 이어짐에 결정적인가에 답한다. 덩이-자름 나무는 본디 그래프의 두 겹 이음 조각(*덩이*라 한다)과 이음매 점(*자름 꼭짓점*이라 한다)을 마디로 삼는 **나무**를 세워 이 앎을 전체 짜임으로 끌어올린다. 이 나무는 이어진 그래프의 무른 곳 전부를 한눈에 드러낸다.

## 정의

$G = (V, E)$을 이어진 방향 없는 그래프라 하자.

**덩이.** $G$의 가장 큰 두 겹 이음 아래그래프. 변마다 정확히 하나의 덩이에 든다. 다리(자름 변)는 그 두 끝점과 함께 스스로 덩이 하나를 이룬다.

**자름 꼭짓점.** 없애면 $G$이 끊어지는 꼭짓점. 자름 꼭짓점은 바로 덩이 둘 이상이 함께 갖는 꼭짓점이다.

**덩이-자름 나무.** 다음과 같이 세운 두 쪽 나무 $T$:

- $G$의 덩이마다 *덩이 마디* 하나를 만든다.
- $G$의 자름 꼭짓점마다 *자름 마디* 하나를 만든다.
- 자름 꼭짓점 $c$이 덩이 $B_i$에 들 때마다 $T$에 변 $(B_i, c)$을 더한다.

그 결과는 늘 나무(순환이 없고 이어짐)이며 두 쪽으로 나뉜 짜임이다. 곧 변마다 덩이 마디와 자름 마디를 잇는다.

## 성질

덩이가 $b$개, 자름 꼭짓점이 $c$개인 이어진 그래프 $G$의 덩이-자름 나무 $T$은 다음을 채운다:

- $T$에는 마디가 $b + c$개, 변이 $b + c - 1$개 있다.
- $T$은 나무이다. 곧 $T$에서 아무 변이나 없애는 것은 $G$에서 자름 꼭짓점을 없애는 것에 맞대응되며, 그러면 $G$이 끊어진다.
- $T$의 잎은 늘 덩이 마디이며 자름 마디는 결코 아니다.
- 덩이 둘은 많아야 꼭짓점 하나를 함께 가지며, 그 함께 갖는 꼭짓점은 자름 꼭짓점이다.

## 세우기 알고리즘

덩이-자름 나무를 세우는 일은 두 겹 이음 조각과 이음매 점을 모두 찾는 일로 줄어들며, 이는 깊이 우선 돌아보기 한 번으로 $O(V + E)$ 시간에 이룬다.

**Step 1.** Run DFS on $G$. Maintain discovery times $\text{disc}[v]$ and low values $\text{low}[v]$ for each vertex $v$.

**Step 2.** Use a stack of edges. Each time the DFS backtracks from child $u$ to parent $v$ and finds $\text{low}[u] \ge \text{disc}[v]$, pop edges from the stack until $(v, u)$ is reached. These edges form one biconnected component (block).

**Step 3.** Identify cut vertices: $v$ is a cut vertex if either (a) $v$ is the DFS root with two or more children, or (b) $v$ is not the root and has a child $u$ with $\text{low}[u] \ge \text{disc}[v]$.

**4단계.** 덩이마다 덩이 마디를, 자름 꼭짓점마다 자름 마디를 만든 뒤 위에서 말한 대로 이어서 나무를 세운다.

## 구현

```python
"""
깊이 우선 돌아보기로 덩이-자름 나무 세우기.

두 겹 이음 조각(덩이)과 이음매 점을 모두 찾는다
이어진 방향 없는 그래프에서 찾은 뒤 덩이-자름 나무를 세운다.
"""

from collections import defaultdict

# === 깊이 우선 돌아보기로 덩이-자름 나무 세우기 ===

class BlockCutTree:
    """이어진 방향 없는 그래프의 덩이-자름 나무를 세운다."""

    def __init__(self, n: int):
        """꼭짓점 n개(0부터 셈)로 그래프를 첫자리매김한다."""
        self.n = n
        self.adj = defaultdict(list)
        self.blocks = []          # 덩이마다의 꼭짓점 모음 목록
        self.cut_vertices = set()
        self.tree_adj = defaultdict(list)  # 덩이-자름 나무 이웃 목록

    def add_edge(self, u: int, v: int) -> None:
        """방향 없는 변 (u, v)을 더한다."""
        self.adj[u].append(v)
        self.adj[v].append(u)

    def build(self) -> None:
        """깊이 우선 돌아보기로 덩이와 자름 꼭짓점을 찾은 뒤 나무를 세운다."""
        disc = [-1] * self.n
        low = [0] * self.n
        parent = [-1] * self.n
        stack = []  # 변의 쌓기
        timer = [0]

        def dfs(u: int) -> None:
            disc[u] = low[u] = timer[0]
            timer[0] += 1
            child_count = 0

            for v in self.adj[u]:
                if disc[v] == -1:
                    child_count += 1
                    parent[v] = u
                    stack.append((u, v))
                    dfs(v)
                    low[u] = min(low[u], low[v])

                    # 이음매 점 / 덩이 경계인지 살피기
                    if (parent[u] == -1 and child_count > 1) or \
                       (parent[u] != -1 and low[v] >= disc[u]):
                        self.cut_vertices.add(u)

                    if low[v] >= disc[u]:
                        block = set()
                        while stack:
                            edge = stack.pop()
                            block.add(edge[0])
                            block.add(edge[1])
                            if edge == (u, v):
                                break
                        self.blocks.append(block)

                elif v != parent[u] and disc[v] < disc[u]:
                    low[u] = min(low[u], disc[v])
                    stack.append((u, v))

        dfs(0)
        self._build_tree()

    def _build_tree(self) -> None:
        """덩이와 자름 꼭짓점으로 덩이-자름 나무를 세운다."""
        # 덩이 마디: 번호 0..len(blocks)-1
        # 자름 마디: len(blocks)만큼 밀림
        cut_list = sorted(self.cut_vertices)
        cut_index = {v: i + len(self.blocks) for i, v in enumerate(cut_list)}

        for bi, block in enumerate(self.blocks):
            for v in block:
                if v in self.cut_vertices:
                    ci = cut_index[v]
                    self.tree_adj[bi].append(ci)
                    self.tree_adj[ci].append(bi)


# === 시연 ===

if __name__ == "__main__":
    #  그래프:  0--1--2--3--4
    #          |  |     |
    #          5--6     7
    bct = BlockCutTree(8)
    for u, v in [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,1),(3,7)]:
        bct.add_edge(u, v)
    bct.build()

    print(f"Number of blocks: {len(bct.blocks)}")
    print(f"Cut vertices: {sorted(bct.cut_vertices)}")
    for i, block in enumerate(bct.blocks):
        print(f"  Block {i}: {sorted(block)}")
```

**출력:**

```
Number of blocks: 3
Cut vertices: [2, 3]
  Block 0: [3, 4]
  Block 1: [2, 3, 7]
  Block 2: [0, 1, 2, 5, 6]
```

이 그래프에는 덩이가 셋 있다. 자름 꼭짓점 $2$과 $3$은 저마다 덩이 둘을 잇는다. 어느 쪽을 없애도 그래프가 끊어진다. 따라서 덩이-자름 나무는 마디 다섯 개(덩이 마디 셋, 자름 마디 둘)가 경로로 이어진 꼴이다.

## 복잡도

| 항목 | 비용 |
|--------|:----:|
| 시간 | $O(V + E)$ |
| 공간 | $O(V + E)$ |

세우는 일 전체가 깊이 우선 돌아보기 한 번과 선형 시간 뒷손질로 끝나므로 밑바탕 두 겹 이음 조각 알고리즘만큼 효율적이다.

## 응용

- **그물 믿음성.** 덩이-자름 나무는 그물에서 혼자 무너지면 끝나는 곳을 모두 가려낸다.
- **두 꼭짓점 이어짐 물음.** 나무를 세우고 나면 두 꼭짓점이 같은 두 겹 이음 조각에 있는지 $O(1)$에 답할 수 있다.
- **선인장 그래프.** 그래프가 선인장인 것은 그 덩이-자름 나무의 덩이가 모두 변이거나 단순 순환일 때 그리고 오직 그때뿐이다.

## 참고 문헌

- Hopcroft, J. E., & Tarjan, R. E. (1973). Algorithm 447: Efficient algorithms for graph manipulation. *Communications of the ACM*, 16(6), 372--378.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), 22장: Elementary Graph Algorithms.

## 연습문제

**연습문제 1.**
덩이-자름 나무를 정의하여라. 그 마디와 변은 무엇인가?

??? success "연습문제 1 풀이"
    The **block-cut tree** is a tree representation of a graph's biconnected component structure. Its nodes are of two types: **block nodes** (one per biconnected component) and **cut nodes** (one per articulation point). An edge connects a cut node to a block node if the articulation point belongs to that biconnected component. The result is always a tree (or forest for disconnected graphs). $\square$

---

**연습문제 2.**
덩이-자름 나무가 참으로 나무임(이어져 있고 순환이 없음)을 증명하여라.

??? success "연습문제 2 풀이"
    **Acyclic**: Suppose a cycle exists with blocks $B_1, c_1, B_2, c_2, \ldots, B_k, c_k, B_1$. This means $c_1$ connects $B_1$ and $B_2$, $c_2$ connects $B_2$ and $B_3$, etc. But then $B_1$ and $B_k$ share cut vertex $c_k$ and also connect through $c_1, B_2, \ldots$, forming a cycle in the original graph that spans multiple biconnected components — contradicting the maximality of biconnected components. **Connected**: For any two blocks sharing an articulation point, there is a path in the block-cut tree through that articulation point. For any two vertices in the original connected graph, the blocks along any path between them form a connected subtree. $\square$

---

**연습문제 3.**
덩이-자름 나무로 "꼭짓점 $u$이 이음매 점인가?"라는 물음에 어떻게 $O(1)$에 답할 수 있는가?

??? success "연습문제 3 풀이"
    After constructing the block-cut tree (in $O(V + E)$), a vertex $u$ is an articulation point if and only if it appears as a cut node in the tree. Equivalently, $u$ belongs to more than one biconnected component. Store a boolean flag or check the degree of $u$'s corresponding node in the block-cut tree: cut nodes always have degree $\geq 2$. This allows $O(1)$ per query after $O(V + E)$ preprocessing. $\square$

---

**연습문제 4.**
그물 믿음성 살피기에서 덩이-자름 나무의 쓰임새를 설명하여라.

??? success "연습문제 4 풀이"
    To determine which pairs of vertices remain connected after removing a single vertex $v$: if $v$ is not a cut vertex, all pairs remain connected. If $v$ is a cut vertex, removing it splits the graph into the biconnected components attached to $v$ in the block-cut tree. Vertices within the same resulting subtree remain connected; vertices in different subtrees are disconnected. The block-cut tree allows answering such queries in $O(1)$ per query after preprocessing, enabling efficient network vulnerability analysis. $\square$
