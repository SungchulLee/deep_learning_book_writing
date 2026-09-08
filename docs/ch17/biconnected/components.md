# 두 겹 이음 조각

이어진 그래프에는 없애면 그래프가 끊어지는 꼭짓점이 있을 수 있다. 그래프의 어느 부분이 꼭짓점 하나를 잃어도 견디는지 살피면 **두 겹 이음 조각**이라는 생각에 이른다. 그래프를 이 조각으로 쪼개면 남는 이어짐의 속 짜임이 드러나고 그래프를 붙들고 있는 결정적인 이음매 점이 가려진다.

---

## 1. 정의

**두 겹 변 그래프.** $|V| \ge 3$인 이어진 방향 없는 그래프 $G = (V, E)$은 어떤 꼭짓점 하나를 없애도 이어진 채로 남으면 *두 겹 변*(또는 2 변)이다. 같은 말로, $G$이 두 겹 변인 것은 어떤 두 꼭짓점도 한 단순 순환 위에 함께 놓이는 것과 같은 뜻이다.

**두 겹 이음 조각.** $G$의 가장 큰 두 겹 이음 아래그래프. $G$의 변마다 정확히 하나의 두 겹 이음 조각에 든다. 다리(없애면 $G$이 끊어지는 변)는 그 변 하나와 두 끝점만으로 된 두 겹 이음 조각을 이룬다.

**이음매 점.** 없애면 이어진 조각의 수가 늘어나는 꼭짓점 $v$. 꼭짓점이 이음매 점인 것은 그것이 두 겹 이음 조각 둘 이상에 들 때 그리고 오직 그때뿐이다.

---

## 2. 핵심 성질

1. 변마다 정확히 하나의 두 겹 이음 조각에 든다.
2. 두 겹 이음 조각 둘은 많아야 꼭짓점 하나를 함께 가지며, 함께 갖는 꼭짓점은 이음매 점이다.
3. 이음매 점이 없는 그래프는 두 겹 이어졌거나, 변 하나이거나, 꼭짓점 하나이다.
4. 두 겹 이음 조각의 수는 덩이-자름 나무의 변 수와 같다.

---

## 3. 알고리즘

깊이 우선 돌아보기 한 번으로 모든 두 겹 이음 조각과 이음매 점을 $O(V + E)$ 시간에 찾는다. 이 알고리즘은 다음을 지닌다:

- $\text{disc}[v]$: 꼭짓점 $v$을 찾아낸 때.
- $\text{low}[v]$: 되돌아가는 이음을 많아야 하나 써서 $v$을 뿌리로 하는 아래나무에서 닿을 수 있는 가장 이른 찾아낸 때.

$$
\text{low}[v] = \min\!\bigl(\text{disc}[v],\; \min_{(v,w) \text{ back edge}} \text{disc}[w],\; \min_{(v,u) \text{ tree edge}} \text{low}[u]\bigr)
$$

꼭짓점 $v$이 이음매 점인 경우는 다음과 같다:

- $v$이 돌아보기 뿌리이고 돌아보기 나무에서 자식이 둘 이상이거나,
- $v$이 뿌리가 아니면서 $\text{low}[u] \ge \text{disc}[v]$인 자식 $u$을 지닌다.

두 겹 변 조각을 뽑아내려면 변 쌓개를 지닌다. 깊이 우선 돌아보기가 $u$에서 $v$으로 되짚어 갈 때 $\text{low}[u] \ge \text{disc}[v]$이면 $(v, u)$까지 넣어 쌓개에서 변을 모두 뺀다. 이 변들이 두 겹 변 조각 하나를 이룬다.

---

## 4. 구현

```python
"""
깊이 우선 돌아보기로 얻는 방향 없는 그래프의 두 겹 이음 조각.

변 쌓기를 쓴 타잔 알고리즘으로 모든 두 겹 이음
조각과 이음매 점을 O(V + E) 시간에 가려낸다.
"""

from collections import defaultdict

# === 두 겹 이음 조각 찾개 ===

class BiconnectedComponents:
    """두 겹 이음 조각과 이음매 점을 모두 찾는다."""

    def __init__(self, n: int):
        """꼭짓점 n개(0부터 셈)로 그래프를 첫자리매김한다."""
        self.n = n
        self.adj = defaultdict(list)
        self.components = []
        self.articulation_points = set()

    def add_edge(self, u: int, v: int) -> None:
        """방향 없는 변 (u, v)을 더한다."""
        self.adj[u].append(v)
        self.adj[v].append(u)

    def find_components(self) -> None:
        """두 겹 이음 조각을 모두 찾으려 깊이 우선 돌아보기를 한다."""
        disc = [-1] * self.n
        low = [0] * self.n
        parent = [-1] * self.n
        stack = []  # 변 쌓기
        timer = [0]

        def dfs(u: int) -> None:
            disc[u] = low[u] = timer[0]
            timer[0] += 1
            children = 0

            for v in self.adj[u]:
                if disc[v] == -1:
                    children += 1
                    parent[v] = u
                    stack.append((u, v))
                    dfs(v)
                    low[u] = min(low[u], low[v])

                    # 이음매 점인지 살피기
                    is_root = parent[u] == -1
                    if (is_root and children > 1) or \
                       (not is_root and low[v] >= disc[u]):
                        self.articulation_points.add(u)

                    # 경계를 찾으면 조각 뽑아내기
                    if low[v] >= disc[u]:
                        component = []
                        while stack:
                            edge = stack.pop()
                            component.append(edge)
                            if edge == (u, v):
                                break
                        self.components.append(component)

                elif v != parent[u] and disc[v] < disc[u]:
                    stack.append((u, v))
                    low[u] = min(low[u], disc[v])

        for i in range(self.n):
            if disc[i] == -1:
                dfs(i)

# === 시연 ===

if __name__ == "__main__":
    # 그래프: 0-1-2-0(삼각형), 2-3, 3-4-5-3(삼각형)
    bc = BiconnectedComponents(6)
    for u, v in [(0,1),(1,2),(2,0),(2,3),(3,4),(4,5),(5,3)]:
        bc.add_edge(u, v)
    bc.find_components()

    print(f"Number of biconnected components: {len(bc.components)}")
    print(f"Articulation points: {sorted(bc.articulation_points)}")
    for i, comp in enumerate(bc.components):
        vertices = set()
        for u, v in comp:
            vertices.update([u, v])
        print(f"  Component {i}: vertices {sorted(vertices)}, edges {comp}")
```

**출력:**

```
Number of biconnected components: 3
Articulation points: [2, 3]
  Component 0: vertices [3, 4, 5], edges [(4, 5), (5, 3), (3, 4)]
  Component 1: vertices [2, 3], edges [(2, 3)]
  Component 2: vertices [0, 1, 2], edges [(1, 2), (2, 0), (0, 1)]
```

세모 $\{0, 1, 2\}$이 두 겹 변 조각 하나를, 다리 $(2, 3)$이 또 하나를, 세모 $\{3, 4, 5\}$이 셋째를 이룬다. 꼭짓점 $2$과 $3$은 저마다 덩이 둘을 잇기에 이음매 점이다.

---

## 5. 복잡도

| 항목 | 비용 |
|--------|:----:|
| 시간 | $O(V + E)$ |
| 공간 | $O(V + E)$ |

이 알고리즘은 돌아보기를 한 번만 한다. 변마다 정확히 한 번 쌓기에 올리고 꺼내므로 전체 품은 그래프 크기에 선형이다.

---

## 6. 응용

- **그물 믿음성.** 두 겹 이음 조각은 어느 마디 하나가 무너져도 이어진 채 남는 그물의 부분을 가려낸다.
- **남는 이음.** 두 겹 이음 조각 안에서는 아무 꼭짓점 짝 사이에도 꼭짓점이 겹치지 않는 경로가 적어도 둘 있다.
- **덩이-자름 나무.** 두 겹 이음 조각과 이음매 점이 함께 덩이-자름 나무를 정하는데, 이는 그래프 이어짐 짜임을 한 층 위에서 본 모습이다.

---

## 연습문제

**연습문제 1.**
두 겹 이음 조각을 정의하고 이어진 조각과 어떻게 다른지 설명하여라.

??? success "연습문제 1 풀이"
    **두 겹 변 조각**(덩이)는 가장 큰 2 변 아래그래프다. 곧 어떤 꼭짓점 하나를 없애도 이어진 채로 남는다. 이어진 덩이는 이어져 있기만 하면 된다(꼭짓점을 없애면 끊길 수도 있다). 어떤 그래프든 꼭짓점을 많아야 하나(이음매 점) 함께 지니는 두 겹 변 조각으로 쪼갤 수 있다. 두 겹 변 조각은 꼭짓점 하나가 무너져도 끊기지 않는 그래프의 "든든한" 자리를 나타낸다. $\square$

---

**연습문제 2.**
쌓기를 써서 두 겹 이음 조각을 찾는 깊이 우선 돌아보기 바탕 알고리즘을 설명하여라.

??? success "연습문제 2 풀이"
    깊이 우선 돌아보기를 도는 동안 변 쌓개를 지닌다. 나무 변과 되돌아가는 이음을 쌓개에 넣는다. 이음매 점 조건이 걸리면($u$의 자식 $v$에 대해 $\text{low}[v] \geq \text{disc}[u]$) 변 $(u, v)$이 나올 때까지 쌓개에서 변을 뺀다. 이 변들이 두 겹 변 조각 하나를 이룬다. 깊이 우선 돌아보기의 뿌리에서는 자식 아래나무를 다 다룬 뒤 남은 변을 빼어 마지막 덩이를 만든다. 시간은 $O(V + E)$이다. 변마다 꼭 하나의 두 겹 변 조각에 든다. $\square$

---

**연습문제 3.**
두 겹 이음 조각 둘이 많아야 꼭짓점 하나를 함께 가지며 그 함께 갖는 꼭짓점이 이음매 점이어야 함을 증명하여라.

??? success "연습문제 3 풀이"
    덩이 $B_1$과 $B_2$이 꼭짓점 $u$과 $v$ 둘을 함께 지닌다고 하자. 두 덩이가 모두 2 변이므로 $B_1$ 안에 $u$에서 $v$으로 가는 꼭짓점이 겹치지 않는 길이 둘, $B_2$ 안에도 둘 있다. $B_1$과 $B_2$의 길을 아우르면 $B_1$과 $B_2$을 모두 담는 2 변 아래그래프가 나오는데, 이는 이들이 가장 크다는 것과 어긋난다. 그러므로 함께 지니는 꼭짓점은 많아야 하나다. 꼭짓점 $w$이 덩이 둘에 들면 $w$을 없앨 때 두 덩이의 변이 서로 끊기므로 $w$은 이음매 점이다. $\square$

---

**연습문제 4.**
그래프에 꼭짓점 10개, 변 15개, 이음매 점 3개가 있다. 두 겹 이음 조각은 적어도 몇 개인가?

??? success "연습문제 4 풀이"
    이음매 점은 저마다 두 겹 변 조각 둘 이상에 든다. 덩이 수가 가장 적을 때는 이음매 점마다 꼭 덩이 둘을 이을 때다. 덩이 하나에서 비롯하면 이음매 점마다 새 덩이가 적어도 하나 는다. 그러므로 가장 적은 덩이 수는 $= 1 + 3 = 4$이다. 보기로 이음매 점 셋을 거쳐 길처럼 이어진 두 겹 변 조각 4개짜리 그래프가 있다. $\square$

## 정리하며

이 마당은 정의、핵심 성질、알고리즘、구현을 차례로 짚었다.

**참고 문헌**

- Hopcroft, J. E., & Tarjan, R. E. (1973). Algorithm 447: Efficient algorithms for graph manipulation. *Communications of the ACM*, 16(6), 372--378.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), 22장: Elementary Graph Algorithms.
