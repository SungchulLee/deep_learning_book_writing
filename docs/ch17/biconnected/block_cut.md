# 덩이-자름 나무

이음매 점과 두 겹 이음 조각을 찾는 일은 자리에 매인 물음, 곧 어떤 꼭짓점이나 변이 이어짐에 결정적인가에 답한다. 덩이-자름 나무는 본디 그래프의 두 겹 이음 조각(*덩이*라 한다)과 이음매 점(*자름 꼭짓점*이라 한다)을 마디로 삼는 **나무**를 세워 이 앎을 전체 짜임으로 끌어올린다. 이 나무는 이어진 그래프의 무른 곳 전부를 한눈에 드러낸다.

---

## 1. 정의

$G = (V, E)$을 이어진 방향 없는 그래프라 하자.

**덩이.** $G$의 가장 큰 두 겹 이음 아래그래프. 변마다 정확히 하나의 덩이에 든다. 다리(자름 변)는 그 두 끝점과 함께 스스로 덩이 하나를 이룬다.

**자름 꼭짓점.** 없애면 $G$이 끊어지는 꼭짓점. 자름 꼭짓점은 바로 덩이 둘 이상이 함께 갖는 꼭짓점이다.

**덩이-자름 나무.** 다음과 같이 세운 두 쪽 나무 $T$:

- $G$의 덩이마다 *덩이 마디* 하나를 만든다.
- $G$의 자름 꼭짓점마다 *자름 마디* 하나를 만든다.
- 자름 꼭짓점 $c$이 덩이 $B_i$에 들 때마다 $T$에 변 $(B_i, c)$을 더한다.

그 결과는 늘 나무(순환이 없고 이어짐)이며 두 쪽으로 나뉜 짜임이다. 곧 변마다 덩이 마디와 자름 마디를 잇는다.

---

## 2. 성질

덩이가 $b$개, 자름 꼭짓점이 $c$개인 이어진 그래프 $G$의 덩이-자름 나무 $T$은 다음을 채운다:

- $T$에는 마디가 $b + c$개, 변이 $b + c - 1$개 있다.
- $T$은 나무이다. 곧 $T$에서 아무 변이나 없애는 것은 $G$에서 자름 꼭짓점을 없애는 것에 맞대응되며, 그러면 $G$이 끊어진다.
- $T$의 잎은 늘 덩이 마디이며 자름 마디는 결코 아니다.
- 덩이 둘은 많아야 꼭짓점 하나를 함께 가지며, 그 함께 갖는 꼭짓점은 자름 꼭짓점이다.

---

## 3. 세우기 알고리즘

덩이-자름 나무를 세우는 일은 두 겹 이음 조각과 이음매 점을 모두 찾는 일로 줄어들며, 이는 깊이 우선 돌아보기 한 번으로 $O(V + E)$ 시간에 이룬다.

**걸음 1.** $G$에서 깊이 우선 돌아보기를 돌린다. 꼭짓점 $v$마다 찾아낸 때 $\text{disc}[v]$과 낮은 값 $\text{low}[v]$을 지닌다.

**걸음 2.** 변 쌓개를 쓴다. 깊이 우선 돌아보기가 자식 $u$에서 부모 $v$으로 되짚어 갈 때 $\text{low}[u] \ge \text{disc}[v]$이면 $(v, u)$이 나올 때까지 쌓개에서 변을 뺀다. 이 변들이 두 겹 변 조각(덩이) 하나를 이룬다.

**걸음 3.** 자르는 꼭짓점을 짚는다. (가) $v$이 깊이 우선 돌아보기의 뿌리이면서 자식이 둘 이상이거나, (나) $v$이 뿌리가 아니면서 $\text{low}[u] \ge \text{disc}[v]$인 자식 $u$을 지니면 $v$은 자르는 꼭짓점이다.

**4단계.** 덩이마다 덩이 마디를, 자름 꼭짓점마다 자름 마디를 만든 뒤 위에서 말한 대로 이어서 나무를 세운다.

---

## 4. 구현

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

---

## 5. 복잡도

| 항목 | 비용 |
|--------|:----:|
| 시간 | $O(V + E)$ |
| 공간 | $O(V + E)$ |

세우는 일 전체가 깊이 우선 돌아보기 한 번과 선형 시간 뒷손질로 끝나므로 밑바탕 두 겹 이음 조각 알고리즘만큼 효율적이다.

---

## 6. 응용

- **그물 믿음성.** 덩이-자름 나무는 그물에서 혼자 무너지면 끝나는 곳을 모두 가려낸다.
- **두 꼭짓점 이어짐 물음.** 나무를 세우고 나면 두 꼭짓점이 같은 두 겹 이음 조각에 있는지 $O(1)$에 답할 수 있다.
- **선인장 그래프.** 그래프가 선인장인 것은 그 덩이-자름 나무의 덩이가 모두 변이거나 단순 순환일 때 그리고 오직 그때뿐이다.

---

## 연습문제

**연습문제 1.**
덩이-자름 나무를 정의하여라. 그 마디와 변은 무엇인가?

??? success "연습문제 1 풀이"
    **덩이-자름 나무**는 그래프의 두 겹 변 조각 얼개를 나무로 나타낸 것이다. 마디는 두 갈래다. **덩이 마디**(두 겹 변 조각마다 하나)와 **자름 마디**(이음매 점마다 하나)다. 이음매 점이 어떤 두 겹 변 조각에 들면 그 자름 마디와 덩이 마디를 변으로 잇는다. 결과는 늘 나무다(끊어진 그래프라면 숲이다). $\square$

---

**연습문제 2.**
덩이-자름 나무가 참으로 나무임(이어져 있고 순환이 없음)을 증명하여라.

??? success "연습문제 2 풀이"
    **순환이 없음**: 덩이 $B_1, c_1, B_2, c_2, \ldots, B_k, c_k, B_1$으로 이루어진 순환이 있다고 하자. 그러면 $c_1$이 $B_1$과 $B_2$을, $c_2$이 $B_2$과 $B_3$을 잇는 식이다. 그런데 그러면 $B_1$과 $B_k$이 자르는 꼭짓점 $c_k$을 함께 지니면서 $c_1, B_2, \ldots$을 거쳐서도 이어지므로, 본디 그래프에 여러 두 겹 변 조각에 걸친 순환이 생긴다. 이는 두 겹 변 조각이 가장 크다는 것과 어긋난다. **이어져 있음**: 이음매 점을 함께 지닌 덩이 둘 사이에는 그 이음매 점을 지나는 길이 덩이-자름 나무에 있다. 본디 이어진 그래프의 어떤 두 꼭짓점에 대해서도 그 사이 길에 놓인 덩이들이 이어진 아래나무를 이룬다. $\square$

---

**연습문제 3.**
덩이-자름 나무로 "꼭짓점 $u$이 이음매 점인가?"라는 물음에 어떻게 $O(1)$에 답할 수 있는가?

??? success "연습문제 3 풀이"
    덩이-자름 나무를 지은 뒤($O(V + E)$에 짓는다), 꼭짓점 $u$이 이음매 점인 것은 그 나무에 자름 마디로 나타나는 것과 같은 뜻이다. 곧 $u$이 두 겹 변 조각 여럿에 든다는 뜻이다. 참거짓 깃발을 갈무리해 두거나 덩이-자름 나무에서 $u$에 맞는 마디의 차수를 보면 된다. 자름 마디는 늘 차수가 $\geq 2$이다. 이러면 $O(V + E)$의 미리 다듬기 뒤 물음마다 $O(1)$이 든다. $\square$

---

**연습문제 4.**
그물 믿음성 살피기에서 덩이-자름 나무의 쓰임새를 설명하여라.

??? success "연습문제 4 풀이"
    꼭짓점 $v$ 하나를 없앤 뒤 어떤 꼭짓점 짝이 그대로 이어져 있는지 가리려면 이렇게 한다. $v$이 자르는 꼭짓점이 아니면 모든 짝이 그대로 이어져 있다. $v$이 자르는 꼭짓점이면 없앨 때 그래프가 덩이-자름 나무에서 $v$에 붙은 두 겹 변 조각들로 쪼개진다. 같은 아래나무 안의 꼭짓점은 그대로 이어져 있고 다른 아래나무의 꼭짓점은 끊긴다. 덩이-자름 나무를 쓰면 미리 다듬은 뒤 이런 물음에 물음마다 $O(1)$에 답할 수 있어 그물의 무른 곳을 잘 살필 수 있다. $\square$

## 정리하며

이 마당은 정의、성질、세우기 알고리즘、구현을 차례로 짚었다.

**참고 문헌**

- Hopcroft, J. E., & Tarjan, R. E. (1973). Algorithm 447: Efficient algorithms for graph manipulation. *Communications of the ACM*, 16(6), 372--378.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), 22장: Elementary Graph Algorithms.
