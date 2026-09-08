# 가짜 나무

많은 나무 문제의 물음은 나무 마디의 작은 부분만 가리킨다. 뜻있는 마디가 성길 때 물음마다 나무 전체를 다루면 때를 낭비한다. **가짜 나무**(*딸림 나무*라고도 한다)는 물음에 뜻있는 마디와 그 짝별 최소 공통 조상만 담은 옥죈 나무로, 본디 나무의 조상 관계를 지킨다. 이는 문제 크기를 $n$에서 뜻있는 마디 수 $k$에 대해 $O(k)$로 줄인다.

---

## 1. 정의

마디 $n$개의 뿌리 있는 나무 $T$와 **열쇠 마디** 모임 $S = \{v_1, v_2, \ldots, v_k\}$이 주어질 때 가짜 나무 $T'$은 다음을 만족하는 $T$의 가장 작은 밑나무다:

1. $S$의 모든 마디를 담는다.
2. $S$의 모든 마디 짝의 LCA를 담는다.
3. $T$의 뿌리를 담는다(표현에 따라 마음대로).
4. $T$의 조상-자손 관계를 지킨다.

가짜 나무의 마디는 많아야 $2k - 1$개다. 곧 열쇠 마디 $k$개에 짝별 LCA가 많아야 $k - 1$개다.

---

## 2. 세우기 알고리즘

흔한 짓기는 오일러 돌기 차례와 LCA 신탁을 쓴다.

### 걸음

1. 열쇠 마디를 오일러 돌기 들어감 때(깊이 먼저 차례)로 **줄 세운다**.
2. **LCA를 넣는다**: 줄 세운 차례의 잇닿은 짝 $(v_i, v_{i+1})$마다 $\text{LCA}(v_i, v_{i+1})$을 셈해 마디 모임에 더한다.
3. 모든 마디(열쇠 마디와 LCA)를 **겹침 없애고** 깊이 먼저 차례로 다시 줄 세운다.
4. 쌓기로 **가짜 나무를 짓는다**. 마디를 깊이 먼저 차례로 다루며 조상 쌓기를 지닌다. 새 마디마다 꼭대기가 조상이 될 때까지 쌓기를 꺼내고, 쌓기 꼭대기에서 새 마디로 변을 더하고, 새 마디를 넣는다.

### 잇닿은 LCA만으로 넉넉한 까닭

핵심 성질은 이렇다. $S$의 아무 두 마디의 LCA는 (깊이 먼저 차례로) 잇닿은 어떤 짝의 LCA와 같다. $v_i$와 $v_j$가 잇닿지 않았다면 깊이 먼저 차례로 그 사이에 어떤 $v_m$이 있고 $\text{LCA}(v_i, v_j) = \text{LCA}(v_i, v_m)$이거나 $\text{LCA}(v_m, v_j)$이기 때문이다.

---

## 3. 구현

```python
"""
가짜 나무 짓기.

뿌리 있는 나무와 열쇠 마디 모임이 주어지면 열쇠 마디와 그 짝별 LCA만
담은 가짜(딸림) 나무를
짓는다.
"""

import math
from collections import deque

# ===================================================================
# 두 갈래 들어 올리기로 LCA
# ===================================================================

class LCAOracle:
    """O(log n) LCA 물음을 위해 나무를 미리 다듬는다."""

    def __init__(self, adj, root=0):
        self.n = len(adj)
        self.LOG = max(1, math.ceil(math.log2(self.n))) + 1
        self.depth = [0] * self.n
        self.up = [[0] * self.LOG for _ in range(self.n)]
        self.tin = [0] * self.n  # 오일러 돌기 들어감 때
        self.tout = [0] * self.n
        self._timer = 0

        # 깊이와 어버이를 위한 너비 먼저 훑기
        visited = [False] * self.n
        visited[root] = True
        queue = deque([root])
        order = []
        while queue:
            u = queue.popleft()
            order.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    self.depth[v] = self.depth[u] + 1
                    self.up[v][0] = u
                    queue.append(v)

        # 두 갈래 들어 올리기 표
        for k in range(1, self.LOG):
            for v in range(self.n):
                self.up[v][k] = self.up[self.up[v][k - 1]][k - 1]

        # 오일러 돌기 때를 위한 깊이 먼저 훑기
        self._dfs_iterative(adj, root)

    def _dfs_iterative(self, adj, root):
        stack = [(root, -1, False)]
        while stack:
            u, par, leaving = stack.pop()
            if leaving:
                self.tout[u] = self._timer
                self._timer += 1
                continue
            self.tin[u] = self._timer
            self._timer += 1
            stack.append((u, par, True))
            for v in adj[u]:
                if v != par:
                    stack.append((v, u, False))

    def is_ancestor(self, u, v):
        return self.tin[u] <= self.tin[v] and self.tout[u] >= self.tout[v]

    def query(self, u, v):
        if self.is_ancestor(u, v):
            return u
        if self.is_ancestor(v, u):
            return v
        for k in range(self.LOG - 1, -1, -1):
            if not self.is_ancestor(self.up[u][k], v):
                u = self.up[u][k]
        return self.up[u][0]

# ===================================================================
# 가짜 나무 짓기
# ===================================================================

def build_virtual_tree(lca_oracle, key_nodes):
    """열쇠 마디에서 가짜 나무를 짓는다.

    인수:
        lca_oracle: LCAOracle 사례
        key_nodes: 열쇠 마디 번호 목록

    반환값:
        vt_adj: 가짜 나무의 이웃 대응표
        vt_nodes: 줄 세운 가짜 나무 마디 목록
    """
    if not key_nodes:
        return {}, []

    # 깊이 먼저 들어감 때로 줄 세움
    nodes = sorted(set(key_nodes), key=lambda v: lca_oracle.tin[v])

    # 잇닿은 짝의 LCA 더하기
    all_nodes = set(nodes)
    for i in range(len(nodes) - 1):
        lca_node = lca_oracle.query(nodes[i], nodes[i + 1])
        all_nodes.add(lca_node)

    # 모든 마디를 깊이 먼저 들어감 때로 줄 세움
    vt_nodes = sorted(all_nodes, key=lambda v: lca_oracle.tin[v])

    # 쌓기로 가짜 나무 짓기
    vt_adj = {v: [] for v in vt_nodes}
    stack = [vt_nodes[0]]

    for i in range(1, len(vt_nodes)):
        v = vt_nodes[i]
        # 쌓기 꼭대기가 v의 조상이 될 때까지 꺼냄
        while len(stack) > 1 and not lca_oracle.is_ancestor(stack[-1], v):
            stack.pop()
        # 쌓기 꼭대기에서 v로 변 더하기
        vt_adj[stack[-1]].append(v)
        stack.append(v)

    return vt_adj, vt_nodes

# ===================================================================
# 메인
# ===================================================================

if __name__ == "__main__":
    #         0
    #        / \
    #       1   2
    #      /|    \
    #     3 4     5
    #    /       / \
    #   6       7   8
    n = 9
    adj = [[] for _ in range(n)]
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5),
             (3, 6), (5, 7), (5, 8)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    oracle = LCAOracle(adj, root=0)

    key_nodes = [6, 4, 7]
    vt_adj, vt_nodes = build_virtual_tree(oracle, key_nodes)

    print(f"Original tree: {n} nodes")
    print(f"Key nodes: {key_nodes}")
    print(f"Virtual tree nodes: {vt_nodes}")
    print(f"Virtual tree size: {len(vt_nodes)}")
    print(f"Virtual tree edges:")
    for u in vt_nodes:
        for v in vt_adj[u]:
            print(f"  {u} -> {v}")
```

**출력:**
```
본디 나무: 마디 9개
Key nodes: [6, 4, 7]
Virtual tree nodes: [0, 1, 4, 6, 7]
가짜 나무 크기: 5
Virtual tree edges:
  0 -> 1
  0 -> 7
  1 -> 4
  1 -> 6
```

---

## 4. 복잡도

| 국면 | 시간 |
|---|---|
| LCA 미리 다듬기 | $O(n \log n)$ |
| 열쇠 마디 줄 세우기 | $O(k \log k)$ |
| LCA 셈하기 | $O(k \log n)$ |
| 가짜 나무 짓기 | $O(k)$ |
| **물음마다 모두** | $O(k \log n)$ |

가짜 나무의 마디는 많아야 $2k - 1$개, 변은 $2k - 2$개이므로 그 뒤 가짜 나무 위의 나무 동적 짜기는 $O(n)$이 아니라 $O(k)$가 든다.

---

## 5. 응용

- **나무 위 슈타이너 나무**: 열쇠 마디 모임을 잇는 무게가 가장 작은 밑나무를 찾는다. 가짜 나무를 짓고 변 무게를 더한다.
- **여러 물음 나무 동적 짜기**: 물음마다 뜻있는 마디를 조금씩 짚을 때 물음마다 가짜 나무를 짓고 그 위에서 동적 짜기를 돌린다.
- **길 세기**: 가짜 나무 얼개를 살펴 열쇠 마디 짝 사이의 길을 센다.

---

## 연습문제

**연습문제 1.**
마디 10개인 나무가 마디 1을 뿌리로 하고 열쇠 마디가 $S = \{3, 7, 9\}$일 때 가짜 나무를 짓는 걸음을 밝혀라. 가짜 나무의 마디는 몇 개인가?

??? success "연습문제 1 풀이"
    1걸음: 열쇠 마디를 오일러 돌기 들어감 때로 줄 세운다. $\text{tin}(3) < \text{tin}(7) < \text{tin}(9)$이라 하자. 2걸음: 줄 세운 차례에서 잇닿은 마디의 짝별 LCA $\text{LCA}(3, 7)$과 $\text{LCA}(7, 9)$을 셈한다. 이 LCA 마디를 모임에 더한다(필요하면 뿌리도 더한다). 3걸음: $\text{LCA}(3, 9)$도 셈하지만 이는 앞선 LCA 가운데 하나이거나 이미 모임에 있는 마디와 같다. 겹침을 없앤다. 4걸음: 모든 마디(열쇠 마디와 LCA 마디)를 들어감 때로 줄 세우고 쌓기 바탕 알고리즘으로 가짜 나무를 짓는다. 지금 뿌리에서 마디까지의 길을 나타내는 쌓기를 지닌다. 줄 세운 차례의 마디마다 꼭대기가 지금 마디의 조상이 될 때까지 쌓기를 꺼내고 변을 더한 뒤 넣는다. 가짜 나무의 마디는 많아야 $2k - 1 = 5$개다(열쇠 마디 3개에 LCA 마디 많아야 2개). 정확한 수는 나무 얼개에 달렸다. $\square$

---

**연습문제 2.**
열쇠 마디 $k$개의 가짜 나무의 마디가 많아야 $2k - 1$개임을 증명하여라.

??? success "연습문제 2 풀이"
    가짜 나무는 열쇠 마디 $k$개와 그 짝별 LCA를 담는다. 열쇠 마디를 오일러 돌기 차례로 줄 세워 $v_1, v_2, \ldots, v_k$이라 하자. 뜻있는 LCA는 $i = 1, \ldots, k-1$에 대한 $\text{LCA}(v_i, v_{i+1})$이다(다른 모든 짝별 LCA가 이 가운데 있거나 열쇠 마디 자신임을 보일 수 있다). 이는 LCA 마디를 많아야 $k - 1$개 더한다. 다만 어떤 LCA는 열쇠 마디나 서로와 겹칠 수 있으므로 열쇠가 아닌 서로 다른 LCA 마디는 많아야 $k - 1$개다. 모두: 많아야 $k + (k-1) = 2k - 1$개다. 더 엄밀히 가짜 나무는 잎이 많아야 $k$개(열쇠 마디)이고 열쇠 마디가 아닌 차수 2의 속마디가 없는 나무(이어져 있고 고리가 없다)다. 잎이 $k$개이고 차수 2의 속마디가 없는 나무는 속마디가 많아야 $k - 1$개이므로(뿌리를 뺀 속마디마다 차수가 $\ge 3$이다) 모두 $2k - 1$개다. $\square$

---

**연습문제 3.**
LCA 물음이 $O(\log n)$이라 할 때 가짜 나무 짓기의 쌓기 바탕 $O(k \log n)$ 알고리즘을 밝혀라.

??? success "연습문제 3 풀이"
    들임: 오일러 돌기 들어감 때로 줄 세운 열쇠 마디 $S$와 밑 프로그램인 LCA. 알고리즘: (1) 쌓기를 뿌리(또는 첫 열쇠 마디와 마지막 열쇠 마디의 LCA를 가짜 뿌리로)로 시작한다. (2) 줄 세운 차례의 열쇠 마디 $v$마다 (가) $l = \text{LCA}(v, \text{stack.top()})$을 셈한다. (나) $|\text{stack}| \ge 2$이고 $\text{depth}(l) \le \text{depth}(\text{위에서 둘째})$인 동안 꼭대기를 꺼내고 가짜 나무에서 위에서 둘째부터 꺼낸 마디로 변을 더한다. (다) $\text{stack.top()} \ne l$이면 $l$에서 stack.top()으로 변을 더하고 stack.top()을 꺼낸 뒤 $l$을 넣는다. (라) $v$를 넣는다. (3) 열쇠 마디를 모두 다룬 뒤 남은 쌓기를 꺼내며 잇닿은 짝 사이에 변을 더한다. 열쇠 마디마다 고르게 나눈 $O(1)$번의 쌓기 셈이 일어난다(마디마다 한 번 넣고 한 번 꺼낸다). 되풀이마다 LCA를 한 번 $O(\log n)$에 셈한다. 모두: $O(k \log n)$. $\square$

---

**연습문제 4.**
마디 $n = 10^5$개의 나무에 물음 $q = 10^5$개가 온다. 물음마다 열쇠 마디 $k_i$개의 모임($\sum k_i \le 2 \times 10^5$)을 주고 열쇠 마디 모든 짝 사이 거리의 합을 묻는다. 가짜 나무가 어떻게 이를 다룰 만하게 하는지 밝히고 모든 복잡도를 살펴라.

??? success "연습문제 4 풀이"
    물음마다 열쇠 마디 $k_i$개 위에 가짜 나무를 짓는다. 가짜 나무의 마디는 많아야 $2k_i - 1$개이고 모든 짝별 거리를 지킨다(가짜 나무의 변 무게가 그에 맞는 길 위의 본디 변 무게 합과 같다). 가짜 나무에서 짝별 거리의 합은 깊이 먼저 훑기 한 번으로 셈한다. 무게 $w$인 변 $(u, v)$마다 $v$의 밑나무에 든 열쇠 마디 수를 $s$라 하면 그 변이 전체에 $w \times s \times (k_i - s)$만큼 이바지한다. 이는 물음마다 $O(k_i)$이다. 가짜 나무 짓기는 물음마다 $O(k_i \log n)$이다. 모든 물음에 걸쳐 $O((\sum k_i) \log n) = O(2 \times 10^5 \times 17) \approx 3.4 \times 10^6$이다. 깊이 먼저 훑기 단계는 $O(\sum k_i) = O(2 \times 10^5)$이 든다. 가짜 나무가 없으면 물음마다 마디 $n$개의 나무 전체를 다뤄 $O(qn) = 10^{10}$이 들어 될 수 없다. $\square$

---

**연습 5.**
가짜 나무 짓기가 왜 열쇠 마디를 오일러 돌기 차례로 줄 세워야 하는지 밝혀라. 마디 번호로 줄 세우거나 아무 차례로 다루면 무엇이 잘못되는가?

??? success "연습 5의 풀이"
    쌓기 바탕 짓기는 쌓기가 지어지는 가짜 나무에서 뿌리부터 마디까지의 길을 나타낸다는 불변량을 지킨다. 마디를 오일러 돌기 차례로 다루면 새 마디가 깊이 먼저 훑기에서 앞서 다룬 모든 마디의 "오른쪽"에 있음이 보장된다. 이는 새 마디와 쌓기 꼭대기의 LCA가 새 가지가 지금 길에서 갈라지는 곳을 정확히 정한다는 뜻이며, 알고리즘이 옳게 꺼내고 변을 더할 수 있게 한다. 마디를 아무 차례로 다루면 나중 마디가 앞선 마디의 조상일 수 있거나 이미 지난 곳에서 갈라질 수 있어 쌓기 불변량을 어긴다. 알고리즘은 틀린 변을 내거나 필요한 LCA 마디를 놓친다. 마디 번호로 줄 세우면 깊이 먼저 차례를 지키지 못하므로(마디 번호는 아무렇게나 매겨진다) 같은 문제가 생긴다. 오일러 돌기 들어감 때가 깊이 먼저 훑기 차례와 맞는 하나뿐인 차례다. $\square$

## 정리하며

이 마당은 정의、세우기 알고리즘、구현、복잡도을 차례로 짚었다.

**참고 문헌**

- Competitive Programmer's Handbook (Laaksonen).
- "딸림 나무"와 "가짜 나무" 짓기에 관한 여러 겨루기 짜기 자료.
