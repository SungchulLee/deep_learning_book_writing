# 최소 공통 조상

뿌리 있는 나무에서 두 마디 $u$와 $v$의 **최소 공통 조상**(LCA)은 둘 모두의 조상 가운데 가장 깊은 마디다. 보기로 나무에서 두 마디 사이의 거리를 찾으려면 가장 깊은 공통 조상을 알아야 한다. $u$에서 $v$까지의 길은 늘 $\text{LCA}(u, v)$을 지난다. LCA 물음은 거리 셈하기, 길 물음에 답하기, 가짜 나무 짓기, 범위 문제 풀기 등 나무 알고리즘 곳곳에 나온다. 좋은 LCA 알고리즘은 이런 일을 물음마다 $O(n)$에서 $O(\log n)$이나 $O(1)$로까지 줄인다.

---

## 1. 정의와 성질

$T$를 뿌리가 $r$인 뿌리 있는 나무라 하자. 아무 마디 $u$에 대해 $u$의 조상 모임은 $u$에서 $r$까지 길 위의 마디 모임이다($r$을 넣는다). $u$와 $v$의 LCA는 다음과 같다:

$$
\text{LCA}(u, v) = \arg\max_{w} \{\text{depth}(w) : w \text{ is an ancestor of both } u \text{ and } v\}
$$

핵심 성질은 다음과 같다.

- 모든 마디 $u$에 대해 $\text{LCA}(u, u) = u$이다.
- $u$가 $v$의 조상일 때 그리고 그때에만 $\text{LCA}(u, v) = u$이다.
- $u$에서 $v$까지의 길은 $\text{LCA}(u, v)$을 지난다.
- 나무에서 $u$와 $v$ 사이의 거리는 $\text{depth}(u) + \text{depth}(v) - 2 \cdot \text{depth}(\text{LCA}(u, v))$이다.

---

## 2. 두 갈래 들어 올리기

겨루기 짜기에서 가장 널리 쓰이는 LCA 알고리즘은 **두 갈래 들어 올리기**(두 배 방법이라고도 한다)다. 나무를 $O(n \log n)$ 때에 미리 다듬고 LCA 물음마다 $O(\log n)$에 답한다.

### 미리 다듬기

마디 $u$와 거듭제곱 $k$마다 $u$의 $2^k$번째 조상 $\text{up}[u][k]$을 갈무리한다:

$$
\text{up}[u][k] = \text{up}[\text{up}[u][k-1]][k-1]
$$

밑동은 $\text{up}[u][0] = \text{parent}(u)$이다. 표는 줄이 $n$개, 칸이 $\lceil \log_2 n \rceil$개다.

### 물음 알고리즘

$\text{LCA}(u, v)$을 찾으려면:

1. 더 깊은 마디를 들어 올려 $u$와 $v$를 같은 깊이로 맞춘다.
2. $u = v$이면 $u$를 돌려준다.
3. 2의 거듭제곱을 줄여 가며 $u$와 $v$를 함께 들어 올리되 LCA 바로 아래에서 멈춘다.
4. $\text{parent}(u)$을 돌려준다.

```python
"""
두 갈래 들어 올리기로 최소 공통 조상 찾기.

뿌리 있는 나무를 O(n log n)에 미리 다듬고 LCA 물음마다
O(log n)에 답한다.
"""

import math
from collections import deque

# ===================================================================
# 두 갈래 들어 올리기 LCA
# ===================================================================

class LCA:
    """뿌리 있는 나무에서 두 갈래 들어 올리기로 LCA 찾기."""

    def __init__(self, adj, root=0):
        """두 갈래 들어 올리기 표를 짓는다.

        인수:
            adj: 이웃 목록(목록의 목록)
            root: 뿌리 마디 번호
        """
        self.n = len(adj)
        self.LOG = max(1, math.ceil(math.log2(self.n))) + 1
        self.depth = [0] * self.n
        self.up = [[0] * self.LOG for _ in range(self.n)]

        # 깊이와 어버이(up[v][0])를 셈하는 너비 먼저 훑기
        visited = [False] * self.n
        visited[root] = True
        queue = deque([root])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    self.depth[v] = self.depth[u] + 1
                    self.up[v][0] = u
                    queue.append(v)

        # 두 갈래 들어 올리기 표 채우기
        for k in range(1, self.LOG):
            for v in range(self.n):
                self.up[v][k] = self.up[self.up[v][k - 1]][k - 1]

    def query(self, u, v):
        """마디 u와 v의 LCA를 돌려준다."""
        # 1걸음: 같은 깊이로 맞춤
        if self.depth[u] < self.depth[v]:
            u, v = v, u
        diff = self.depth[u] - self.depth[v]
        for k in range(self.LOG):
            if (diff >> k) & 1:
                u = self.up[u][k]

        if u == v:
            return u

        # 2걸음: LCA 바로 아래까지 둘을 들어 올림
        for k in range(self.LOG - 1, -1, -1):
            if self.up[u][k] != self.up[v][k]:
                u = self.up[u][k]
                v = self.up[v][k]

        return self.up[u][0]

    def distance(self, u, v):
        """u와 v 사이의 거리(변 수)를 돌려준다."""
        w = self.query(u, v)
        return self.depth[u] + self.depth[v] - 2 * self.depth[w]

# ===================================================================
# 메인
# ===================================================================

if __name__ == "__main__":
    # 나무 짓기:
    #       0
    #      / \
    #     1   2
    #    / \   \
    #   3   4   5
    #  /
    # 6
    n = 7
    adj = [[] for _ in range(n)]
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (3, 6)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    lca = LCA(adj, root=0)

    print(f"LCA(3, 4) = {lca.query(3, 4)}")  # 1
    print(f"LCA(6, 5) = {lca.query(6, 5)}")  # 0
    print(f"LCA(6, 4) = {lca.query(6, 4)}")  # 1
    print(f"LCA(6, 6) = {lca.query(6, 6)}")  # 6
    print(f"dist(6, 5) = {lca.distance(6, 5)}")  # 5
```

**출력:**
```
LCA(3, 4) = 1
LCA(6, 5) = 0
LCA(6, 4) = 1
LCA(6, 6) = 6
dist(6, 5) = 5
```

---

## 3. 복잡도 분석

| 마디 | 시간 | 공간 |
|---|---|---|
| 미리 다듬기 | $O(n \log n)$ | $O(n \log n)$ |
| 물음 | $O(\log n)$ | -- |

두 갈래 들어 올리기 표의 $O(n \log n)$ 자리는 $n \le 10^5$에서 받아들일 만하다($\log n \approx 17$일 때 대략 170만 칸).

---

## 4. 다른 길

### 오일러 돌기와 성긴 표

$O(n \log n)$ 미리 다듬기 뒤 물음마다 $O(1)$을 이루는 다른 길도 있다:

1. 나무의 오일러 돌기를 셈하며 걸음마다 깊이를 적는다.
2. $u$와 $v$의 LCA는 오일러 돌기에서 $u$와 $v$가 처음 나타난 자리 사이에 깊이가 가장 작은 마디다.
3. 이는 범위 최소 물음(RMQ)이며 성긴 표로 $O(1)$에 풀린다.

이 길은 [오일러 돌기](euler_tour.md) 마디에서 자세히 다룬다.

### 타잔의 묶음 처리 LCA

물음을 모두 미리 알 때 타잔의 묶음 처리 알고리즘은 깊이 먼저 훑기와 합치기-찾기로 물음 $q$개에 모두 $O(n \cdot \alpha(n) + q)$에 답한다. 이는 가장 좋지만 묶음 처리가 필요하다.

---

## 5. 응용

- **나무 거리 물음**: $\text{dist}(u, v) = \text{depth}(u) + \text{depth}(v) - 2 \cdot \text{depth}(\text{LCA}(u, v))$.
- **길 모으기**: LCA에서 갈라 $u$에서 $v$까지 길의 합, 최대, 최소를 셈한다.
- **가짜 나무**: 물음에 뜻있는 마디와 그 LCA만 담은 옥죈 나무를 짓는다([가짜 나무](virtual_tree.md) 참고).
- **무거움-가벼움 쪼개기**: LCA가 길 물음에서 어느 사슬을 갈아탈지 정한다([HLD](hld.md) 참고).

---

## 연습문제

**연습문제 1.**
뿌리 있는 나무에서 LCA로 두 마디 $u$와 $v$ 사이의 거리를 셈하는 길을 보여라. 깊이와 LCA로 그 식을 나타내어라.

??? success "연습문제 1 풀이"
    $u$에서 $v$까지의 하나뿐인 길은 $\text{LCA}(u, v)$을 지난다. $u$에서 $\text{LCA}(u, v)$까지 올라간 뒤 $v$로 내려간다. 거리는 다음과 같다:

    $$
    d(u, v) = \text{depth}(u) + \text{depth}(v) - 2 \cdot \text{depth}(\text{LCA}(u, v))
    $$

    $\text{depth}(u) - \text{depth}(\text{LCA}(u,v))$이 $u$에서 LCA까지 올라가는 변의 수이고 $\text{depth}(v) - \text{depth}(\text{LCA}(u,v))$이 LCA에서 $v$로 내려가는 변의 수이므로 이것이 성립한다. 그 합이 전체 길 길이다. $\square$

---

**연습문제 2.**
LCA를 위한 두 갈래 들어 올리기 알고리즘을 밝혀라. 미리 다듬기의 때와 자리는 얼마이고 물음 때는 얼마인가?

??? success "연습문제 2 풀이"
    두 갈래 들어 올리기는 마디 $v$와 거듭제곱 $j$마다 $2^j$번째 조상 $\text{up}[v][j]$을 미리 셈한다. 밑동: $\text{up}[v][0] = \text{parent}(v)$. 점화식: $\text{up}[v][j] = \text{up}[\text{up}[v][j-1]][j-1]$. 미리 다듬기: 때와 자리가 $O(n \log n)$이다($j$가 $\lfloor \log_2 n \rfloor$까지). 물음 $\text{LCA}(u, v)$은 먼저 깊이 차이의 두 갈래 표현을 써서 더 깊은 마디를 뛰어올려 $u$와 $v$를 같은 깊이로 맞춘다. 그런 다음 $u = v$이면 $u$를 돌려준다. 아니면 $j$를 $\lfloor \log_2 n \rfloor$에서 0까지 줄이며 $\text{up}[u][j] \ne \text{up}[v][j]$이면 $u = \text{up}[u][j]$, $v = \text{up}[v][j]$로 둔다. 되돌이 뒤 $\text{LCA}(u, v) = \text{parent}(u)$이다. 물음 때: $O(\log n)$. $\square$

---

**연습문제 3.**
두 갈래 들어 올리기 LCA 물음이 옳음을 증명하여라. 곧 깊이를 맞추고 두 갈래로 내려간 뒤 $\text{parent}(u)$이 정말 LCA임을 보여라.

??? success "연습문제 3 풀이"
    깊이를 맞추면 $u$와 $v$는 같은 깊이에 있다. $u = v$이면 그 자신이 LCA다. 아니면 두 갈래 내려가기가 가장 큰 뜀 $j = \lfloor \log_2 n \rfloor$에서 0까지 되풀이한다. 걸음마다 $\text{up}[u][j] \ne \text{up}[v][j]$이면 두 마디를 $2^j$만큼 올린다. 불변량은 늘 LCA가 $u$와 $v$보다 엄격히 위에 있다는 것이다(곧 $u$와 $v$가 LCA의 서로 다른 밑나무에 있다). 모든 $j$를 다루고 나면 $u$와 $v$는 깊이 $\text{depth}(\text{LCA}) + 1$, 곧 LCA 바로 아래 한 걸음에 있다. 뜀들이 모두 $\sum_{j : \text{jumped}} 2^j$켜를 건너뛰는데, 두 갈래 표현에 따라 이 합이 정확히 $\text{depth}(u_0) - \text{depth}(\text{LCA}) - 1$과 같기 때문이다. 여기서 $u_0$은 깊이를 맞춘 처음 마디다. 따라서 $\text{parent}(u) = \text{LCA}(u, v)$이다. $\square$

---

**연습문제 4.**
변에 무게가 있는 나무가 있다. 두 갈래 들어 올리기로 $u$에서 $v$까지 길의 최대 변 무게를 찾는, 미리 다듬기 $O(n \log n)$, 물음 $O(\log n)$인 알고리즘을 설계하여라.

??? success "연습문제 4 풀이"
    두 갈래 들어 올리기 표에 둘째 배열을 더한다. $\text{maxw}[v][j]$은 $v$에서 그 $2^j$번째 조상까지 길의 최대 변 무게를 갈무리한다. 밑동: $\text{maxw}[v][0] = w(v, \text{parent}(v))$. 점화식: $\text{maxw}[v][j] = \max(\text{maxw}[v][j-1], \text{maxw}[\text{up}[v][j-1]][j-1])$. 미리 다듬기: $O(n \log n)$. 물음에서는 깊이 맞추기와 두 갈래 내려가기 동안 뜀에서 만난 모든 $\text{maxw}$ 값의 흐르는 최댓값을 좇는다. 곧 $u$를 $2^j$만큼 올릴 때마다 $u = \text{up}[u][j]$로 두기 전에 $\text{ans} = \max(\text{ans}, \text{maxw}[u][j])$로 고친다. 마지막 답은 쌓인 최댓값이다. 물음 때: $O(\log n)$. $\square$

---

**연습 5.**
LCA 알고리즘 셋, 곧 어설픈 기어오르기, 두 갈래 들어 올리기, 오일러 돌기와 RMQ를 미리 다듬기 때, 물음 때, 자리로 견주어라. 저마다 언제 쓰겠는가?

??? success "연습 5의 풀이"
    | 방법 | 미리 다듬기 | 물음 | 자리 |
    |---|---|---|---|
    | 어설픈 기어오르기 | $O(n)$ | 가장 나쁠 때 $O(n)$ | $O(n)$ |
    | 두 갈래 들어 올리기 | $O(n \log n)$ | $O(\log n)$ | $O(n \log n)$ |
    | 오일러 돌기와 성긴 표 RMQ | $O(n \log n)$ | $O(1)$ | $O(n \log n)$ |
    | 오일러 돌기와 벤더-파라크콜턴 | $O(n)$ | $O(1)$ | $O(n)$ |

    $n$이 아주 작거나($n < 100$) 물음이 몇 개뿐일 때는 어설픈 기어오르기가 낫다. 문제가 조상 뛰어오르기(보기로 $k$번째 조상 찾기)나 길 최대/최소 물음도 요구하면 들어 올리기 표가 이런 것으로 자연스레 늘어나므로 두 갈래 들어 올리기가 낫다. 물음 수가 아주 많아 물음마다 $O(1)$이 뜻있고 LCA 말고 다른 길 모음 앎이 필요 없다면 오일러 돌기와 RMQ가 낫다. $\square$

## 정리하며

이 마당은 정의와 성질、두 갈래 들어 올리기、복잡도 분석、다른 길을 차례로 짚었다.

**참고 문헌**

음이 아닌 정수 $x$가 주어질 때 비트 셈만 써서 $x$의 가장 낮은 켜진 비트만 남기는 식을 적어라(곧 그 비트만 켜진 값을 만들어라). $x = 0$일 때 그 식은 무엇을 돌려주는가?
