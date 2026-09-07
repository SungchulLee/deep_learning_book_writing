# 무거움-가벼움 쪼개기

무거움-가벼움 쪼개기(HLD)는 뿌리 있는 나무를 꼭짓점이 겹치지 않는 사슬로 나누어
뿌리에서 잎까지의 아무 길이 많아야 $O(\log n)$개의 사슬을 지나게 한다. 사슬 위에
토막 나무를 얹으면 길 물음과 고침에
$O(\log^2 n)$ 때에 답한다.

## 직관

뿌리 있는 나무에서 어떤 아이는 다른 아이보다 "무겁다"(밑나무가 크다).
지금 사슬을 늘 가장 무거운 아이로 늘리면 다른 사슬로 갈아탈 때 밑나무 크기가
적어도 반이 됨을 보장할 수 있다. 이로써 아무 길에서 사슬을 갈아타는 횟수가
$O(\log n)$으로 가둬진다.

## 정의

$T$를 마디 $n$개의 뿌리 있는 나무라 하자. 잎이 아닌 마디 $v$마다:

- $v$의 **무거운 아이**는 밑나무가 가장 큰 아이 $u$다:
  $v$의 모든 아이 $w$에 대해 $\text{size}(u) \ge \text{size}(w)$이다. 동점은 아무렇게나
  가른다.
- 무거운 아이로 가는 변 $(v, u)$은 **무거운 변**이고 다른 모든 아이 변은
  **가벼운 변**이다.
- 무거운 변으로 된 가장 긴 길이 **무거운 사슬**을 이룬다.
- 사슬의 **머리**는 가장 위의 마디(뿌리에 가장 가까운 마디)다.

**가벼운 변의 핵심 성질.** $(v, u)$이 가벼운 변이면

$$
\text{size}(u) \le \frac{\text{size}(v)}{2}
$$

이다. $u$가 가장 무거운 아이가 아니기 때문이다.

## 사슬 수 가둠

**정리.** 뿌리에서 잎까지의 아무 길은 많아야 $\lfloor \log_2 n \rfloor + 1$개의
사슬을 지난다.

??? note "증명"
    길이 $v$에서 아이 $u$로 가벼운 변을 지날 때마다 밑나무 크기가
    적어도 반이 된다: $\text{size}(u) \le \text{size}(v)/2$. $\text{size}(\text{root}) = n$에서
    시작해 크기 $1$인 잎에서 끝나므로 반이 되는 횟수는
    많아야 $\lfloor \log_2 n \rfloor$이다. 반이 될 때마다 새 사슬에 들어가므로
    사슬은 많아야 $\lfloor \log_2 n \rfloor + 1$개다.

## 알고리즘

### 1걸음 — 미리 다듬기

1. 나무에 뿌리를 잡고 깊이 먼저 훑기로 밑나무 크기를 셈한다.
2. 마디마다 무거운 아이를 가려낸다.
3. 둘째 깊이 먼저 훑기를 돌려 마디마다 납작 배열에서의 **자리** $\text{pos}[v]$을
   매긴다. 사슬 안에서는 자리가 이어진다.
4. $\text{head}[v]$, 곧 $v$가 든 사슬의 머리를 적는다.

### 2걸음 — 길 물음

$u$에서 $v$까지의 길을 물으려면:

1. $\text{head}[u] \ne \text{head}[v]$인 동안:
    - 사슬 머리가 더 깊은 쪽을 $u$라 한다(필요하면 맞바꾼다).
    - 토막 나무에 $[\text{pos}[\text{head}[u]],\; \text{pos}[u]]$을 묻는다.
    - $u$를 $\text{parent}[\text{head}[u]]$로 올린다.
2. 두 마디가 같은 사슬에 오면
   $[\min(\text{pos}[u], \text{pos}[v]),\; \max(\text{pos}[u], \text{pos}[v])]$을 묻는다.

되풀이마다 사슬 하나가 없어지고($O(\log n)$번 되풀이) 토막 나무 물음마다
$O(\log n)$이 들어 모두 $O(\log^2 n)$이다.

## 풀이 예제

```
        1
       /|\
      2  3  6
     / \
    4   5
```

밑나무 크기: $\text{size}(1)=6,\; \text{size}(2)=3,\; \text{size}(3)=1,\;
\text{size}(4)=1,\; \text{size}(5)=1,\; \text{size}(6)=1$이다.

무거운 아이: $1 \to 2$(크기 3 대 1), $2 \to 4$(동점, 4를 뽑음).

사슬: $[1, 2, 4]$, $[5]$, $[3]$, $[6]$.

길 물음 $5 \to 6$:

1. $\text{head}[5] = 5$, $\text{head}[6] = 6$. $5$를 $\text{parent}[5] = 2$로 올린다.
2. $\text{head}[2] = 1$, $\text{head}[6] = 6$. $6$을 $\text{parent}[6] = 1$로 올린다.
3. 이제 둘 다 사슬 $[1, 2, 4]$에 있다. $[\text{pos}[1], \text{pos}[2]]$을 묻는다.

모두: 토막 나무 물음 3번.

## 구현

```python
"""길 최댓값 물음을 하는 무거움-가벼움 쪼개기."""

import sys
from collections import defaultdict

# === 상수 ===
sys.setrecursionlimit(300_000)
INF = float("inf")


# === 무거움-가벼움 쪼개기 짓기 ===
class HLD:
    """뿌리 없는 나무의 무거움-가벼움 쪼개기."""

    def __init__(self, adj, root, n):
        self.n = n
        self.adj = adj
        self.root = root
        self.parent = [-1] * n
        self.depth = [0] * n
        self.size = [1] * n
        self.heavy = [-1] * n
        self.head = list(range(n))
        self.pos = [0] * n
        self._timer = 0

        self._compute_sizes()
        self._decompose()

    def _compute_sizes(self):
        """밑나무 크기와 무거운 아이를 셈하는 되풀이 깊이 먼저 훑기."""
        stack = [(self.root, -1, False)]
        order = []
        while stack:
            v, par, entered = stack.pop()
            if entered:
                for u in self.adj[v]:
                    if u != par:
                        self.size[v] += self.size[u]
                        if self.heavy[v] == -1 or self.size[u] > self.size[self.heavy[v]]:
                            self.heavy[v] = u
                continue
            self.parent[v] = par
            stack.append((v, par, True))
            order.append(v)
            for u in self.adj[v]:
                if u != par:
                    self.depth[u] = self.depth[v] + 1
                    stack.append((u, v, False))

    def _decompose(self):
        """사슬 머리와 납작 자리를 매긴다."""
        stack = [(self.root, self.root)]
        while stack:
            v, h = stack.pop()
            self.head[v] = h
            self.pos[v] = self._timer
            self._timer += 1
            # 가벼운 아이를 먼저 다룸(쌓기 차례에 맞춰 뒤집음)
            children = [u for u in self.adj[v] if u != self.parent[v]]
            for u in children:
                if u != self.heavy[v]:
                    stack.append((u, u))
            # 무거운 아이를 마지막에 넣어 먼저 다뤄지게 함(쌓기는 나중에 넣은 것이 먼저)
            if self.heavy[v] != -1:
                stack.append((self.heavy[v], h))

    def path_query(self, u, v, seg_query):
        """토막 나무 물음 함수로 u-v 길을 묻는다.

        seg_query(l, r)은 구간 [l, r]의 답을 돌려주어야 한다.
        """
        result = 0
        while self.head[u] != self.head[v]:
            if self.depth[self.head[u]] < self.depth[self.head[v]]:
                u, v = v, u
            result = max(result, seg_query(self.pos[self.head[u]], self.pos[u]))
            u = self.parent[self.head[u]]
        if self.depth[u] > self.depth[v]:
            u, v = v, u
        result = max(result, seg_query(self.pos[u], self.pos[v]))
        return result


# === 시연 ===
if __name__ == "__main__":
    n = 6
    adj = defaultdict(list)
    edges = [(0, 1), (0, 2), (0, 5), (1, 3), (1, 4)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    hld = HLD(adj, 0, n)
    print("pos: ", hld.pos)
    print("head:", hld.head)
    print("heavy:", hld.heavy)
```

## 복잡도 요약

| 연산 | 시간 | 공간 |
|-----------|------|-------|
| 미리 다듬기 | $O(n)$ | $O(n)$ |
| 길 물음 | $O(\log^2 n)$ | — |
| 길 고침 | $O(\log^2 n)$ | — |
| 밑나무 물음 | $O(\log n)$ | — |

## 참고 문헌

- Sleator, D. D. & Tarjan, R. E. (1983). *A Data Structure for Dynamic Trees*
음이 아닌 정수 $x$가 주어질 때 비트 셈만 써서 $x$의 가장 낮은 켜진 비트만 남기는 식을 적어라(곧 그 비트만 켜진 값을 만들어라). $x = 0$일 때 그 식은 무엇을 돌려주는가?

## 연습문제

**연습문제 1.**
마디 1을 뿌리로 하는 길 그래프 $1 - 2 - 3 - 4 - 5$가 주어질 때 무거운 변과 가벼운 변, 그리고 무거움-가벼움 쪼개기의 결과 사슬을 밝혀라.

??? success "연습문제 1 풀이"
    밑나무 크기: $\text{sz}(5) = 1$, $\text{sz}(4) = 2$, $\text{sz}(3) = 3$, $\text{sz}(2) = 4$, $\text{sz}(1) = 5$. 마디마다 무거운 아이는 밑나무가 가장 큰 아이다. 마디 1의 하나뿐인 아이는 2(무거움)다. 마디 2의 하나뿐인 아이는 3(무거움)이다. 마디 3의 하나뿐인 아이는 4(무거움)다. 마디 4의 하나뿐인 아이는 5(무거움)다. 모든 변이 무거우므로 길 전체가 사슬 하나 $[1, 2, 3, 4, 5]$를 이룬다. 사슬 머리는 마디 1이다. 아무 두 마디 사이의 길 물음은 이 사슬 하나만 지나므로 토막 나무 물음 한 번이면 되어 $O(\log^2 n)$이 아니라 $O(\log n)$이다. 이는 길에서 HLD가 가장 좋게 돌아감을 보여 준다. $\square$

---

**연습문제 2.**
마디 $n$개의 나무에서 뿌리에서 잎까지의 아무 길이 무거움-가벼움 쪼개기의 가벼운 변을 많아야 $O(\log n)$개 지남을 증명하여라.

??? success "연습문제 2 풀이"
    가벼운 변 $(u, v)$은 $u$를 무거운 아이가 아닌 아이 $v$와 잇는다. 뜻매김에 따라 $v$의 밑나무 크기는 $\text{sz}(v) \le \lfloor \text{sz}(u) / 2 \rfloor$을 만족한다(아니면 $v$가 무거운 아이일 것이다). 아래로 가벼운 변을 지날 때마다 밑나무 크기가 적어도 반이 된다. 밑나무 크기 $n$인 뿌리에서 시작해 가벼운 변 $k$개를 지난 뒤 밑나무 크기는 많아야 $n / 2^k$이다. 밑나무 크기가 적어도 1이므로 $n / 2^k \ge 1$이어야 하고 이는 $k \le \log_2 n$을 준다. 따라서 뿌리에서 잎까지의 아무 길은 가벼운 변을 많아야 $\lfloor \log_2 n \rfloor$개 지나며, 곧 사슬을 많아야 $\lfloor \log_2 n \rfloor + 1$개 지난다. $\square$

---

**연습문제 3.**
HLD와 토막 나무로 다음 물음에 $O(\log^2 n)$에 답하는 알고리즘을 설계하여라. "$u$에서 $v$까지 길에서 변 무게의 최댓값은 얼마인가?" 변을 마디에 어떻게 맞대응시키는지 밝혀라.

??? success "연습문제 3 풀이"
    변마다 그 아이 마디에 맞댄다. 곧 변 $(u, \text{parent}(u))$을 마디 $u$에 갈무리한다. 뿌리에는 딸린 변이 없다. HLD 차례로 늘어놓은 변 무게 배열 위에 토막 나무를 짓는다. $u$에서 $v$까지의 길 물음은 (1) $u$와 $v$가 다른 사슬에 있는 동안 사슬 머리에서 더 깊은 마디까지의 범위를 토막 나무에 묻고 그 마디를 사슬 머리의 어버이로 옮기며, (2) $u$와 $v$가 같은 사슬에 오면 둘 사이의 범위를 묻되 위쪽 마디를 뺀다(변이 아이에 갈무리되어 있고 LCA 마디의 변은 그 어버이의 것이기 때문이다). 사슬을 건널 때마다 토막 나무 물음 한 번에 $O(\log n)$이 들고 사슬 건너기가 $O(\log n)$번이므로 모두 $O(\log^2 n)$이다. $\square$

---

**연습문제 4.**
HLD와 토막 나무를 쓴 길 물음의 복잡도를 오일러 돌기와 펜윅 나무를 쓴 것과 견주어라. 어떤 갈래의 물음에 저마다 어느 쪽이 나은가?

??? success "연습문제 4 풀이"
    HLD와 토막 나무: 길 물음($u$에서 $v$까지 길의 합, 최대, 최소)이 $O(\log^2 n)$, 밑나무 물음이 $O(\log n)$(토막 나무 범위 하나)이다. 오일러 돌기와 펜윅 나무: 밑나무 물음(합)이 $O(\log n)$이고 점 고침도 $O(\log n)$이다. 길 물음은 LCA 셈과 들어감/나옴 차이 다루기가 필요한데 이는 합에는 되지만 최대나 최소에는 안 된다. 길 물음이 되돌릴 수 없는 셈(최대, 최소)을 쓸 때는 HLD가 낫다. 길을 이어진 사슬 토막 $O(\log n)$개로 쪼개기 때문이다. 밑나무 물음만 필요하면 사슬 쪼개기의 번거로움을 피하므로 오일러 돌기가 낫다. 길 합만 놓고 보면 둘 다 되지만 오일러 돌기와 펜윅 나무가 상수 인수가 더 작다(들어감-나옴 재주로 물음마다 $O(\log n)$).  $\square$

---

**연습 5.**
마디 $10^5$개에 정수 값이 있는 나무가 있다. 물음은 (가) $u$에서 $v$까지 길의 모든 마디 값에 $x$ 더하기와 (나) $w$의 밑나무에 든 마디 값의 합 묻기를 오간다. HLD로 두 셈을 모두 $O(\log^2 n)$에 받치는 자료 얼개를 설계하여라.

??? success "연습 5의 풀이"
    HLD로 나무를 쪼갠다. HLD 마디 차례 위에 게으른 퍼뜨리기를 하는 토막 나무를 짓는다. (가)는 $u$에서 $v$까지의 길을 사슬 토막 $O(\log n)$개로 쪼갠다. 토막마다 게으른 퍼뜨리기로 토막 나무에 범위 더하기 고침을 $O(\log n)$에 한다. 모두 $O(\log^2 n)$이다. (나)는 $w$의 밑나무가 HLD 차례에서 이어진 범위이므로(HLD가 쓰는 깊이 먼저 차례에서 밑나무 마디가 이어진 구간을 이루기 때문이다) 이 범위를 토막 나무에 $O(\log n)$에 묻는다. 핵심 깨침은 HLD의 깊이 먼저 차례가 길을 사슬로 쪼개는 것과 밑나무가 이어지는 것을 함께 받쳐, 게으른 퍼뜨리기를 하는 토막 나무 하나로 두 셈을 모두 다룰 수 있다는 점이다. $\square$
