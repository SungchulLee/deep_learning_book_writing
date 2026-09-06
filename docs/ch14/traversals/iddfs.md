# 되풀이 깊이 늘리기

BFS은 최단 경로를 찾지만 앞자락 전체를 기억 공간에 담으며, 그 크기는 $b$이 갈라짐 인자이고 $d$이 풀이의 깊이일 때 $O(b^d)$일 수 있다. DFS은 기억 공간을 $O(d)$만 쓰지만 아무리 깊은 가지로도 파고들어 얕은 풀이를 놓칠 수 있다. **되풀이 깊이 늘리기 깊이 우선 찾기(IDDFS)**는 둘의 좋은 점을 합친다. 곧 깊이를 제한한 DFS을 되풀이해 돌리되 한 바퀴마다 깊이 한도를 하나씩 늘려 목표를 찾을 때까지 이어 간다. 그러면 BFS의 가장 좋음과 DFS의 기억 공간 효율을 함께 얻는다.

## 깊이를 제한한 DFS

IDDFS의 밑돌은 정해진 깊이 한도 $\ell$에 이르면 살펴보기를 멈추는 DFS이다. 깊이 $\ell$에서 찾기는 더 넓히지 않고 되돌아온다. 그러면 알고리즘이 아무리 깊은 가지로 내려가는 일을 막는다.

## IDDFS 알고리즘

IDDFS은 목표를 찾거나 그래프 전체를 살펴볼 때까지 한도 $\ell = 0, 1, 2, \ldots$으로 깊이를 제한한 DFS을 부른다. 되풀이 $\ell$에서 깊이 $\leq \ell$인 꼭짓점을 모두 들른다. 목표에 닿는 첫 되풀이가 있을 수 있는 가장 작은 깊이를 쓰므로, IDDFS은 무게 없는 그래프에서 가장 좋은(최단) 경로를 찾는다.

## 시간 복잡도

얼핏 보면 일을 되풀이하는 것이 헤퍼 보인다. 그러나 마디를 넓히는 총 횟수에는 한계가 있다. 풀이가 깊이 $d$에 있고 꼭짓점마다 자식이 많아야 $b$개면 되풀이 $\ell$은 많아야 꼭짓점 $O(b^\ell)$개를 들른다. 모든 되풀이에 걸친 전체 일은 다음과 같다

$$
\sum_{\ell=0}^{d} O(b^\ell) = O(b^d)
$$

등비급수가 마지막 항에 눌리기 때문이다. 이는 시간에서 BFS과 맞먹으면서 되돌이 더미에 $O(d)$ 기억 공간만 쓴다.

!!! tip "짐이 작다"
    전체 일과 마지막 되풀이의 일의 비는 $\frac{b^{d+1} - 1}{(b-1) \cdot b^d} \approx \frac{b}{b-1}$이며 $b \geq 2$이면 많아야 2이다. 되풀이하는 일은 전체 셈을 많아야 두 배로 만든다.

## 공간 복잡도

IDDFS은 뿌리에서 가장 깊은 꼭짓점까지의 지금 길만 되돌이 더미에 담는다. 이는 $O(d)$ 기억 공간이 들어 DFS과 같고 BFS이 쓸 수 있는 $O(b^d)$보다 훨씬 적다.

## 성질

- **온전함:** IDDFS은 끝이 있는 그래프에서 온전하다(풀이가 있으면 찾는다).
- **가장 좋음:** 무게 없는 그래프에서 IDDFS은 깊이 $d + 1$의 꼭짓점을 보기 전에 깊이 $d$의 꼭짓점을 모두 살펴보므로 최단 경로를 찾는다.
- **시간:** $O(b^d)$이며 여기서 $d$은 가장 얕은 풀이의 깊이이다.
- **공간:** $O(d)$ — 가장 얕은 풀이의 깊이.

## 구현

```python
"""
되풀이 깊이 늘리기 깊이 우선 찾기(IDDFS).

깊이 한도를 늘려 가며 깊이를 제한한 DFS을 돌려 BFS의 가장 좋음과
DFS의 기억 공간 효율을 합친다.
"""

# === 깊이를 제한한 DFS ======================================================

def depth_limited_dfs(graph, source, target, limit):
    """주어진 깊이 한도에서 멈추는 DFS.

    매개변수
    ----------
    graph : dict[int, list[int]]
        이웃 목록.
    source : int
        시작 꼭짓점.
    target : int
        목표 꼭짓점.
    limit : int
        살펴볼 최대 깊이.

    반환값
    -------
    list[int] | None
        source에서 target까지의 경로, 한도 안에서 찾지 못하면 None.
    """
    if source == target:
        return [source]
    if limit <= 0:
        return None

    for neighbor in graph[source]:
        result = depth_limited_dfs(graph, neighbor, target, limit - 1)
        if result is not None:
            return [source] + result
    return None


# === IDDFS ==================================================================

def iddfs(graph, source, target, max_depth=100):
    """되풀이 깊이 늘리기 깊이 우선 찾기.

    매개변수
    ----------
    graph : dict[int, list[int]]
        이웃 목록.
    source : int
        시작 꼭짓점.
    target : int
        목표 꼭짓점.
    max_depth : int
        끝없는 되풀이를 막는 깊이의 위 한계.

    반환값
    -------
    list[int] | None
        source에서 target까지의 최단 경로, 닿을 수 없으면 None.
    """
    for depth_limit in range(max_depth + 1):
        result = depth_limited_dfs(graph, source, target, depth_limit)
        if result is not None:
            return result
    return None


# === 메인 =====================================================================

if __name__ == "__main__":
    graph = {
        0: [1, 2],
        1: [3, 4],
        2: [5, 6],
        3: [],
        4: [7],
        5: [],
        6: [],
        7: [],
    }

    for target in [0, 4, 7, 6]:
        path = iddfs(graph, 0, target)
        depth = len(path) - 1 if path else None
        print(f"IDDFS 0 -> {target}: path={path}, depth={depth}")
```

**출력:**
```
IDDFS 0 -> 0: path=[0], depth=0
IDDFS 0 -> 4: path=[0, 1, 4], depth=2
IDDFS 0 -> 7: path=[0, 1, 4, 7], depth=3
IDDFS 0 -> 6: path=[0, 2, 6], depth=2
```

경로마다 있을 수 있는 가장 작은 깊이를 가지므로, IDDFS이 무게 없는 그래프에서 가장 좋은 풀이를 찾음이 확인된다.

## IDDFS을 언제 쓰나

| 상황 | 고를 알고리즘 |
|---|---|
| 무게 없음, 목표 깊이를 모름, 기억 공간이 빠듯함 | IDDFS |
| 무게 없음, 기억 공간이 넉넉함 | BFS |
| 무게 있는 변 | 데이크스트라 또는 A* |
| 갈라짐이 좁고 풀이가 아주 깊음 | DFS으로 넉넉할 수 있다 |

BFS이 기억 공간을 다 써 버릴 텐데도 가장 좋음이 필요하다면 IDDFS이 표준 고름이다. 갈라짐 인자가 크고 기억 공간이 병목인 놀이 나무 찾기(이를테면 체스 엔진)에서 흔히 쓴다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 22. MIT Press.
- Korf, R. E. (1985). Depth-first iterative-deepening: An optimal admissible tree search. *Artificial Intelligence*, 27(1), 97-109.

## 연습문제

**연습문제 1.**
IDDFS은 깊이 한도마다 일을 되풀이한다. 넓힌 마디의 총 개수가 $O(b^d)$임을 보여라. 여기서 $b$은 갈라짐 인자, $d$은 풀이의 깊이이다.

??? success "연습문제 1 풀이"
    깊이 한도 $i$에서 DFS은 마디 $O(b^i)$개를 넓힌다. 모든 되풀이의 합은 $\sum_{i=0}^{d} b^i = \frac{b^{d+1} - 1}{b - 1} = O(b^d)$이다. 마지막 되풀이가 마디 $b^d$개를 넓혀 지배하고, 앞선 되풀이를 모두 합쳐도 마디 $\frac{b^d - 1}{b - 1} \approx \frac{b^d}{b - 1}$개를 넓힌다. $b \geq 2$이면 짐 인자는 $\frac{b}{b-1}$이며 $b = 2$일 때 많아야 2이고 $b$이 커질수록 1에 가까워진다. $\square$

---

**연습문제 2.**
갈라짐 인자 $b = 10$이고 풀이 깊이 $d = 5$인 나무에서 BFS, DFS, IDDFS의 기억 공간 씀씀이를 견주어라.

??? success "연습문제 2 풀이"

    - **BFS**: 깊이 $d$의 앞자락 전체를 담아 마디 $O(b^d) = O(10^5) = O(100{,}000)$개가 필요하다.
    - **DFS**: 지금 길만 담아 마디 $O(bd) = O(50)$개가 필요하다. 그러나 DFS은 가장 얕은 풀이를 못 찾을 수 있다.
    - **IDDFS**: 되풀이마다 DFS만큼의 기억 공간을 쓰므로 마디 $O(bd) = O(50)$개이다. IDDFS은 BFS처럼 가장 얕은 풀이를 찾음을 보장하면서 기억 공간은 DFS만큼만 쓴다.

    기억 공간이 빠듯하고 가장 좋음이 필요하다면 IDDFS이 뚜렷이 낫다. $\square$

---

**연습문제 3.**
갈라짐 인자가 끝이 있으면 IDDFS이 끝없는 그래프에서도 왜 온전한지(풀이가 있으면 반드시 찾는지) 설명하여라.

??? success "연습문제 3 풀이"
    깊이 $d^*$에 풀이가 있으면 IDDFS은 (되풀이마다 한도를 1씩 올리므로) 언젠가 깊이 한도 $d^*$에 이른다. 깊이 한도 $d^*$에서 깊이를 제한한 DFS은 길이 $d^*$ 이하의 모든 길을 살펴보며 여기에 풀이의 길이 들어 있다. 갈라짐 인자에 끝이 있으므로 깊이를 제한한 DFS마다 끝이 있는 시간에(마디 $O(b^{d^*})$개) 멈춘다. 그러므로 IDDFS은 끝이 있는 시간에 풀이를 찾음이 보장되어 온전하다. $\square$

---

**연습문제 4.**
깊이 문턱값 대신 값 문턱값으로 굴러가도록 IDDFS을 고쳐라. 이 판을 IDA*이라 한다. 핵심 생각을 밝혀라.

??? success "연습문제 4 풀이"
    IDA*(되풀이 깊이 늘리기 A*)는 깊이 한도를 $f$-값 문턱값으로 바꾼다. 여기서 $f(n) = g(n) + h(n)$이다($g$은 경로 값, $h$은 어림짐작 어림값). 문턱값을 $h(\text{start})$으로 시작한다. 되풀이마다 DFS을 돌리되 $f(n) > \text{threshold}$인 마디를 쳐 낸다. 목표를 찾지 못하면 지금 문턱값을 넘긴 $f$ 값 가운데 가장 작은 것을 새 문턱값으로 놓는다. 목표를 찾을 때까지 되풀이한다. IDDFS처럼 IDA*도 $O(bd)$ 기억 공간을 쓴다. 받아들일 만하고 한결같은 어림짐작을 쓰면 IDA*은 가장 좋은 풀이를 찾는다. 15-퍼즐 같은 조합 퍼즐의 표준 알고리즘이다. $\square$
