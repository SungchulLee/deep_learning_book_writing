# 나무 위의 동적 짜기

나무는 저절로 되돌이다. 뿌리를 없애면 나무가 서로 얽히지 않는 아래 나무로 갈라진다. 이 짜임 덕분에 나무는 동적 짜기에 안성맞춤이며, 마디마다의 가장 좋은 풀이가 자식들의 가장 좋은 풀이를 아우른다. 나무 위의 동적 짜기는 마디를 **뒤 차례**(어버이보다 자식 먼저)로 처리해, 어버이의 상태를 셈하기 앞서 자식의 아래 문제가 모두 풀리도록 한다. 쓰임새는 최대 홀로서기 모임과 최소 꼭짓점 덮기 찾기부터 나무의 지름과 아래 나무 합 셈하기까지 이른다.

---

## 1. 일반 얼개

마디가 $n$개인 뿌리 있는 나무가 주어질 때 $dp[v]$을 $v$을 뿌리로 하는 아래 나무의 가장 좋은 값이라 정하자. 되돌이 관계식은 모든 자식의 결과를 모은다:

$$
dp[v] = f\bigl(dp[c_1], dp[c_2], \ldots, dp[c_k]\bigr)
$$

여기서 $c_1, \ldots, c_k$은 $v$의 자식이고 $f$은 문제마다 다르다.

**바탕 경우**: 잎 마디 $v$에 대해 $dp[v]$을 곧바로 정한다(흔히 0이나 1).

**돌아보기 차례**: 깊이 먼저 돌아보기의 뒤 차례가 $dp[v]$을 셈하기 앞서 $dp[c]$이 마련되어 있음을 보장한다.

---

## 2. 보기: 최대 홀로서기 모임

**홀로서기 모임**은 어느 둘도 이웃하지 않는 꼭짓점의 모임이다. 나무에서 최대 홀로서기 모임은 $O(n)$ 시간에 풀 수 있다.

마디마다 상태를 둘 둔다:

- $dp[v][0]$: $v$을 **뺀** 채, $v$의 아래 나무에서 최대 홀로서기 모임의 크기
- $dp[v][1]$: $v$을 **넣은** 채, $v$의 아래 나무에서 최대 홀로서기 모임의 크기

**되돌이**:

$$
dp[v][0] = \sum_{c \in \text{children}(v)} \max\bigl(dp[c][0],\; dp[c][1]\bigr)
$$

$$
dp[v][1] = 1 + \sum_{c \in \text{children}(v)} dp[c][0]
$$

$v$을 빼면 자식마다 넣거나 뺄 수 있다. $v$을 넣으면 어떤 자식도 넣을 수 없다(이웃 제약).

**답**: $\max\bigl(dp[\text{root}][0],\; dp[\text{root}][1]\bigr)$.

---

## 3. 보기: 최소 꼭짓점 덮기

**꼭짓점 덮기**는 모든 변이 적어도 한쪽 끝점을 그 안에 갖는 꼭짓점의 모임이다.

- $dp[v][0]$: $v$을 **뺀** 채, $v$의 아래 나무에서 최소 꼭짓점 덮기
- $dp[v][1]$: $v$을 **넣은** 채, $v$의 아래 나무에서 최소 꼭짓점 덮기

**되돌이**:

$$
dp[v][0] = \sum_{c \in \text{children}(v)} dp[c][1]
$$

$$
dp[v][1] = 1 + \sum_{c \in \text{children}(v)} \min\bigl(dp[c][0],\; dp[c][1]\bigr)
$$

$v$을 빼면 $v$에 닿는 변을 덮으려 모든 자식을 넣어야 한다. $v$을 넣으면 자식마다 넣거나 뺄 수 있다.

---

## 4. 보기: 나무의 지름

나무의 **지름**은 어느 두 마디 사이 가장 긴 길의 길이이다. $\text{depth}(v)$을 $v$에서 아래로 뻗는 가장 긴 길의 길이라 정하자.

$$
\text{depth}(v) = \begin{cases} 0 & \text{if } v \text{ is a leaf} \\ 1 + \max_{c \in \text{children}(v)} \text{depth}(c) & \text{otherwise} \end{cases}
$$

$v$을 지나는 지름은 $v$에서 아래로 뻗는 가장 긴 길 둘을 쓴다:

$$
\text{diameter through } v = \text{depth}_1(v) + \text{depth}_2(v)
$$

여기서 $\text{depth}_1$과 $\text{depth}_2$은 자식의 깊이 가운데 가장 큰 둘이다. 전체 지름은 모든 마디에 걸친 최댓값이다.

---

## 5. 구현

```python
"""
나무 위의 동적 짜기: 최대 홀로서기 모임, 최소 꼭짓점 덮기, 나무의 지름.
"""

from collections import defaultdict

# ===================================================================
# 변으로 이웃 목록을 세운다
# ===================================================================
def build_tree(n: int, edges: list[tuple[int, int]], root: int = 0):
    """방향 없는 변으로 뿌리 있는 나무를 세운다.

    매개변수
    ----------
    n : int
        마디의 수.
    edges : list[tuple[int, int]]
        방향 없는 변.
    root : int
        뿌리 마디.

    반환값
    -------
    tuple
        이웃 목록과 어버이 배열.
    """
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    children = defaultdict(list)
    parent = [-1] * n
    visited = [False] * n
    stack = [root]
    order = []
    visited[root] = True

    while stack:
        node = stack.pop()
        order.append(node)
        for neighbor in adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                parent[neighbor] = node
                children[node].append(neighbor)
                stack.append(neighbor)

    return children, order

# ===================================================================
# 최대 홀로서기 모임
# ===================================================================
def max_independent_set(n: int, edges: list[tuple[int, int]]) -> int:
    """나무에서 최대 홀로서기 모임의 크기를 찾는다.

    매개변수
    ----------
    n : int
        마디의 수.
    edges : list[tuple[int, int]]
        나무의 변.

    반환값
    -------
    int
        최대 홀로서기 모임의 크기.
    """
    children, order = build_tree(n, edges)

    dp = [[0, 0] for _ in range(n)]

    # 거꾸로 된 차례(뒤 차례)로 처리한다
    for v in reversed(order):
        dp[v][1] = 1
        for c in children[v]:
            dp[v][0] += max(dp[c][0], dp[c][1])
            dp[v][1] += dp[c][0]

    return max(dp[0][0], dp[0][1])

# ===================================================================
# 최소 꼭짓점 덮기
# ===================================================================
def min_vertex_cover(n: int, edges: list[tuple[int, int]]) -> int:
    """나무에서 최소 꼭짓점 덮기의 크기를 찾는다.

    매개변수
    ----------
    n : int
        마디의 수.
    edges : list[tuple[int, int]]
        나무의 변.

    반환값
    -------
    int
        최소 꼭짓점 덮기의 크기.
    """
    children, order = build_tree(n, edges)

    dp = [[0, 0] for _ in range(n)]

    for v in reversed(order):
        dp[v][1] = 1
        for c in children[v]:
            dp[v][0] += dp[c][1]
            dp[v][1] += min(dp[c][0], dp[c][1])

    return min(dp[0][0], dp[0][1])

# ===================================================================
# 나무의 지름
# ===================================================================
def tree_diameter(n: int, edges: list[tuple[int, int]]) -> int:
    """나무의 지름을 찾는다.

    매개변수
    ----------
    n : int
        마디의 수.
    edges : list[tuple[int, int]]
        나무의 변.

    반환값
    -------
    int
        가장 긴 길의 길이(변의 수).
    """
    children, order = build_tree(n, edges)

    depth = [0] * n
    diameter = 0

    for v in reversed(order):
        top_two = [0, 0]
        for c in children[v]:
            d = depth[c] + 1
            if d > top_two[0]:
                top_two[1] = top_two[0]
                top_two[0] = d
            elif d > top_two[1]:
                top_two[1] = d
        depth[v] = top_two[0]
        diameter = max(diameter, top_two[0] + top_two[1])

    return diameter

# ===================================================================
# 메인
# ===================================================================
if __name__ == "__main__":
    #       0
    #      / \
    #     1   2
    #    / \   \
    #   3   4   5
    #       |
    #       6
    n = 7
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (4, 6)]

    print(f"Max independent set: {max_independent_set(n, edges)}")
    print(f"Min vertex cover: {min_vertex_cover(n, edges)}")
    print(f"Tree diameter: {tree_diameter(n, edges)}")
```

**출력:**
```
Max independent set: 5
Min vertex cover: 2
Tree diameter: 4
```

---

## 6. 복잡도

세 보기 모두 나무를 뒤 차례로 한 번만 훑으며 $O(n)$ 시간과 $O(n)$ 공간에 돈다.

| 문제 | 시간 | 공간 | 마디마다의 상태 수 |
|---------|------|-------|----------------|
| 최대 홀로서기 모임 | $O(n)$ | $O(n)$ | 2 |
| 최소 꼭짓점 덮기 | $O(n)$ | $O(n)$ | 2 |
| 나무의 지름 | $O(n)$ | $O(n)$ | 1 |

---

## 7. 뿌리 옮기기 재주

어떤 나무 동적 짜기 문제는 **마디마다 그것을 뿌리로 삼아** 답을 셈해야 한다(예컨대 "마디 $v$마다 $v$에서 가장 먼 마디를 찾아라"). 막무가내로 뿌리를 옮겨 가며 동적 짜기를 다시 돌리면 $O(n^2)$이 든다. **뿌리 옮기기 재주**는 $n$개 답을 모두 합해 $O(n)$에 셈한다:

1. 아무 뿌리로나 동적 짜기를 한 번 돌려 모든 $v$의 $dp[v]$을 얻는다
2. 두 번째 깊이 먼저 돌아보기에서 어버이의 $dp$ 값에서 $v$의 몫을 뺀 것으로 $dp_{\text{up}}[v]$($v$ 위쪽 아래 나무의 몫)을 셈한다
3. 마디마다 $dp[v]$과 $dp_{\text{up}}[v]$을 아우른다

!!! tip "언제 뿌리 옮기기를 쓰는가"
    어느 마디를 뿌리로 삼느냐에 따라 달라지는 값을 묻고, 동적 짜기가 자식의 값을 **되돌릴 수 있는** 연산(합, 곱, 둘째 최댓값까지 좇는 최댓값)으로 아우를 때 뿌리 옮기기를 쓴다.

---

## 연습문제

**연습문제 1.**
나무 위의 동적 짜기의 상태, 옮아감, 바탕 경우를 가려내어라.

??? success "연습문제 1 풀이"
    **상태**는 아래 문제를 적는 데 필요한 앎을 담는다. **옮아감**(되돌이 관계식)은 어떤 상태의 가장 좋은 값을 더 작은 상태로 나타낸다. **바탕 경우**는 곧바로 풀 수 있는 가장 작은 아래 문제의 값을 준다. 이 셋이 함께 동적 짜기 풀이를 온전히 정한다. $\square$

---

**연습문제 2.**
나무 위의 동적 짜기의 위에서 아래로(적어 두기) 짜기와 아래에서 위로(표 채우기) 짜기를 견주어라. 어느 쪽이 나으며 왜 그런가?

??? success "연습문제 2 풀이"
    **위에서 아래로**: 곳간을 곁들인 되돌이. 정말 필요한 아래 문제만 셈한다(게으른 값매김). 되돌이 관계식에서 옮겨 적기 쉽다. 되돌이 깊이 문제가 생길 수 있다. **아래에서 위로**: 되풀이로 기댐 차례에 따라 표를 채운다. 필요 없는 것까지 모든 아래 문제를 셈한다. 되돌이 군더더기가 없다. 공간을 줄이기 쉽다. 이 문제에서는 아래 문제가 모두 필요하면 아래에서 위로가 흔히 낫고, 닿지 않는 아래 문제가 많으면 위에서 아래로가 낫다. $\square$

---

**연습문제 3.**
나무 위의 동적 짜기의 시간 복잡도와 공간 복잡도는 무엇인가? 공간을 더 줄일 수 있는가?

??? success "연습문제 3 풀이"
    시간 복잡도는 상태의 수에 상태마다의 옮아감 값을 곱한 것으로 정해진다. 공간은 담아 두는 상태의 수와 같다. 옮아감이 앞선 상태 가운데 한정된 몇 개에만 기대면(예컨대 2차원 표의 바로 앞 가로줄) 그 상태만 기억 공간에 두어 공간을 줄일 수 있으며, 흔히 $O(n^2)$에서 $O(n)$으로 줄어든다. $\square$

---

**연습문제 4.**
나무 위의 동적 짜기의 알고리즘을 네가 고른 작은 보기에 대해 좇아라. 동적 짜기 표의 값을 보여라.

??? success "연습문제 4 풀이"
    작은 들임(예컨대 $n = 5$이나 짧은 글줄/배열)을 골라라. 동적 짜기 표를 한 걸음씩 채우면서 각 칸이 앞서 셈한 칸에서 어떻게 나오는지 보여라. 마지막 답을 막무가내로 다 세어 본 것과 견주어 확인하라. 이렇게 좇아 보면 되돌이 관계식이 옳음을 확인하고 알고리즘에 대한 직관이 선다. $\square$

## 정리하며

이 마당은 일반 얼개、보기: 최대 홀로서기 모임、보기: 최소 꼭짓점 덮기、보기: 나무의 지름을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 15. MIT Press.
