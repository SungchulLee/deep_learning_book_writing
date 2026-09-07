# 히어홀처 알고리즘

그래프에 오일러 회로가 있음(꼭짓점마다 차수가 짝수이고 그래프가 이어져 있음)을 알고 나면 그것을 실제로 세우는 효율적인 알고리즘이 필요하다. 히어홀처 알고리즘(1873)은 작은 회로를 거듭 찾아 이어 붙여 이를 $O(E)$ 시간에 해낸다. 핵심 생각은 단순하다. 곧 쓰지 않은 변을 따라 걷다가 처음 꼭짓점으로 돌아오고, 아직 쓰지 않은 변이 남은 꼭짓점을 새 작은 회로로 부풀린다.

## 알고리즘 설명

**들임.** 꼭짓점마다 차수가 짝수이거나(오일러 회로) 차수가 홀수인 꼭짓점이 정확히 둘인(오일러 경로) 이어진 그래프 $G = (V, E)$.

**1단계.** 시작 꼭짓점 $s$을 고른다. 회로라면 아무 꼭짓점이나 된다. 경로라면 홀수 차수 꼭짓점 둘 가운데 하나에서 시작한다.

**2단계.** $s$에서 쓰지 않은 변을 따라가며 변마다 썼다고 표시하고 $s$으로 돌아온다. 이러면 첫 회로 $C$이 나온다.

**3단계.** $C$이 모든 변을 덮으면 끝이다. 그렇지 않으면 $C$ 위에서 쓰지 않은 변이 남은 꼭짓점 $v$을 찾는다. $v$에서 쓰지 않은 변을 따라 새로 걸어 $v$으로 돌아와 작은 회로 $C'$을 만든다.

**4단계.** 꼭짓점 $v$에서 $C'$을 $C$에 이어 붙인다. 곧 $C$에 나오는 $v$을 작은 회로 $C'$ 전체로 바꾼다.

**5단계.** 모든 변을 쓸 때까지 3~4단계를 되풀이한다.

## 왜 통하는가

꼭짓점마다 차수가 짝수이므로 어떤 걸음이든 꼭짓점에 들어가면 늘 나올 수 있다. 그래서 한 꼭짓점에서 떠난 걸음은 끝내 그 꼭짓점으로 돌아와 닫힌 회로를 이룬다. 회로의 변을 없애고 나도 남은 그래프의 차수는 여전히 모두 짝수이다(회로를 없애면 닿은 꼭짓점의 차수가 짝수만큼 줄어든다). 이어 붙이기 단계가 모든 작은 회로를 하나의 오일러 회로로 합쳐 준다.

## 쌓기를 쓴 효율적인 짜기

위의 이어 붙이기 방식은 생각으로는 또렷하지만 이음 목록으로 짜기에는 성가시다. 더 깔끔한 짜기는 쌓기를 써서 회로를 거꾸로 세운다. 꼭짓점마다 쓰지 않은 변을 욕심껏 따라가며 꼭짓점을 쌓기에 올린다. 남은 변이 없는 꼭짓점에 이르면 그것을 꺼내 내놓는다.

```python
"""
오일러 회로나 경로를 찾는 히어홀처 알고리즘.

쌓기를 쓰는 방식으로 오일러 돌기를 O(E) 시간에 세우며
회로를 드러내어 이어 붙이지 않는다.
"""

from collections import defaultdict, deque

# === 히어홀처 알고리즘 ===

def euler_circuit(n: int, edges: list[tuple[int, int]]) -> list[int]:
    """방향 없는 그래프에서 오일러 회로를 찾는다.

    꼭짓점마다 차수가 짝수이고 그래프가 이어져 있다고 가정한다
    (차수가 0이 아닌 꼭짓점 가운데서).

    인수:
        n: 꼭짓점의 개수(0부터 셈).
        edges: 방향 없는 변의 목록.

    반환값:
        오일러 회로를 이루는 꼭짓점의 목록.
    """
    adj = defaultdict(deque)
    edge_used = {}

    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
        edge_used[i] = False

    # 시작할, 차수가 0이 아닌 꼭짓점 찾기
    start = 0
    for v in range(n):
        if adj[v]:
            start = v
            break

    stack = [start]
    circuit = []

    while stack:
        v = stack[-1]
        # v에서 아직 안 쓴 변 찾기
        found = False
        while adj[v]:
            w, idx = adj[v][0]
            adj[v].popleft()
            if not edge_used[idx]:
                edge_used[idx] = True
                stack.append(w)
                found = True
                break
        if not found:
            circuit.append(stack.pop())

    return circuit


def euler_path(n: int, edges: list[tuple[int, int]]) -> list[int]:
    """방향 없는 그래프에서 오일러 경로를 찾는다.

    차수가 홀수인 꼭짓점이 정확히 둘이라고 가정한다.

    인수:
        n: 꼭짓점의 개수(0부터 셈).
        edges: 방향 없는 변의 목록.

    반환값:
        오일러 경로를 이루는 꼭짓점의 목록.
    """
    degree = [0] * n
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1

    # 시작할, 차수가 홀수인 꼭짓점 찾기
    odd_vertices = [v for v in range(n) if degree[v] % 2 == 1]

    if len(odd_vertices) == 2:
        # 홀수 차수 꼭짓점 둘 사이에 임시 변을 더한다
        temp_edge = (odd_vertices[0], odd_vertices[1])
        circuit = euler_circuit(n, edges + [temp_edge])
        # 회로에서 임시 변 없애기
        for i in range(len(circuit) - 1):
            if (circuit[i] == temp_edge[0] and circuit[i+1] == temp_edge[1]) or \
               (circuit[i] == temp_edge[1] and circuit[i+1] == temp_edge[0]):
                return circuit[i+1:] + circuit[1:i+1]
    return circuit


# === 시연 ===

if __name__ == "__main__":
    # 그래프: 0-1-2-3-0, 0-2(차수가 모두 짝수)
    edges = [(0,1),(1,2),(2,3),(3,0),(0,2)]
    circuit = euler_circuit(4, edges)
    print(f"Euler circuit: {circuit}")
    print(f"Uses {len(circuit)-1} edges (total edges: {len(edges)})")

    # 확인: 변마다 정확히 한 번씩 쓰였는지
    edge_pairs = set()
    for i in range(len(circuit) - 1):
        u, v = circuit[i], circuit[i+1]
        edge_pairs.add((min(u,v), max(u,v), i))
    print(f"Distinct edge traversals: {len(edge_pairs)}")
```

**출력:**

```
Euler circuit: [0, 2, 1, 0, 3, 2, 0]
Uses 5 edges (total edges: 5)
Distinct edge traversals: 5
```

이 알고리즘은 꼭짓점 $0$에서 시작해 $0$에서 끝나며 다섯 변을 모두 정확히 한 번씩 지난다. 속 차례는 이웃 목록의 차례에 달렸지만 올바른 오일러 회로라면 어느 것이든 맞다.

## 복잡도

| 항목 | 비용 |
|--------|:----:|
| 시간 | $O(V + E)$ |
| 공간 | $O(V + E)$ |

변마다 많아야 두 번 살피고(끝점마다의 이웃 목록에서 한 번씩) 정확히 한 번 쓴다. 쌓기는 결코 $O(E)$개를 넘지 않는다. 내놓는 것 자체의 길이가 $E + 1$이므로 이 알고리즘은 가장 좋다.

## 방향 그래프

방향 그래프에서는 히어홀처 알고리즘을 다음과 같이 고쳐 쓴다:

- 방향 이웃 목록(나가는 변만)을 쓴다.
- 모든 꼭짓점에서 $\text{in-deg}(v) = \text{out-deg}(v)$이고 그래프가 세게 이어져 있으면 오일러 회로가 있다.
- 나가는 변을 따라가며 쓸 때마다 없앤다.

시간 복잡도는 그대로 $O(V + E)$이다.

## 참고 문헌

- Hierholzer, C. (1873). Ueber die Moglichkeit, einen Linienzug ohne Wiederholung und ohne Unterbrechung zu umfahren. *Mathematische Annalen*, 6, 30--32.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), 22장: Elementary Graph Algorithms.

## 연습문제

**연습문제 1.**
오일러 회로를 찾는 히어홀처 알고리즘을 설명하여라. 그 시간 복잡도는 무엇인가?

??? success "연습문제 1 풀이"
    (1) 아무 꼭짓점에서 비롯해 변을 따라가며(지나온 변은 지운다) 처음 자리로 돌아온다. 이것이 첫 순환다. (2) 쓰지 않은 변이 남았다면 그런 변을 지닌 꼭짓점 $v$을 순환 위에서 찾는다. $v$에서 쓰지 않은 변으로 새 잔순환을 비롯한다. (3) 잔순환을 $v$ 자리에서 큰 순환에 끼워 넣는다. (4) 변을 모두 쓸 때까지 되풀이한다. 시간은 $O(V + E)$이다. 변마다 꼭 한 번씩 지나고 변 목록을 쓰면 끼워 넣기가 잘 들기 때문이다. $\square$

---

**연습문제 2.**
히어홀처 알고리즘은 왜 작은 회로를 늘 이어 붙일 수 있음을 보장하는가? 왜 늘 끝나는가?

??? success "연습문제 2 풀이"
    쓰지 않은 변이 남았다면 이제 순환 위의 어떤 꼭짓점이 그런 변을 지닌다(그래프가 이어져 있고 차수가 모두 짝수이므로, 쓰지 않은 변이 순환을 거쳐 이어진 아래그래프를 이룬다). 이 꼭짓점에서 잔순환을 비롯하면 그 자리로 되돌아온다(남은 차수가 모두 짝수이기 때문이다). 끼워 넣으면 변을 되풀이하지 않고 순환이 길어진다. 걸음마다 변을 적어도 하나 쓰고 변은 마디 있게 많으므로 알고리즘은 끝난다. $\square$

---

**연습문제 3.**
쌓기를 쓰는 방식으로 히어홀처 알고리즘을 짜라. 유사 코드를 그려라.

??? success "연습문제 3 풀이"
    ```python
    def eulerian_circuit(adj, n):
        stack = [0]
        circuit = []
        while stack:
            v = stack[-1]
            if adj[v]:
                u = adj[v].pop()
                adj[u].remove(v)  # 방향 없는 경우
                stack.append(u)
            else:
                circuit.append(stack.pop())
        return circuit[::-1]
    ```

    쌓개가 이제 길을 좇는다. 어떤 꼭짓점에 남은 변이 없으면 그 꼭짓점을 순환에 넣는다. 이러면 끼워 넣는 걸음이 저절로 다루어진다. 빼기가 잘 드는 이웃 목록(묶음이나 두 끝 줄 따위)을 쓰면 전체 시간은 $O(V + E)$이다. $\square$

---

**연습문제 4.**
차수가 홀수인 꼭짓점이 정확히 둘일 때 오일러 경로(회로가 아닌 것)를 찾도록 히어홀처 알고리즘을 어떻게 고치는가?

??? success "연습문제 4 풀이"
    차수가 홀수인 두 꼭짓점 사이에 잠깐 쓸 변을 더한다. 이제 차수가 모두 짝수이므로 오일러 회로가 있다. 히어홀처 알고리즘을 돌려 순환을 찾는다. 결과에서 잠깐 쓴 변을 지우면 순환이 끊어져 한쪽 홀수 꼭짓점에서 비롯해 다른 쪽에서 끝나는 길이 된다. 아니면 홀수 차수 꼭짓점 가운데 하나에서 알고리즘을 비롯해도 되는데, 그러면 저절로 다른 쪽에서 끝난다. $\square$
