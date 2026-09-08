# 최단 경로의 성질

최단 경로 알고리즘은 모두 최단 경로가 어떻게 굴러가는지를 다스리는 몇 안 되는 짜임 성질에 기댄다. 이 성질들은 늦추기 기반 알고리즘이 *왜* 모이는지, 그리고 최단 경로가 *언제* 있음이 보장되는지를 알려 준다. 이를 몸에 익히면 데이크스트라, 벨먼-포드, DAG 최단 경로의 맞음 증명이 거의 기계적으로 된다.

---

## 1. 가장 좋은 밑짜임

최단 경로의 가장 근본적인 성질은 최단 경로의 부분 길마다 그 자체로 최단 경로라는 것이다.

!!! note "최단 경로의 가장 좋은 밑짜임"
    무게 함수 $w$을 갖는 무게 있는 방향 그래프 $G = (V, E)$에서 $p = \langle v_0, v_1, \dots, v_k \rangle$을 $v_0$에서 $v_k$까지의 최단 경로라 하자. $0 \le i \le j \le k$인 아무 $i, j$에 대해 부분 길 $p_{ij} = \langle v_i, v_{i+1}, \dots, v_j \rangle$은 $v_i$에서 $v_j$까지의 최단 경로이다.

**어긋냄으로 증명.** $w(p'_{ij}) < w(p_{ij})$인 $v_i$에서 $v_j$까지의 길 $p'_{ij}$이 있다고 하자. 그러면 $p$에서 $p_{ij}$을 $p'_{ij}$으로 바꾸면 무게가 $w(p)$보다 반드시 작은 $v_0$에서 $v_k$까지의 길이 나와, $p$이 최단 경로라는 가정과 어긋난다.
$\square$

이 성질 덕분에 동적 계획법과 욕심 방식으로 최단 경로를 셈할 수 있다. 곧 가장 좋은 풀이가 가장 좋은 부분 풀이로 쪼개진다.

---

## 2. 최단 경로 무게

$\delta(u, v)$을 $u$에서 $v$까지의 **최단 경로 짐**이라 하자.

$$
\delta(u, v) =
\begin{cases}
\min\{w(p) : u \xrightarrow{p} v\} & \text{if a path from } u \text{ to } v \text{ exists} \\
\infty & \text{otherwise}
\end{cases}
$$

$v$으로 가는 길 위에서 $u$로부터 무게가 음인 고리에 닿을 수 있으면, 고리를 얼마든지 돌 수 있으므로 최단 경로 무게를 $\delta(u, v) = -\infty$으로 정한다.

---

## 3. 삼각 부등식

For any edge $(u, v) \in E$:

$$
\delta(s, v) \le \delta(s, u) + w(u, v)
$$

$v$까지의 최단 경로는 $u$까지의 최단 경로에 곧바른 변 $(u, v)$을 이은 것보다 길 수 없다. 그렇다면 그 이어 붙인 것이 $v$까지의 더 짧은 길이 되어 $\delta$의 정의와 어긋난다.

---

## 4. 위 한계 성질

`INITIALIZE-SINGLE-SOURCE(G, s)`을 부른 뒤 모든 꼭짓점 $v \in V$에 대해

$$
d[v] \ge \delta(s, v)
$$

이 불변량은 늦추기 기반 알고리즘이 도는 내내 지켜지며, 한번 $d[v] = \delta(s, v)$이 되면 그 값은 결코 바뀌지 않는다.

**증명.** 처음에 $d[s] = 0 = \delta(s, s)$이고 $v \ne s$이면 $d[v] = \infty \ge \delta(s, v)$이다. $\textsc{Relax}(u, v, w)$을 부를 때마다 새 값이 더 작을 때만 $d[v] \leftarrow d[u] + w(u, v)$으로 놓는다. 귀납으로, 늦추기 전에 $d[u] \ge \delta(s, u)$이면 다음이 성립한다

$$
d[v] = d[u] + w(u, v) \ge \delta(s, u) + w(u, v) \ge \delta(s, v)
$$

여기서 마지막 걸음은 삼각 부등식에서 따라 나온다. $\square$

---

## 5. 길 없음 성질

$s$에서 $v$까지 길이 없으면 $\delta(s, v) = \infty$이다. 위 한계 성질에 따라 $d[v] \ge \delta(s, v) = \infty$이므로 알고리즘 내내 $d[v] = \infty$이다. 곧 알고리즘이 닿을 수 없는 꼭짓점을 따로 다루지 않고도 자연스럽게 처리한다는 뜻이다.

---

## 6. 모임 성질

$s \leadsto u \to v$이 최단 경로이고 변 $(u, v)$을 늦추기 전 어느 때든 $d[u] = \delta(s, u)$이면, 늦춘 뒤 $d[v] = \delta(s, v)$이고 그 뒤로 이 값은 바뀌지 않는다.

이 성질이 데이크스트라 알고리즘을 굴리는 엔진이다. 곧 꼭짓점 $u$을 $d[u] = \delta(s, u)$으로 우선순위 줄에서 꺼내면, (무게가 음이 아니라면) $u$에서 나가는 변을 모두 늦춰 이웃마다 거리가 맞게 정해진다.

---

## 7. 길 늦추기 성질

!!! note "길 늦추기 성질"
    $p = \langle v_0, v_1, \dots, v_k \rangle$을 다음에서 비롯하는 최단 경로라 하자
    $s = v_0$에서 $v_k$까지. 변 $(v_0, v_1), (v_1, v_2), \dots, (v_{k-1}, v_k)$을 이 차례로 늦추면(사이에 다른 늦추기가 섞여도 된다) $d[v_k] = \delta(s, v_k)$이다.

이 성질이 한 근원 알고리즘 셋의 옳음을 하나로 묶는다.

- **벨먼-포드:** $i$번째 훑기 뒤에 변을 많아야 $i$개 쓰는 최단 경로가 모두 맞게 셈해진다. 최단 경로의 변이 많아야 $|V| - 1$개이므로 $|V| - 1$번 훑으면 넉넉하다.
- **DAG 최단 경로:** 위상 차례가 $d[u]$이 $\delta(s, u)$에 이른 뒤에 변 $(u, v)$이 늦춰짐을 보장한다.
- **데이크스트라:** 욕심껏 꺼내는 차례가 $u$을 다룰 때 $d[u] = \delta(s, u)$임을 보장하므로, $u$에서 나가는 변을 늦추면 거리가 맞게 퍼진다.

---

## 8. 앞선 것 부분 그래프 성질

앞선 것 가리개 $\pi[v]$은 다음을 만족하는 **앞선 것 부분 그래프** $G_\pi = (V_\pi, E_\pi)$을 정한다:

$$
V_\pi = \{s\} \cup \{v \in V : \pi[v] \ne \text{NIL}\}
$$

$$
E_\pi = \{(\pi[v], v) : v \in V_\pi \setminus \{s\}\}
$$

최단 경로 알고리즘이 멈춘 뒤 $G_\pi$은 **최단 경로 나무**이다. 곧 $s$을 뿌리로 하는 나무이며, $G_\pi$에서 $s$부터 닿을 수 있는 꼭짓점 $v$까지의 하나뿐인 길이 $G$에서의 최단 경로이다.

---

## 9. 성질 간추림

| 성질 | 진술 | 핵심 결과 |
|---|---|---|
| 가장 좋은 밑짜임 | 최단 경로의 부분 길은 최단 경로이다 | 동적 계획법과 욕심 방식을 쓸 수 있게 한다 |
| 삼각 부등식 | $\delta(s,v) \le \delta(s,u) + w(u,v)$ | 늦추기가 안전하다 |
| 위 한계 | 늘 $d[v] \ge \delta(s,v)$ | 어림값은 나아지기만 한다 |
| 길 없음 | 닿을 수 없음 $\Rightarrow$ 늘 $d[v] = \infty$ | 따로 다룰 필요가 없다 |
| 모임 | 맞는 $d[u]$ + $(u,v)$ 늦추기 $\Rightarrow$ 맞는 $d[v]$ | 데이크스트라를 굴린다 |
| 길 늦추기 | 최단 경로 차례로 변을 늦춤 $\Rightarrow$ 맞음 | 모든 알고리즘을 하나로 꿴다 |
| 앞선 것 부분 그래프 | $G_\pi$은 최단 경로 나무이다 | 길 되살리기 |

---

## 10. 구현

```python
"""
최단 경로 성질 보이기.

핵심 불변식(위 경계, 삼각 부등식,
모임)을 확인하려고 걸음마다 늦추기를 돌리며
성질을 살핀다.
"""

from math import inf

# === 그래프 차리기 ===========================================================

def build_graph():
    """보기로 든 무게 방향 그래프를 이웃 목록으로 돌려주기.

    그래프:
        s --10--> a --1--> c
        s --5---> b --3--> a
                  b --8--> c
    """
    return {
        "s": [("a", 10), ("b", 5)],
        "a": [("c", 1)],
        "b": [("a", 3), ("c", 8)],
        "c": [],
    }

# === 성질을 살피는 늦추기 ===================================================

def initialize(vertices, source):
    """거리 어림값과 앞선 꼭짓점 첫값 잡기."""
    dist = {v: inf for v in vertices}
    dist[source] = 0
    pred = {v: None for v in vertices}
    return dist, pred

def relax(u, v, w, dist, pred):
    """변 (u, v)을 늦추고 새로 고쳐졌는지 돌려주기."""
    if dist[u] + w < dist[v]:
        dist[v] = dist[u] + w
        pred[v] = u
        return True
    return False

def verify_upper_bound(dist, true_dist):
    """모든 꼭짓점에 대해 d[v] >= delta(s,v)인지 살피기."""
    for v in dist:
        assert dist[v] >= true_dist[v], (
            f"Upper-bound violated: d[{v}]={dist[v]} < delta={true_dist[v]}"
        )
    return True

def verify_triangle_inequality(true_dist, graph):
    """모든 변에 대해 delta(s,v) <= delta(s,u) + w(u,v)인지 살피기."""
    for u in graph:
        for v, w in graph[u]:
            assert true_dist[v] <= true_dist[u] + w, (
                f"Triangle inequality violated for edge ({u},{v})"
            )
    return True

# === 주 실행 =================================================================

if __name__ == "__main__":
    graph = build_graph()
    vertices = list(graph.keys())

    # 참 최단 거리(미리 셈함)
    true_dist = {"s": 0, "a": 8, "b": 5, "c": 9}

    # 참 거리에서 삼각 부등식 확인하기
    assert verify_triangle_inequality(true_dist, graph)
    print("Triangle inequality: VERIFIED")

    # 늦추기를 돌리고 걸음마다 위 경계 살피기
    dist, pred = initialize(vertices, "s")
    edges = [("s", "a", 10), ("s", "b", 5), ("b", "a", 3),
             ("a", "c", 1), ("b", "c", 8)]

    for u, v, w in edges:
        relax(u, v, w, dist, pred)
        assert verify_upper_bound(dist, true_dist)
        print(f"After relaxing ({u},{v}): d = {dict(dist)}  "
              f"Upper-bound: VERIFIED")

    # 모임 살피기: 마지막 거리가 참 거리와 맞는가
    assert dist == true_dist
    print(f"\nConvergence: VERIFIED — final distances match true shortest paths")
```

**출력:**

```
Triangle inequality: VERIFIED
After relaxing (s,a): d = {'s': 0, 'a': 10, 'b': inf, 'c': inf}  Upper-bound: VERIFIED
After relaxing (s,b): d = {'s': 0, 'a': 10, 'b': 5, 'c': inf}  Upper-bound: VERIFIED
After relaxing (b,a): d = {'s': 0, 'a': 8, 'b': 5, 'c': inf}  Upper-bound: VERIFIED
After relaxing (a,c): d = {'s': 0, 'a': 8, 'b': 5, 'c': 9}  Upper-bound: VERIFIED
After relaxing (b,c): d = {'s': 0, 'a': 8, 'b': 5, 'c': 9}  Upper-bound: VERIFIED

Convergence: VERIFIED — final distances match true shortest paths
```

---

## 연습문제

**연습문제 1.**
최단 경로의 삼각 부등식을 밝히고 증명하여라. 곧 아무 변 $(u, v)$에 대해 $\delta(s, v) \leq \delta(s, u) + w(u, v)$이다.

??? success "연습문제 1 풀이"
    $s$에서 $v$까지의 최단 경로는 $s$에서 $v$까지의 어떤 길보다도 길지 않다. 그런 길 하나는 최단 경로로 $s$에서 $u$까지 간 뒤(무게 $\delta(s, u)$) 변 $(u, v)$을 탄다(무게 $w(u, v)$). 그러므로 $\delta(s, v) \leq \delta(s, u) + w(u, v)$이다. 이는 모든 변에 성립하며 늦추기 연산의 바탕이 된다. 곧 $d[v] > d[u] + w(u, v)$이면 $d[v]$이 아직 가장 좋지 않다는 뜻이다. $\square$

---

**연습문제 2.**
가장 좋은 밑짜임 성질을 증명하여라. 곧 최단 경로의 부분 길은 그 자체로 최단 경로이다.

??? success "연습문제 2 풀이"
    $p = v_0 \to v_1 \to \cdots \to v_k$을 $v_0$에서 $v_k$까지의 최단 경로라 하자. 아무 부분 길 $p_{ij} = v_i \to \cdots \to v_j$($0 \leq i \leq j \leq k$)을 생각하자. 어긋냄을 위해 $v_i$에서 $v_j$까지 더 짧은 길 $p'_{ij}$이 있다고 하자. 그러면 $p$의 $p_{ij}$을 $p'_{ij}$으로 바꾸면 무게가 $w(p) - w(p_{ij}) + w(p'_{ij}) < w(p)$인 $v_0$에서 $v_k$까지의 길이 나와 $p$이 가장 좋다는 것과 어긋난다. $\square$

---

**연습문제 3.**
모임 성질을 설명하여라. 곧 한번 $d[v] = \delta(s, v)$이 되면 올바른 어떤 최단 경로 알고리즘에서도 $d[v]$ 값이 결코 바뀌지 않는다. 왜 그런가?

??? success "연습문제 3 풀이"
    늦추기 연산은 $d[v]$을 줄이기만 한다. 곧 $d[v] = \min(d[v], d[u] + w(u,v))$이다. $d[v] \geq \delta(s, v)$이 불변량으로 지켜지므로(위 한계 성질 — $d[v]$은 참된 최단 거리 아래로 결코 내려가지 않는다), $d[v]$이 한번 $\delta(s, v)$에 이르면 더 줄어들 수 없다. 그러므로 $d[v]$은 $\delta(s, v)$에 영영 머문다. $\square$

---

**연습문제 4.**
길 늦추기 성질을 증명하여라. 곧 최단 경로 $s = v_0, v_1, \ldots, v_k$을 따라 변을 $(v_0, v_1), (v_1, v_2), \ldots, (v_{k-1}, v_k)$의 차례로(반드시 잇달지 않아도 된다) 늦추면 $d[v_k] = \delta(s, v_k)$이다.

??? success "연습문제 4 풀이"
    $(v_0, v_1)$을 늦춘 뒤 $d[v_1] \leq d[v_0] + w(v_0, v_1) = 0 + w(v_0, v_1) = \delta(s, v_1)$이다. 위 한계 성질에 따라 $d[v_1] \geq \delta(s, v_1)$이므로 $d[v_1] = \delta(s, v_1)$이다. 귀납으로, $(v_{i-1}, v_i)$을 늦춘 뒤 $d[v_{i-1}] = \delta(s, v_{i-1})$이라고 놓자. 그러면 (가장 좋은 밑짜임에 따라) $d[v_i] \leq d[v_{i-1}] + w(v_{i-1}, v_i) = \delta(s, v_{i-1}) + w(v_{i-1}, v_i) = \delta(s, v_i)$이다. $d[v_i] \geq \delta(s, v_i)$과 합치면 $d[v_i] = \delta(s, v_i)$을 얻는다. $k$번 모두 늦춘 뒤 $d[v_k] = \delta(s, v_k)$이다. $\square$

## 정리하며

이 마당은 가장 좋은 밑짜임、최단 경로 무게、삼각 부등식、위 한계 성질을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to
  Algorithms* (4th ed.), Chapter 24: Single-Source Shortest Paths.
