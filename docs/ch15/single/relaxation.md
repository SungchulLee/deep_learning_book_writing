# 변 늦추기

최단 경로 알고리즘은 공통된 장치를 함께 쓴다. 곧 꼭짓점마다 잠정 거리 어림값을 지키고 변을 살피며 그 어림값을 되풀이해 낫게 만든다. **늦추기**라 부르는 이 연산이 데이크스트라 알고리즘, 벨먼-포드, DAG 최단 경로의 근본 밑돌이다. 늦추기를 따로 떼어 이해하면 알고리즘마다 왜 굴러가고 무엇이 다른지 또렷해진다.

## 첫걸음 잡기

늦추기를 하기 전에 꼭짓점마다 잠정 거리와 앞선 것 가리개를 받는다. 샘 꼭짓점은 거리 0으로, 나머지는 모두 무한대로 시작한다.

근원이 $s$인 그래프 $G = (V, E)$에 대해

$$
d[v] =
\begin{cases}
0 & \text{if } v = s \\
\infty & \text{otherwise}
\end{cases}
\quad \text{and} \quad
\pi[v] = \text{NIL} \quad \forall\, v \in V
$$

여기서 $d[v]$은 지금의 최단 경로 어림값이고 $\pi[v]$은 $s$에서 오는, 알려진 가장 좋은 길 위 $v$의 앞선 꼭짓점을 적는다.

## 늦추기 연산

무게가 $w(u, v)$인 변 $(u, v)$이 주어지면, 늦추기는 $u$을 지나는 길이 $v$의 지금 어림값을 낫게 하는지 살핀다. 길 $s \leadsto u \to v$이 지금까지 찾은 $v$까지의 가장 좋은 길보다 짧으면 어림값과 앞선 것을 새로 고친다.

$$
\textsc{Relax}(u, v, w): \quad
\text{if } d[v] > d[u] + w(u, v) \text{ then }
\begin{cases}
d[v] \leftarrow d[u] + w(u, v) \\
\pi[v] \leftarrow u
\end{cases}
$$

이 연산은 거리 어림값을 줄이기만 하고 결코 늘리지 않으므로 안전하다. 늦춘 뒤 변 $(u, v)$에 대해 불변량 $d[v] \le d[u] + w(u, v)$이 지켜진다.

## 늦추기가 왜 맞는가

늦추기는 최단 경로 알고리즘이 올바른 답으로 모임을 보장하는 핵심 성질 몇 가지를 지킨다.

### 위 한계 성질

언제나 $d[v] \ge \delta(s, v)$이며 여기서 $\delta(s, v)$은 참된 최단 경로 무게이다. 늦추기는 $d[v]$을 줄이기만 하고, 조건 $d[v] > d[u] + w(u, v)$이 새 값을 $d[u] + w(u, v)$과 같게 만드는데 이는 삼각 부등식에 따라 적어도 $\delta(s, v)$이다.

### 모임 성질

변 $(u, v)$을 늦출 때 $d[u] = \delta(s, u)$이면 늦춘 뒤 $d[v] \le \delta(s, v)$이다. 위 한계 성질과 합치면 $d[v] = \delta(s, v)$, 곧 어림값이 정확하다는 뜻이다.

### 길 늦추기 성질

$p = \langle v_0, v_1, \dots, v_k \rangle$이 $s = v_0$에서 $v_k$까지의 최단 경로이고 변 $(v_0, v_1), (v_1, v_2), \dots, (v_{k-1}, v_k)$을 이 차례로 늦추면(사이에 다른 늦추기가 섞여도 된다) $d[v_k] = \delta(s, v_k)$이다.

이 성질이 알고리즘마다 통하게 하는 까닭이다.

| 알고리즘 | 늦추는 차례를 어떻게 보장하나 |
|---|---|
| **데이크스트라** | 우선순위 줄에서 욕심껏 꺼내므로 $u$을 다룰 때 $d[u] = \delta(s, u)$이 보장된다 |
| **벨먼-포드** | 모든 변을 $\lvert V \rvert - 1$번 훑어 있을 수 있는 최단 경로 길이를 모두 덮는다 |
| **DAG 최단 경로** | 위상 차례가 앞선 것이 모두 먼저 확정됨을 보장한다 |

## 삼각 부등식

어떤 변 $(u, v) \in E$에 대해서도 최단 경로 거리는 다음을 채운다.

$$
\delta(s, v) \le \delta(s, u) + w(u, v)
$$

이것이 어긋나면 길 $s \leadsto u \to v$이 $\delta(s, v)$보다 짧아져 최단 경로 무게의 정의와 어긋난다. 늦추기는 이 부등식을 곧바로 써먹는다.

## 풀이 예제

꼭짓점 넷과 다음 변을 지닌 그래프를 보자.

| 변 | 무게 |
|---|---|
| $(s, a)$ | 10 |
| $(s, b)$ | 5 |
| $(b, a)$ | 3 |
| $(a, c)$ | 1 |
| $(b, c)$ | 8 |

**첫걸음 뒤:** $d[s] = 0$, $d[a] = \infty$, $d[b] = \infty$, $d[c] = \infty$.

**$(s, a)$ 늦추기:** $d[a] = \infty > 0 + 10 = 10$이므로 $d[a] \leftarrow 10$, $\pi[a] \leftarrow s$.

**$(s, b)$ 늦추기:** $d[b] = \infty > 0 + 5 = 5$이므로 $d[b] \leftarrow 5$, $\pi[b] \leftarrow s$.

**$(b, a)$ 늦추기:** $d[a] = 10 > 5 + 3 = 8$이므로 $d[a] \leftarrow 8$, $\pi[a] \leftarrow b$.

**$(a, c)$ 늦추기:** $d[c] = \infty > 8 + 1 = 9$이므로 $d[c] \leftarrow 9$, $\pi[c] \leftarrow a$.

**$(b, c)$ 늦추기:** $d[c] = 9 \not> 5 + 8 = 13$이므로 새로 고치지 않는다. 무게 9인 기존 길 $s \to b \to a \to c$이 이미 더 낫다.

## 구현

```python
"""
단일 근원 최단 경로를 위한 변 늦추기.

데이크스트라 알고리즘, 벨먼-포드, 유향 비순환 그래프 최단 경로의
핵심을 이루는 늦추기 연산을 보인다.
"""

from math import inf


# === 첫값 잡기 ==============================================================

def initialize_single_source(vertices: list, source) -> tuple[dict, dict]:
    """최단 경로 찾기를 위한 거리 어림값과 앞선 꼭짓점 차리기.

    매개변수
    ----------
    vertices : list
        그래프의 모든 꼭짓점 이름.
    source : hashable
        근원 꼭짓점.

    반환값
    -------
    dist : dict
        잠정 거리(근원은 0, 나머지는 모두 inf).
    pred : dict
        앞선 꼭짓점 가리개(처음에는 모두 None).
    """
    dist = {v: inf for v in vertices}
    dist[source] = 0
    pred = {v: None for v in vertices}
    return dist, pred


# === 늦추기 ==================================================================

def relax(u, v, weight: float, dist: dict, pred: dict) -> bool:
    """주어진 무게로 변 (u, v) 늦추기.

    u을 지나는 길이 v의 거리 어림값을 낫게 하면
    dist[v]과 pred[v]을 새로 고친다.

    어림값이 나아졌으면 True을, 아니면 False을 돌려준다.
    """
    if dist[u] + weight < dist[v]:
        dist[v] = dist[u] + weight
        pred[v] = u
        return True
    return False


# === 경로 되짚기 =============================================================

def reconstruct_path(pred: dict, source, target) -> list:
    """과녁에서 근원까지 앞선 꼭짓점 사슬 되짚기."""
    path = []
    current = target
    while current is not None:
        path.append(current)
        current = pred[current]
    path.reverse()
    if path[0] != source:
        return []  # 과녁에 닿을 수 없음
    return path


# === 보임 ====================================================================

if __name__ == "__main__":
    # 그래프: s -> a (10), s -> b (5), b -> a (3), a -> c (1), b -> c (8)
    vertices = ["s", "a", "b", "c"]
    edges = [("s", "a", 10), ("s", "b", 5), ("b", "a", 3),
             ("a", "c", 1), ("b", "c", 8)]

    dist, pred = initialize_single_source(vertices, "s")
    print(f"After init: {dist}")

    # 동작을 보이려고 정해진 차례로 변 늦추기
    for u, v, w in edges:
        changed = relax(u, v, w, dist, pred)
        status = "updated" if changed else "no change"
        print(f"Relax ({u},{v},w={w}): d[{v}]={dist[v]}, {status}")

    print(f"\nFinal distances: {dist}")
    print(f"Path s->c: {reconstruct_path(pred, 's', 'c')}")
```

**출력:**

```
After init: {'s': 0, 'a': inf, 'b': inf, 'c': inf}
Relax (s,a,w=10): d[a]=10, updated
Relax (s,b,w=5): d[b]=5, updated
Relax (b,a,w=3): d[a]=8, updated
Relax (a,c,w=1): d[c]=9, updated
Relax (b,c,w=8): d[c]=9, no change
Final distances: {'s': 0, 'a': 8, 'b': 5, 'c': 9}
Path s->c: ['s', 'b', 'a', 'c']
```

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to
  Algorithms* (4th ed.), Chapter 24: Single-Source Shortest Paths.

## 연습문제

**연습문제 1.**
변 늦추기를 엄밀히 정의하여라. 왜 "늦추기"라 부르는가?

??? success "연습문제 1 풀이"
    무게가 $w(u, v)$인 변 $(u, v)$의 늦추기는 이렇다. 곧 $d[v] > d[u] + w(u, v)$이면 $d[v] = d[u] + w(u, v)$으로 새로 고치고 $\pi[v] = u$으로 놓는다. "늦추기"라는 말은 제약을 느슨하게 푸는 데 빗댄 것이다. $d[v]$은 $\delta(s, v)$의 위 한계를 나타내고, 늦추기는 너무 늘어난 용수철을 놓아 주듯 이 한계를 참값 쪽으로 조인다(낮춘다). $\square$

---

**연습문제 2.**
위 한계 성질을 증명하여라. 곧 $v \neq s$이면 $d[v]$을 $\infty$으로, $d[s] = 0$으로 첫걸음 잡았다면, 늦추기를 쓰는 어떤 알고리즘에서도 내내 모든 꼭짓점 $v$에 대해 $d[v] \geq \delta(s, v)$이다.

??? success "연습문제 2 풀이"
    늦추기 걸음의 개수에 대한 귀납으로 증명한다. 처음에 $d[s] = 0 = \delta(s, s)$이고 $v \neq s$인 모든 $v$에 대해 $d[v] = \infty \geq \delta(s, v)$이다. 변 $(u, v)$을 늦추기 전에 $d[v] \geq \delta(s, v)$이 성립한다고 하자. 늦춘 뒤 $d[v] = \min(d[v], d[u] + w(u,v))$이다. $d[u] + w(u,v) \geq \delta(s, v)$이 필요하다. 귀납 가정에 따라 $d[u] \geq \delta(s, u)$이다. 삼각 부등식에 따라 $\delta(s, u) + w(u,v) \geq \delta(s, v)$이다. 그러므로 $d[u] + w(u,v) \geq \delta(s, v)$이며 불변량이 지켜진다. $\square$

---

**연습문제 3.**
늦추기는 어떤 차례로 해도 안전하다. 차례가 왜 효율에는 중요하고 맞음에는 중요하지 않은가?

??? success "연습문제 3 풀이"
    늦추기 연산마다 위 한계 성질을 지키고 $d[v]$을 $\delta(s, v)$에 더 가깝게 옮긴다(또는 그대로 둔다). 마지막 결과는 알맞은 차례로 변을 넉넉히 늦추었느냐에만 달렸다(길 늦추기 성질). 차례는 효율에 영향을 준다. 데이크스트라는 가장 좋은 차례로 늦추고(변마다 한 번이면 된다), 벨먼-포드는 모든 변을 $V-1$바퀴 돌리며, SPFA은 줄에 기댄 차례를 쓴다. 모두 같은 결과로 모이지만 늦추기 연산의 총 횟수가 다르다. $\square$

---

**연습문제 4.**
늦추기가 멱등임을 보여라. 곧 ($d$ 값이 바뀌지 않은 채) 같은 변을 두 번 늦추면 두 번째에는 아무 바뀜이 없다.

??? success "연습문제 4 풀이"
    $(u, v)$을 처음 늦춘 뒤 $d[v] \leq d[u] + w(u, v)$이다. (첫 번째 이후 $d[u]$과 $d[v]$이 바뀌지 않았다고 놓고) 두 번째로 늦출 때 조건 $d[v] > d[u] + w(u, v)$이 거짓이므로 새로 고쳐지지 않는다. 다만 두 늦추기 사이에 ($u$으로 들어오는 다른 변을 늦춰) $d[u]$이 줄었다면 $(u, v)$의 두 번째 늦추기가 $d[v]$을 더 줄일 수 있다. 멱등성은 입력 값이 바뀌지 않았을 때만 성립한다. $\square$
