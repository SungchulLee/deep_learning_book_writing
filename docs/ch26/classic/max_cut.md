# 최대 자름 어림

방향 없는 그래프가 주어질 때 **최대 자름** 문제는 가름을 가로지르는 모서리의 수(또는 온 무게)를 가장 크게 하는, 꼭짓점을 두 모임으로 가르는 방법을 묻는다. 최대 자름은 NP-어려움이지만 단순한 마구잡이 알고리즘이 1/2 어림을 이루고, 욕심쟁이 가까이 찾기 알고리즘이 이 보장을 정해진 방식으로 맞춘다.

---

## 1. 문제의 정의

모서리 무게가 $w_e \ge 0$인 방향 없는 그래프 $G = (V, E)$이 주어질 때 다음을 가장 크게 하는 $V$의 가름 $(S, \bar{S})$을 찾아라.

$$
w(S, \bar{S}) = \sum_{\substack{(u,v) \in E \\ u \in S,\, v \in \bar{S}}} w_{uv}
$$

무게 없는 그래프에서는 모든 모서리에 $w_e = 1$이고 목표는 가로지르는 모서리의 수를 센다.

---

## 2. 마구잡이 1/2 어림

**직관.** 꼭짓점마다 서로 매이지 않게 확률 1/2으로 $S$에 넣는다. 모서리마다 정확히 확률 1/2으로 자름을 가로지르므로 평균으로 온 무게의 반을 담는다.

!!! tip "정리"
    마구잡이 알고리즘은 $\mathbb{E}[w(S, \bar{S})] = W/2$을 이루며 여기서 $W = \sum_{e \in E} w_e$이다. $\text{OPT} \le W$이므로 기댓값으로 1/2 어림이 된다.

**밝힘.** 모서리 $e = (u, v)$마다 $e$이 자름을 가로지르면
$X_e = 1$인 표시를 둔다. 그러면 $\Pr[X_e = 1] = 2 \cdot \frac{1}{2} \cdot \frac{1}{2} = \frac{1}{2}$이다.
기댓값의 선형성에 따라,

$$
\mathbb{E}[w(S, \bar{S})] = \sum_{e \in E} w_e \cdot \Pr[X_e = 1]
= \frac{1}{2} \sum_{e \in E} w_e = \frac{W}{2}
\ge \frac{\text{OPT}}{2} \qquad \square
$$

---

## 3. 욕심쟁이 가까이 찾기

**직관.** 아무 가름에서 시작한다. 꼭짓점 하나를 반대쪽으로 옮겨 자름이 커지면 옮긴다. 나아지는 옮김이 없을 때까지 되풀이한다.

**알고리즘:**

1. $S = \emptyset$, $\bar{S} = V$에서 시작한다
2. 꼭짓점 $v$마다 옮겼을 때의 이득, 곧 $v$이 쪽을 바꿀 때 늘어나는 자름 무게를 셈한다
3. 이득이 양수인 꼭짓점이 있으면 이득이 가장 큰 것을 옮긴다
4. 이득이 양수인 옮김이 없을 때까지 되풀이한다

!!! tip "정리"
    가까이 찾기 알고리즘은 무게가 적어도 $W/2$인 자름을 돌려준다.

**밝힘.** 멈출 때 모든 꼭짓점 $v$에 대해 $v$을 옮겨도 자름이 커지지 않는다. $d_{\text{in}}(v)$을 $v$에서 같은 쪽 꼭짓점으로 가는 모서리의 무게, $d_{\text{out}}(v)$을 반대쪽으로 가는 무게라 하자. 나아지지 않음 조건에서 모든 $v$에 대해 $d_{\text{out}}(v) \ge d_{\text{in}}(v)$이다. 모든 꼭짓점에 걸쳐 더하면:

$$
\sum_v d_{\text{out}}(v) \ge \sum_v d_{\text{in}}(v)
$$

가로지르는 모서리는 두 끝점 모두의 $d_{\text{out}}$에, 가로지르지 않는 모서리는 두 끝점 모두의 $d_{\text{in}}$에 보태므로:

$$
2 \cdot w(S, \bar{S}) \ge 2 \cdot (W - w(S, \bar{S}))
$$

$$
w(S, \bar{S}) \ge W/2 \ge \text{OPT}/2 \qquad \square
$$

---

## 4. 괴만스-윌리엄슨 알고리즘

이름난 Goemans와 Williamson(1995)의 반정부호 계획 바탕 알고리즘은 어림 비율 $\alpha_{\text{GW}} \approx 0.878$을 이루며, 이는 하나뿐인 놀이 추측을 가정하면 가장 좋다. 이 알고리즘은 정수 계획을 반정부호 계획으로 느슨하게 한 뒤 아무 초평면 반올림으로 풀이를 반올림한다.

---

## 5. 구현

```python
"""
최대 자름: 마구잡이와 욕심쟁이 가까이 찾기 어림 알고리즘.
"""

import random

# === 마구잡이 1/2 어림 ========================================================

def max_cut_random(n, edges):
    """
    마구잡이 최대 자름: 꼭짓점마다 확률 1/2으로 S에 든다.

    (자름 무게, 모임 S)을 돌려준다.
    기댓값 어림 비율: 1/2.
    """
    S = set()
    for v in range(n):
        if random.random() < 0.5:
            S.add(v)

    cut = sum(w for u, v, w in edges if (u in S) != (v in S))
    return cut, S

# === 욕심쟁이 가까이 찾기 =====================================================

def max_cut_local_search(n, edges):
    """
    가까이 찾기 최대 자름: 자름을 키우려 꼭짓점을 거듭 옮긴다.

    (자름 무게, 모임 S)을 돌려준다.
    보장된 어림 비율: 1/2.
    """
    # 무게를 지닌 이웃 관계를 세운다
    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))

    # 시작: 모든 꼭짓점이 bar_S에 있다
    in_S = [False] * n

    improved = True
    while improved:
        improved = False
        for v in range(n):
            gain = 0
            for u, w in adj[v]:
                if in_S[v] == in_S[u]:
                    gain += w  # 가로지르게 된다
                else:
                    gain -= w  # 가로지르기를 멈춘다
            if gain > 0:
                in_S[v] = not in_S[v]
                improved = True

    S = {v for v in range(n) if in_S[v]}
    cut = sum(w for u, v, w in edges if in_S[u] != in_S[v])
    return cut, S

# === 보여 주기 ===============================================================

if __name__ == "__main__":
    n = 5
    edges = [
        (0, 1, 3), (0, 2, 2), (1, 2, 1),
        (1, 3, 4), (2, 4, 5), (3, 4, 2),
    ]
    W = sum(w for _, _, w in edges)

    random.seed(42)
    cut_rand, S_rand = max_cut_random(n, edges)
    print(f"Random:       cut={cut_rand}, S={S_rand}")

    cut_local, S_local = max_cut_local_search(n, edges)
    print(f"Local search: cut={cut_local}, S={S_local}")
    print(f"Total weight: {W}, lower bound (W/2): {W / 2}")
```

---

## 연습문제

**연습문제 1.**
최대 자름 어림의 어림 알고리즘을 설명하고 그 어림 보장을 밝혀라.

??? success "연습문제 1 풀이"
    이 알고리즘은 다항식 시간에 돌며 가장 좋은 값의 밝힐 수 있는 갑절 안에 드는 풀이를 낸다. 어림 비율은 알고리즘이 내놓은 것을 가장 좋은 값의 아래 한계(가장 작게 하기)나 위 한계(가장 크게 하기), 곧 선형 계획 느슨하게 하기 값이나 조합 한계, 문제의 짜임 성질과 이어 밝힌다. $\square$

---

**연습문제 2.**
최대 자름 어림의 어림 비율을 밝히는 데 어떤 아래 한계 재주를 쓰는가?

??? success "연습문제 2 풀이"
    밝힘은 흔히 알고리즘의 풀이를 느슨하게 한 한계(선형 계획 느슨하게 하기, 분수 풀이, 조합 아래 한계)와 견준다. 가장 작게 하기에서는 $ALG \leq \rho \cdot LP^* \leq \rho \cdot OPT$이다. 가장 크게 하기에서는 $ALG \geq OPT / \rho$이다. 아래 한계는 효율 좋게 셈할 수 있고 쓸모 있는 비율을 줄 만큼 빡빡해야 한다. $\square$

---

**연습문제 3.**
최대 자름 어림의 어림 비율을 더 좋게 할 수 있는가? 알려진 어려움 결과는 무엇인가?

??? success "연습문제 3 풀이"
    어림 비율이 얼마나 빡빡한지는 복잡도 이론의 가정(P $\neq$ NP, 하나뿐인 놀이 추측 등)에 달렸다. 어떤 문제에서는 단순한 욕심쟁이나 반올림 알고리즘이 여느 가정 아래 이미 가장 좋다. 다른 문제에서는 가장 좋은 알고리즘과 가장 센 어려움 결과 사이에 틈이 있어 아직 풀리지 않은 연구 문제로 남아 있다. $\square$

---

**연습문제 4.**
최대 자름 어림을 구체적인 보기에 써서 어림 비율이 참임을 확인하라.

??? success "연습문제 4 풀이"
    작은 보기(예컨대 꼭짓점이나 물건 5~6개)를 고른다. 어림 알고리즘을 한 걸음씩 돌린다. 알고리즘이 내놓은 것을 (작은 보기에서 막무가내로 찾은) 가장 좋은 풀이와 견준다. 비율 $ALG/OPT$(또는 $OPT/ALG$)이 밝힌 한계 안에 드는지 확인한다. 그러면 구체적인 보기에서 이론이 굳어진다. $\square$

## 정리하며

| 알고리즘 | 비율 | 시간 |
|---|---|---|
| 마구잡이 | $1/2$(기댓값) | $O(m)$ |
| 가까이 찾기 | $1/2$ | 한 번 지날 때 $O(nm)$ |
| 괴만스-윌리엄슨 | $\approx 0.878$ | 다항식(반정부호 계획) |

**참고 문헌**

- Goemans, M. X. and Williamson, D. P. "Improved Approximation Algorithms for Maximum Cut." *JACM*, 1995.
- Vazirani, V. V. *Approximation Algorithms*. Springer, 2001.
