# 보로노이-들로네 짝됨

보로노이 그림과 들로네 삼각 나누기는 같은 기하의 앎을 서로 짝이 되는 꼴로 담는다. 이 짝됨을 알면 하나에서 다른 하나를 $O(n)$ 시간에 셈하고 두 짜임 사이에서 성질을 자유로이 옮길 수 있다. 이 이음은 셈 기하에서 가장 아름다운 관계 가운데 하나이다.

---

## 1. 짝됨의 대응

터 $n$개 $P = \{p_1, \ldots, p_n\}$이 주어질 때 **보로노이 그림** $\text{Vor}(P)$과 **들로네 삼각 나누기** $DT(P)$은 다음 옮김으로 이어진다.

| 보로노이 | 들로네 |
|---|---|
| 자리 $V(p_i)$ | 꼭짓점 $p_i$ |
| $V(p_i)$과 $V(p_j)$ 사이의 모서리 | 모서리 $\overline{p_i p_j}$ |
| 꼭짓점(모서리 셋 이상이 만나는 점) | 삼각형(터 셋 이상이 둘레를 이룸) |

간추리면 면은 꼭짓점이 되고 모서리는 모서리로 남으며 꼭짓점은 면이 된다.

---

## 2. 엄밀한 서술

**정리.** 두 터 $p_i$과 $p_j$이 들로네 모서리로 이어지는 것과 그 보로노이 자리가 가장자리 모서리를 나누어 가지는 것은 서로 같다.

**밝힘 밑그림.** $V(p_i)$과 $V(p_j)$이 모서리 $e$을 나누어 가지면 $e$ 위의 모든 점은 $p_i$과 $p_j$에서 같은 거리에 있고 다른 모든 터보다는 멀다. $e$의 가운데 점(또는 $e$ 위 아무 점)은 $p_i$과 $p_j$을 지나면서 안에 다른 터가 없는 동그라미의 중심이다. 이것이 바로 들로네 모서리 $\overline{p_i p_j}$의 빈 둘레 동그라미 조건이다. $\square$

---

## 3. 꼭짓점과 삼각형의 대응

세 자리 $V(p_i)$, $V(p_j)$, $V(p_k)$이 만나는 보로노이 꼭짓점 $v$은 $p_i$, $p_j$, $p_k$에서 같은 거리에 있다. 따라서 $v$은 $\triangle p_i p_j p_k$의 둘레 중심이고 이 삼각형은 $DT(P)$에 나타난다.

$$
\|v - p_i\| = \|v - p_j\| = \|v - p_k\| = r
$$

여기서 $r$은 둘레 반지름이다. 빈 둘레 동그라미 성질은 $v$을 중심으로 반지름 $r$인 동그라미 안에 다른 터가 없음을 보장한다.

---

## 4. 하나에서 다른 하나 세우기

### 보로노이에서 들로네로

$\text{Vor}(P)$이 주어질 때:

1. $V(p_i)$과 $V(p_j)$을 가르는 보로노이 모서리마다 모서리 $\overline{p_i p_j}$을 $DT(P)$에 더한다.

보로노이 그림의 모서리가 $O(n)$개이므로 $O(n)$ 시간이 든다.

### 들로네에서 보로노이로

$DT(P)$이 주어질 때:

1. 들로네 삼각형마다 둘레 중심을 셈한다. 이것이 보로노이 꼭짓점이다.
2. 이웃한 삼각형(들로네 모서리를 나누어 가진 것)의 둘레 중심을 잇는다. 이 이음이 보로노이 모서리를 이룬다.

이 또한 $O(n)$ 시간이 든다.

---

## 5. 짝됨으로 옮겨 가는 성질

!!! note "무엇이 옮겨 가는가"
    많은 성질이 자연스러운 짝 풀이를 갖는다.

    - $p_i$의 **가장 가까운 이웃**(들로네 성질)은 $V(p_i)$과 가장 긴 가장자리를 나누어 가진 보로노이 자리에 맞물린다.
    - **유클리드 최소 뻗음 나무**는 $DT(P)$의 아래 그래프이므로 $DT(P)$을 먼저 셈하여 $O(n \log n)$ 시간에 찾을 수 있다.
    - $P$의 **볼록 껍질 모서리**는 끝이 열린 보로노이 모서리에 맞물린다.

---

## 6. 풀이 예제

터 넷을 보자: $p_1 = (0, 0)$, $p_2 = (4, 0)$, $p_3 = (4, 4)$, $p_4 = (0, 4)$.

**들로네 삼각 나누기**에는 삼각형 둘이 있다: $\triangle p_1 p_2 p_3$과 $\triangle p_1 p_3 p_4$(대각선 $\overline{p_1 p_3}$을 쓴다).

**보로노이 꼭짓점**은 둘레 중심이다.

- $\triangle p_1 p_2 p_3$: 둘레 중심은 $(2, 2)$
- $\triangle p_1 p_3 p_4$: 둘레 중심은 $(2, 2)$

네 점이 한 동그라미 위에 있으므로($(2, 2)$을 중심으로 반지름 $2\sqrt{2}$인 동그라미) 두 둘레 중심이 $(2, 2)$에서 겹치고, 이 하나뿐인 보로노이 꼭짓점에서 네 자리가 모두 만난다.

---

## 7. 구현

```python
"""
보로노이-들로네 짝됨: 두 짜임 사이의 바꾸기.

보로노이 꼭짓점과 들로네 삼각형 사이의 대응을
둘레 중심 셈하기로 보인다.
"""

import math

# === 둘레 중심 ===

def circumcenter(a, b, c):
    """삼각형 abc의 둘레 중심을 셈한다.

    둘레 중심은 이 들로네 삼각형에 짝이 되는 보로노이 꼭짓점이다.
    """
    ax, ay = a
    bx, by = b
    cx, cy = c
    D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(D) < 1e-10:
        return None
    ux = ((ax * ax + ay * ay) * (by - cy)
        + (bx * bx + by * by) * (cy - ay)
        + (cx * cx + cy * cy) * (ay - by)) / D
    uy = ((ax * ax + ay * ay) * (cx - bx)
        + (bx * bx + by * by) * (ax - cx)
        + (cx * cx + cy * cy) * (bx - ax)) / D
    return (ux, uy)

# === 짝됨 뽑아내기 ===

def delaunay_to_voronoi_vertices(triangles, points):
    """들로네 삼각형에서 보로노이 꼭짓점을 뽑아낸다.

    들로네 삼각형은 저마다 정확히 보로노이 꼭짓점 하나에 맞물리며
    (그 둘레 중심).
    """
    vertices = []
    for tri in triangles:
        a, b, c = [points[i] for i in tri]
        cc = circumcenter(a, b, c)
        if cc is not None:
            vertices.append(cc)
    return vertices

def voronoi_edges_from_delaunay(triangles, points):
    """이웃한 들로네 삼각형에서 보로노이 모서리를 뽑아낸다.

    모서리를 나누어 가진 두 들로네 삼각형은 그 둘레 중심을 이어
    보로노이 모서리를 낸다.
    """
    # 이웃 관계 세우기: 모서리 -> 삼각형 어깨수 목록
    edge_to_tri = {}
    for idx, tri in enumerate(triangles):
        for i in range(3):
            edge = tuple(sorted([tri[i], tri[(i + 1) % 3]]))
            edge_to_tri.setdefault(edge, []).append(idx)

    voronoi_edges = []
    voronoi_vertices = delaunay_to_voronoi_vertices(triangles, points)

    for edge, tri_list in edge_to_tri.items():
        if len(tri_list) == 2:
            v1 = voronoi_vertices[tri_list[0]]
            v2 = voronoi_vertices[tri_list[1]]
            voronoi_edges.append((v1, v2))

    return voronoi_edges

# === 메인 ===

if __name__ == "__main__":
    # 점 다섯
    points = [(0, 0), (4, 0), (4, 4), (0, 4), (2, 2)]

    # 올바른 들로네 삼각 나누기(점의 어깨수)
    triangles = [
        (0, 1, 4),
        (1, 2, 4),
        (2, 3, 4),
        (3, 0, 4),
    ]

    print("Points:", points)
    print("Delaunay triangles:", triangles)

    # 보로노이 꼭짓점
    vv = delaunay_to_voronoi_vertices(triangles, points)
    print("\nVoronoi vertices (circumcenters):")
    for i, v in enumerate(vv):
        print(f"  Triangle {triangles[i]} -> ({v[0]:.3f}, {v[1]:.3f})")

    # 보로노이 모서리
    ve = voronoi_edges_from_delaunay(triangles, points)
    print(f"\nVoronoi edges ({len(ve)} edges):")
    for v1, v2 in ve:
        length = math.hypot(v2[0] - v1[0], v2[1] - v1[1])
        print(f"  ({v1[0]:.2f},{v1[1]:.2f}) -- ({v2[0]:.2f},{v2[1]:.2f})"
              f"  length={length:.3f}")
```

**출력:**

```
Points: [(0, 0), (4, 0), (4, 4), (0, 4), (2, 2)]
Delaunay triangles: [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]

Voronoi vertices (circumcenters):
  Triangle (0, 1, 4) -> (2.000, 0.000)
  Triangle (1, 2, 4) -> (4.000, 2.000)
  Triangle (2, 3, 4) -> (2.000, 4.000)
  Triangle (3, 0, 4) -> (0.000, 2.000)

Voronoi edges (4 edges):
  (2.00,0.00) -- (4.00,2.00)  length=2.828
  (2.00,0.00) -- (0.00,2.00)  length=2.828
  (4.00,2.00) -- (2.00,4.00)  length=2.828
  (2.00,4.00) -- (0.00,2.00)  length=2.828
```

---

## 8. 3차원으로 들어 올리기

이 짝됨에는 **포물면 들어 올리기**를 거친 아름다운 기하 풀이가 있다. 각 점 $(x, y)$을 포물면 $z = x^2 + y^2$ 위의 $(x, y, x^2 + y^2)$으로 옮긴다. 들어 올린 점의 아래 볼록 껍질을 내리쏘면 들로네 삼각 나누기가 되고, 포물면에 닿는 평면은 보로노이 자리에 이어진다.

---

## 연습문제

**연습문제 1.**
보로노이-들로네 짝됨의 핵심 기하 통찰과 그 시간 복잡도를 설명하라.

??? success "연습문제 1 풀이"
    보로노이-들로네 짝됨은 기하의 성질(방향, 거리, 각 차례, 훑는 선 사건)을 이용해 점이나 선, 다각형의 모임을 효율 좋게 다룬다. 시간 복잡도는 흔히 $O(n \log n)$(견줌에 바탕한 기하 문제에서 가장 좋다)에서, 본디 이차 짜임을 지닌 문제의 $O(n^2)$까지이다. 핵심 통찰은 기하 문제를 여느 알고리즘이 풀 수 있는 조합 문제로 줄이는 것이다. $\square$

---

**연습문제 2.**
작은 점 모임 $\{(0,0), (1,3), (3,1), (4,4), (2,2)\}$에서 보로노이-들로네 짝됨을 좇아라.

??? success "연습문제 2 풀이"
    알고리즘의 방책(자리값으로 정렬하기, 각으로 훑기, 사건에 따라 다루기)에 따라 점을 다룬다. 걸음마다 기하 짜임(볼록 껍질, 만남 목록, 보로노이 칸 등)을 새로 고친다. 마지막 결과가 이 들임에 대한 알고리즘의 내놓기이다. 손으로 셈한 것과 견주어 기하의 성질을 살펴 옳음을 확인하라. $\square$

---

**연습문제 3.**
보로노이-들로네 짝됨은 어떤 찌그러진 경우를 다루어야 하는가? 흔히 어떻게 푸는가?

??? success "연습문제 3 풀이"
    흔한 찌그러진 경우는 이렇다. (1) **한 줄에 놓인 점**: 셋 이상이 한 선 위에 있으면 방향 살피기가 애매해진다. (2) **겹친 점**: 자리값이 똑같다. (3) **세로선**: 기울기 셈에서 0으로 나누게 된다. (4) **한 동그라미 위의 점**: 네 점이 한 동그라미 위에 있으면 들로네 삼각 나누기에 영향을 준다. 푸는 방책은 튼튼한 판정(정확한 셈)을 쓰거나, 기호로 살짝 흔들거나(일반 자리를 흉내 냄), 찌그러진 경우를 따로 다루는 코드를 두는 것이다. $\square$

---

**연습문제 4.**
보로노이-들로네 짝됨을 막무가내 방식과 견주어라. 점 $n = 10^6$개에서 얼마나 빨라지는지 수로 나타내라.

??? success "연습문제 4 풀이"
    막무가내 방식은 짝이나 세 짝을 모두 살피므로 흔히 $O(n^2)$이나 $O(n^3)$이 든다. 보로노이-들로네 짝됨은 $O(n \log n)$ 또는 그보다 좋다. $n = 10^6$이면 막무가내는 셈이 $10^{12}$번이나 $10^{18}$번(몇 시간에서 몇 해) 필요하지만 효율 좋은 알고리즘은 $\approx 2 \times 10^7$번(몇 초)이면 된다. 빨라지는 갑절은 $10^5$에서 $10^{11}$이므로 들임이 클 때는 효율 좋은 알고리즘이 꼭 필요하다. $\square$

## 정리하며

이 마당은 짝됨의 대응、엄밀한 서술、꼭짓점과 삼각형의 대응、하나에서 다른 하나 세우기을 차례로 짚었다.

**참고 문헌**

- de Berg, M., Cheong, O., van Kreveld, M., & Overmars, M. *Computational Geometry: Algorithms and Applications*. Springer, Chapter 7 and 9.
- Edelsbrunner, H. & Seidel, R. "Voronoi Diagrams and Arrangements." *Discrete & Computational Geometry*, 1986.
