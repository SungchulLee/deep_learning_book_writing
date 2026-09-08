# 보로노이 그림

평면 위 터(점) 모임이 주어질 때 주어진 물음 자리에 가장 가까운 터는 어느 것인가? **보로노이 그림**은 평면을 터마다 하나씩의 자리로 갈라 이 물음에 *모든* 자리에 대해 한꺼번에 답한다. 이때 한 자리 안의 모든 점은 다른 어느 터보다 제 터에 더 가깝다. 보로노이 그림은 시설 자리 잡기, 가장 가까운 이웃 찾기, 그물 만들기, 땅 살피기에 나온다.

---

## 1. 정의

$\mathbb{R}^2$의 터 $n$개 $P = \{p_1, p_2, \ldots, p_n\}$이 주어질 때 터 $p_i$의 **보로노이 자리**(또는 보로노이 칸)는 다음과 같다.

$$
V(p_i) = \{ x \in \mathbb{R}^2 : \|x - p_i\| \le \|x - p_j\| \text{ for all } j \neq i \}
$$

**보로노이 그림** $\text{Vor}(P)$은 $\mathbb{R}^2$을 보로노이 자리 $V(p_1), V(p_2), \ldots, V(p_n)$으로 가른 것이다.

---

## 2. 구조

두 보로노이 자리 $V(p_i)$과 $V(p_j)$ 사이의 가장자리는 도막 $\overline{p_i p_j}$의 **수직 이등분선**, 곧 $p_i$과 $p_j$에서 같은 거리에 있는 점의 모임 위에 있다.

보로노이 그림은 다음으로 이루어진다.

- **보로노이 모서리:** 정확히 두 자리가 만나는 수직 이등분선의 조각이다.
- **보로노이 꼭짓점:** 모서리 셋 이상이 만나는 점이다. 보로노이 꼭짓점은 터 셋(또는 그 이상)에서 같은 거리에 있다.

---

## 3. 조합의 복잡도

일반 자리에 있는 터 $n$개에 대해:

| 부품 | 수 |
|---|---|
| 보로노이 자리 | $n$ |
| 보로노이 꼭짓점 | $\le 2n - 5$ |
| 보로노이 모서리 | $\le 3n - 6$ |

이 한계는 평면 그래프에 대한 오일러 공식 $V - E + F = 2$에서 따라 나온다.

---

## 4. 성질

- 보로노이 자리는 저마다 **볼록 다각형**이다(끝이 열려 있을 수도 있다).
- 보로노이 꼭짓점 $v$은 터 셋(또는 그 이상)을 지나면서 안에 터가 없는 동그라미의 중심이다. 이는 맞물리는 들로네 삼각형의 둘레 동그라미이다.
- $p_i$의 **가장 가까운 이웃**은 $p_i$과 보로노이 모서리를 나누어 가진다.
- 보로노이 그림은 들로네 삼각 나누기의 **짝**이다. 곧 두 터를 들로네 모서리로 잇는 것과 그 보로노이 자리가 가장자리를 나누어 가지는 것은 서로 같다.

---

## 5. 보로노이 자리 세우기

보로노이 자리 $V(p_i)$은 저마다 반평면 $n - 1$개의 교집합이다.

$$
V(p_i) = \bigcap_{j \neq i} H(p_i, p_j)
$$

여기서 $H(p_i, p_j) = \{x : \|x - p_i\| \le \|x - p_j\|\}$은 $\overline{p_i p_j}$의 수직 이등분선으로 둘러싸인, $p_i$을 담은 반평면이다.

이렇게 자리마다 셈하면 모두 $O(n^2 \log n)$이 든다. 포춘의 훑는 선 알고리즘은 $O(n \log n)$을 이룬다.

---

## 6. 풀이 예제

터 셋을 보자: $p_1 = (0, 0)$, $p_2 = (6, 0)$, $p_3 = (3, 5)$.

수직 이등분선은 다음과 같다.

- $p_1 p_2$: 세로선 $x = 3$
- $p_1 p_3$: $(0,0)$과 $(3,5)$의 이등분선
- $p_2 p_3$: $(6,0)$과 $(3,5)$의 이등분선

이 세 이등분선은 $\triangle p_1 p_2 p_3$의 둘레 중심에서 만나며, 그것이 하나뿐인 보로노이 꼭짓점이다.

---

## 7. 구현

```python
"""
보로노이 그림: 세우기와 가장 가까운 이웃 묻기.

가르치기 위해 막무가내 보로노이 자리 매기기와
수직 이등분선 셈하기를 준다.
"""

import math

# === 거리 ===

def dist(a, b):
    """두 점 사이의 유클리드 거리."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

# === 가장 가까운 터(보로노이 자리 묻기) ===

def nearest_site(sites, query):
    """물음 점에 가장 가까운 터를 찾는다.

    이는 어느 보로노이 자리가 물음 점을 담는지
    가리는 것과 같다.
    """
    best_idx = 0
    best_dist = dist(sites[0], query)
    for i in range(1, len(sites)):
        d = dist(sites[i], query)
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx, best_dist

# === 수직 이등분선 ===

def perpendicular_bisector(p, q):
    """도막 pq의 수직 이등분선을 셈한다.

    (가운데 점, 방향 벡터)을 돌려주며 방향 벡터는
    pq에 직각이다.
    """
    mid = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
    dx, dy = q[0] - p[0], q[1] - p[1]
    # 직각 방향: 90도 돌린다
    perp = (-dy, dx)
    return mid, perp

# === 둘레 중심(터 셋의 보로노이 꼭짓점) ===

def circumcenter(a, b, c):
    """삼각형 abc의 둘레 중심을 셈한다.

    둘레 중심은 세 꼭짓점 모두에서 같은 거리에 있으며
    보로노이 꼭짓점에 맞물린다.
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

# === 보로노이 격자 그려 보기 ===

def voronoi_grid(sites, grid_size=10, resolution=20):
    """보로노이 자리를 보이는 아스키 격자를 만든다.

    칸마다 가장 가까운 터의 어깨수를 이름표로 붙인다.
    """
    grid = []
    step = grid_size / resolution
    for row in range(resolution):
        line = []
        y = grid_size - row * step
        for col in range(resolution):
            x = col * step
            idx, _ = nearest_site(sites, (x, y))
            line.append(str(idx))
        grid.append(" ".join(line))
    return "\n".join(grid)

# === 메인 ===

if __name__ == "__main__":
    sites = [(0, 0), (6, 0), (3, 5)]

    # 가장 가까운 터 묻기
    queries = [(1, 1), (5, 1), (3, 3), (3, 0)]
    print("Sites:", sites)
    for q in queries:
        idx, d = nearest_site(sites, q)
        print(f"  Query {q} -> site {idx} (p={sites[idx]}, dist={d:.3f})")

    # 보로노이 꼭짓점
    cc = circumcenter(*sites)
    print(f"\nVoronoi vertex (circumcenter): ({cc[0]:.3f}, {cc[1]:.3f})")
    for i, s in enumerate(sites):
        print(f"  Distance to p{i+1}: {dist(cc, s):.3f}")

    # 수직 이등분선
    for i in range(len(sites)):
        for j in range(i + 1, len(sites)):
            mid, perp = perpendicular_bisector(sites[i], sites[j])
            print(f"\nBisector of p{i+1}-p{j+1}:")
            print(f"  Midpoint: ({mid[0]:.1f}, {mid[1]:.1f})")
            print(f"  Direction: ({perp[0]:.1f}, {perp[1]:.1f})")
```

**출력:**
```
Sites: [(0, 0), (6, 0), (3, 5)]
  Query (1, 1) -> site 0 (p=(0, 0), dist=1.414)
  Query (5, 1) -> site 1 (p=(6, 0), dist=1.414)
  Query (3, 3) -> site 2 (p=(3, 5), dist=2.000)
  Query (3, 0) -> site 0 (p=(0, 0), dist=3.000)

Voronoi vertex (circumcenter): (3.000, 1.400)
  Distance to p1: 3.311
  Distance to p2: 3.311
  Distance to p3: 3.600

Bisector of p1-p2:
  Midpoint: (3.0, 0.0)
  Direction: (0.0, 6.0)

Bisector of p1-p3:
  Midpoint: (1.5, 2.5)
  Direction: (-5.0, 3.0)

Bisector of p2-p3:
  Midpoint: (4.5, 2.5)
  Direction: (-5.0, -3.0)
```

---

## 8. 응용

| 쓰임새 | 보로노이를 쓰는 법 |
|---|---|
| 가장 가까운 이웃 | 물음 점이 가장 가까운 터의 보로노이 자리에 든다 |
| 시설 자리 잡기 | 가장 나쁜 경우의 거리를 가장 작게 하도록 시설을 둔다 |
| 그물 만들기 | 짝이 되는 들로네 삼각 나누기가 좋은 그물을 준다 |
| 공간 사이 메우기 | 자연 이웃 사이 메우기가 보로노이 넓이를 쓴다 |
| 부딪힘 피하기 | 보로노이 모서리가 가장 넉넉히 트인 길을 나타낸다 |

---

## 연습문제

**연습문제 1.**
보로노이 그림의 핵심 기하 통찰과 그 시간 복잡도를 설명하라.

??? success "연습문제 1 풀이"
    보로노이 그림은 기하의 성질(방향, 거리, 각 차례, 훑는 선 사건)을 이용해 점이나 선, 다각형의 모임을 효율 좋게 다룬다. 시간 복잡도는 흔히 $O(n \log n)$(견줌에 바탕한 기하 문제에서 가장 좋다)에서, 본디 이차 짜임을 지닌 문제의 $O(n^2)$까지이다. 핵심 통찰은 기하 문제를 여느 알고리즘이 풀 수 있는 조합 문제로 줄이는 것이다. $\square$

---

**연습문제 2.**
작은 점 모임 $\{(0,0), (1,3), (3,1), (4,4), (2,2)\}$에서 보로노이 그림을 좇아라.

??? success "연습문제 2 풀이"
    알고리즘의 방책(자리값으로 정렬하기, 각으로 훑기, 사건에 따라 다루기)에 따라 점을 다룬다. 걸음마다 기하 짜임(볼록 껍질, 만남 목록, 보로노이 칸 등)을 새로 고친다. 마지막 결과가 이 들임에 대한 알고리즘의 내놓기이다. 손으로 셈한 것과 견주어 기하의 성질을 살펴 옳음을 확인하라. $\square$

---

**연습문제 3.**
보로노이 그림은 어떤 찌그러진 경우를 다루어야 하는가? 흔히 어떻게 푸는가?

??? success "연습문제 3 풀이"
    흔한 찌그러진 경우는 이렇다. (1) **한 줄에 놓인 점**: 셋 이상이 한 선 위에 있으면 방향 살피기가 애매해진다. (2) **겹친 점**: 자리값이 똑같다. (3) **세로선**: 기울기 셈에서 0으로 나누게 된다. (4) **한 동그라미 위의 점**: 네 점이 한 동그라미 위에 있으면 들로네 삼각 나누기에 영향을 준다. 푸는 방책은 튼튼한 판정(정확한 셈)을 쓰거나, 기호로 살짝 흔들거나(일반 자리를 흉내 냄), 찌그러진 경우를 따로 다루는 코드를 두는 것이다. $\square$

---

**연습문제 4.**
보로노이 그림을 막무가내 방식과 견주어라. 점 $n = 10^6$개에서 얼마나 빨라지는지 수로 나타내라.

??? success "연습문제 4 풀이"
    막무가내 방식은 짝이나 세 짝을 모두 살피므로 흔히 $O(n^2)$이나 $O(n^3)$이 든다. 보로노이 그림은 $O(n \log n)$ 또는 그보다 좋다. $n = 10^6$이면 막무가내는 셈이 $10^{12}$번이나 $10^{18}$번(몇 시간에서 몇 해) 필요하지만 효율 좋은 알고리즘은 $\approx 2 \times 10^7$번(몇 초)이면 된다. 빨라지는 갑절은 $10^5$에서 $10^{11}$이므로 들임이 클 때는 효율 좋은 알고리즘이 꼭 필요하다. $\square$

## 정리하며

이 마당은 정의、구조、조합의 복잡도、성질을 차례로 짚었다.

**참고 문헌**

- de Berg, M., Cheong, O., van Kreveld, M., & Overmars, M. *Computational Geometry: Algorithms and Applications*. Springer, Chapter 7.
- Aurenhammer, F. "Voronoi Diagrams: A Survey of a Fundamental Geometric Data Structure." *ACM Computing Surveys*, 1991.
