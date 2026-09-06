# 가장 가까운 점 짝

평면 위 $n$개 점이 주어질 때 어느 두 점이 가장 가까운가? 막무가내 방식은 $\binom{n}{2}$개 짝을 모두 $O(n^2)$ 시간에 살핀다. 옛부터의 나누어 이기기 알고리즘은 이를 $O(n \log n)$에 풀고, 훑는 선 변형도 같은 한계를 이룬다. 이 쪽에서는 두 방식을 모두 보인다.

## 문제 서술

$\mathbb{R}^2$의 점 $n$개 모임 $P = \{p_1, p_2, \ldots, p_n\}$이 주어질 때 유클리드 거리를 가장 작게 하는 $i \neq j$인 짝 $(p_i, p_j)$을 찾아라.

$$
d(p_i, p_j) = \sqrt{(p_{i,x} - p_{j,x})^2 + (p_{i,y} - p_{j,y})^2}
$$

## 나누어 이기기 방식

### 알고리즘

1. 점을 $x$자리값으로 **정렬한다**.
2. 정렬한 목록을 가운데 $x$값에서 두 반 $P_L$과 $P_R$으로 **가른다**.
3. $P_L$(거리 $\delta_L$)과 $P_R$(거리 $\delta_R$)에서 가장 가까운 짝을 되돌이로 찾아 **이긴다**. $\delta = \min(\delta_L, \delta_R)$이라 하자.
4. 가르는 선을 넘는 짝을 살펴 **아우른다**. 가운데값 둘레의 너비 $2\delta$인 세로 띠 안의 점만 더 가까운 짝을 이룰 수 있다. 이 띠 안에서 점마다 많아야 다른 7개 점과 견주면 된다($y$자리값으로 정렬한 차례에서).

### 왜 견줌이 7번뿐인가?

!!! tip "띠의 성김 논증"
    띠 안의 $\delta \times 2\delta$ 직사각형을 보자. 서로 거리가 $\ge \delta$을 지키면서 이 직사각형에 들어갈 수 있는 점은 많아야 8개(양쪽에서 4개씩)이다. 따라서 점마다 $y$으로 정렬한 차례에서 뒤따르는 많아야 7개 점만 살핀다.

### 점화식

$$
T(n) = 2T(n/2) + O(n)
$$

마스터 정리에 따라 $T(n) = O(n \log n)$이다.

## 훑는 선 방식

훑는 선 변형은 점을 왼쪽에서 오른쪽으로 다루며 $y$자리값으로 정렬한 살아 있는 점 모임을 고른 이진 찾기 나무에 지킨다.

1. 점을 $x$자리값으로 정렬한다.
2. $\delta = \infty$과 빈 살아 있는 모임을 첫자리매김한다.
3. 각 점 $p$에 대해(왼쪽에서 오른쪽으로):
    - 살아 있는 모임에서 $x$자리값이 $p_x - \delta$보다 작은 점을 모두 뺀다.
    - 살아 있는 모임에서 $y$자리값이 $[p_y - \delta, p_y + \delta]$에 드는 점을 묻는다.
    - 더 가까운 이웃이 있으면 $\delta$을 새로 고친다.
    - $p$을 살아 있는 모임에 넣는다.

점마다 한 번 넣고 한 번 지운다. 물음마다 $O(1)$개 점을 돌려준다(같은 성김 논증에 따라). 모두 $O(n \log n)$ 시간이다.

## 풀이 예제

점: $(2,3)$, $(12,30)$, $(40,50)$, $(5,1)$, $(12,10)$, $(3,4)$.

$x$으로 정렬한 뒤: $(2,3)$, $(3,4)$, $(5,1)$, $(12,10)$, $(12,30)$, $(40,50)$.

**왼쪽 반** $\{(2,3), (3,4), (5,1)\}$: 가장 가까운 짝은 $((2,3), (3,4))$이며 $\delta_L = \sqrt{2} \approx 1.414$이다.

**오른쪽 반** $\{(12,10), (12,30), (40,50)\}$: 가장 가까운 짝은 $((12,10), (12,30))$이며 $\delta_R = 20$이다.

$\delta = \min(1.414, 20) = 1.414$이다. 가운데값($x \approx 5$) 둘레의 너비 $2\delta \approx 2.83$인 띠에는 왼쪽 절반의 점만 있으므로 가장자리를 넘는 더 가까운 짝은 없다.

**결과:** 가장 가까운 짝은 $((2,3), (3,4))$이며 거리는 $\sqrt{2}$이다.

## 구현

```python
"""
평면에서 가장 가까운 점 짝.

O(n log n) 시간을 이루는 나누어 이기기 알고리즘을 짠다.
"""

import math


# === 거리 ===

def dist(p, q):
    """두 점 사이의 유클리드 거리를 셈한다."""
    return math.hypot(p[0] - q[0], p[1] - q[1])


# === 막무가내(바탕 경우) ===

def brute_force(points):
    """작은 점 모임에서 가장 가까운 짝을 찾는다.

    아래 문제의 점이 3개 이하일 때 바탕 경우로 쓴다.
    """
    min_d = float("inf")
    pair = (None, None)
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            d = dist(points[i], points[j])
            if d < min_d:
                min_d = d
                pair = (points[i], points[j])
    return min_d, pair


# === 띠 살피기 ===

def closest_in_strip(strip, delta):
    """띠 안의 짝 가운데 거리가 delta보다 작은 것을 살핀다.

    띠는 y자리값으로 정렬되어 있다. 점마다 뒤따르는
    많아야 7개 점과 견준다.
    """
    min_d = delta
    pair = (None, None)
    strip.sort(key=lambda p: p[1])

    for i in range(len(strip)):
        j = i + 1
        while j < len(strip) and (strip[j][1] - strip[i][1]) < min_d:
            d = dist(strip[i], strip[j])
            if d < min_d:
                min_d = d
                pair = (strip[i], strip[j])
            j += 1

    return min_d, pair


# === 나누어 이기기 ===

def closest_pair_rec(px):
    """x자리값으로 정렬한 점에 대한 되돌이 가장 가까운 짝."""
    n = len(px)
    if n <= 3:
        return brute_force(px)

    mid = n // 2
    mid_x = px[mid][0]

    dl, pair_l = closest_pair_rec(px[:mid])
    dr, pair_r = closest_pair_rec(px[mid:])

    if dl < dr:
        delta, best_pair = dl, pair_l
    else:
        delta, best_pair = dr, pair_r

    strip = [p for p in px if abs(p[0] - mid_x) < delta]
    ds, pair_s = closest_in_strip(strip, delta)

    if ds < delta:
        return ds, pair_s
    return delta, best_pair


def closest_pair(points):
    """가장 가까운 점 짝을 O(n log n) 시간에 찾는다.

    인수:
        points: (x, y) 짝의 목록.

    반환값:
        (거리, (점_a, 점_b)).
    """
    px = sorted(points, key=lambda p: (p[0], p[1]))
    return closest_pair_rec(px)


# === 메인 ===

if __name__ == "__main__":
    pts = [(2, 3), (12, 30), (40, 50), (5, 1), (12, 10), (3, 4)]
    d, pair = closest_pair(pts)
    print(f"Points: {pts}")
    print(f"Closest pair: {pair}")
    print(f"Distance: {d:.4f}")

    # 또 다른 예
    pts2 = [(0, 0), (1, 0), (0, 1), (1, 1), (0.5, 0.5)]
    d2, pair2 = closest_pair(pts2)
    print(f"\nPoints: {pts2}")
    print(f"Closest pair: {pair2}")
    print(f"Distance: {d2:.4f}")
```

**출력:**
```
Points: [(2, 3), (12, 30), (40, 50), (5, 1), (12, 10), (3, 4)]
Closest pair: ((2, 3), (3, 4))
Distance: 1.4142

Points: [(0, 0), (1, 0), (0, 1), (1, 1), (0.5, 0.5)]
Closest pair: ((0.5, 0.5), (0, 0))
Distance: 0.7071
```

## 복잡도 요약

| 방법 | 시간 | 공간 |
|---|---|---|
| 막무가내 | $O(n^2)$ | $O(1)$ |
| 나누어 이기기 | $O(n \log n)$ | $O(n)$ |
| 훑는 선 | $O(n \log n)$ | $O(n)$ |
| 마구잡이 | 기댓값 $O(n)$ | $O(n)$ |

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms*. MIT Press, Chapter 33.
- de Berg, M., Cheong, O., van Kreveld, M., & Overmars, M. *Computational Geometry: Algorithms and Applications*. Springer.

## 연습문제

**연습문제 1.**
가장 가까운 점 짝의 핵심 기하 통찰과 그 시간 복잡도를 설명하라.

??? success "연습문제 1 풀이"
    가장 가까운 점 짝은 기하의 성질(방향, 거리, 각 차례, 훑는 선 사건)을 이용해 점이나 선, 다각형의 모임을 효율 좋게 다룬다. 시간 복잡도는 흔히 $O(n \log n)$(견줌에 바탕한 기하 문제에서 가장 좋다)에서, 본디 이차 짜임을 지닌 문제의 $O(n^2)$까지이다. 핵심 통찰은 기하 문제를 여느 알고리즘이 풀 수 있는 조합 문제로 줄이는 것이다. $\square$

---

**연습문제 2.**
작은 점 모임 $\{(0,0), (1,3), (3,1), (4,4), (2,2)\}$에서 가장 가까운 점 짝을 좇아라.

??? success "연습문제 2 풀이"
    알고리즘의 방책(자리값으로 정렬하기, 각으로 훑기, 사건에 따라 다루기)에 따라 점을 다룬다. 걸음마다 기하 짜임(볼록 껍질, 만남 목록, 보로노이 칸 등)을 새로 고친다. 마지막 결과가 이 들임에 대한 알고리즘의 내놓기이다. 손으로 셈한 것과 견주어 기하의 성질을 살펴 옳음을 확인하라. $\square$

---

**연습문제 3.**
가장 가까운 점 짝은 어떤 찌그러진 경우를 다루어야 하는가? 흔히 어떻게 푸는가?

??? success "연습문제 3 풀이"
    흔한 찌그러진 경우는 이렇다. (1) **한 줄에 놓인 점**: 셋 이상이 한 선 위에 있으면 방향 살피기가 애매해진다. (2) **겹친 점**: 자리값이 똑같다. (3) **세로선**: 기울기 셈에서 0으로 나누게 된다. (4) **한 동그라미 위의 점**: 네 점이 한 동그라미 위에 있으면 들로네 삼각 나누기에 영향을 준다. 푸는 방책은 튼튼한 판정(정확한 셈)을 쓰거나, 기호로 살짝 흔들거나(일반 자리를 흉내 냄), 찌그러진 경우를 따로 다루는 코드를 두는 것이다. $\square$

---

**연습문제 4.**
가장 가까운 점 짝을 막무가내 방식과 견주어라. 점 $n = 10^6$개에서 얼마나 빨라지는지 수로 나타내라.

??? success "연습문제 4 풀이"
    막무가내 방식은 짝이나 세 짝을 모두 살피므로 흔히 $O(n^2)$이나 $O(n^3)$이 든다. 가장 가까운 점 짝은 $O(n \log n)$ 또는 그보다 좋다. $n = 10^6$이면 막무가내는 셈이 $10^{12}$번이나 $10^{18}$번(몇 시간에서 몇 해) 필요하지만 효율 좋은 알고리즘은 $\approx 2 \times 10^7$번(몇 초)이면 된다. 빨라지는 갑절은 $10^5$에서 $10^{11}$이므로 들임이 클 때는 효율 좋은 알고리즘이 꼭 필요하다. $\square$
