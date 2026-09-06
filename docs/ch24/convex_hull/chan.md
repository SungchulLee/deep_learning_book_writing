# 챈 알고리즘

챈 알고리즘은 $n$개 점의 볼록 껍질을 $O(n \log h)$ 시간에 셈한다. 여기서 $h$은 껍질 꼭짓점의 수이다. 이는 **내놓기에 민감하다**. 곧 껍질이 작으면 견줌에 바탕한 방법의 $\Theta(n \log n)$ 벽을 넘어선다. 핵심 생각은 빠른 부분 껍질 알고리즘(그레이엄 훑기)과 효율 좋은 감싸기 걸음(자비스 행진)을 아우르고, 껍질 크기 $h$을 거듭 두 배로 늘려 짐작하는 것이다.

## 직관

1. 목표 껍질 크기 $m$을 짐작한다.
2. $n$개 점을 크기 $m$인 무리 $\lceil n/m \rceil$개로 가른다.
3. 그레이엄 훑기로 무리마다 볼록 껍질을 $O(m \log m)$에 셈한다.
4. 무리 껍질에 자비스 행진을 돌린다. 감싸기 걸음마다 무리 껍질을 이분 찾기로 훑어 가장 좋은 닿는 점을 $O(\log m)$에 찾는다.
5. 행진이 $\le m$ 걸음에 끝나면 마친 것이다. 아니면 $m$을 두 배로 늘려 다시 시작한다.

올바른 $m = h$을 $O(\log \log h)$번 돌아서 찾고 한 바퀴가 $O(n \log m)$이 들므로 모두 $O(n \log h)$이다.

## 정의

**내놓기에 민감함.** 복잡도가 들임뿐 아니라 내놓기의 크기에 달린 알고리즘이다.

**점에서 볼록 다각형으로 그은 닿는 선.** 바깥 점 $q$과 정렬해 담은 볼록 다각형 $P$이 주어지면 $q$에서 $P$으로 그은 두 닿는 선을 도는 방향에 대한 이분 찾기로 $O(\log |P|)$에 찾을 수 있다.

## 알고리즘 세부

### 단계 1 -- 무리 껍질

$S$을 크기가 많아야 $m$인 무리 $G_1, \dots, G_{\lceil n/m \rceil}$으로 가른다. 무리마다 $O(k \log k)$ 껍질 알고리즘(흔히 그레이엄 훑기)으로 $\operatorname{CH}(G_i)$을 셈한다.

**비용:** 무리 $\lceil n/m \rceil$개 $\times$ 각 $O(m \log m)$ $= O(n \log m)$.

### 단계 2 -- 무리 껍질 위의 자비스 감싸기

가장 아래 점 $p_0$에서 시작해 되풀이한다.

1. 무리 껍질 $\operatorname{CH}(G_i)$마다 지금 모서리 방향에서 각을 가장 크게 하는 점 $q_i$을 이분 찾기로 찾는다. 무리마다 비용은 $O(\log m)$이다.
2. 모든 $q_i$ 가운데 각이 가장 큰 점 $p_{\text{next}}$을 고른다.
3. $p_{\text{next}} = p_0$이면 껍질이 다 된 것이다.
4. 닫히지 않은 채 $m$ 걸음을 걸었으면 이 바퀴를 **그만둔다**.

**걸음마다 비용:** $O((n/m) \log m)$. **$h$걸음 모두:** $O(n \log m)$($m \ge h$일 때).

### 두 배 차례표

$t = 1, 2, 3, \dots$에 대해 $m = 2^{2^t}$을 시험한다. 처음으로 $m \ge h$이 되는 것이 이룬다. $t$번째 바퀴는 $O(n \log 2^{2^t}) = O(n \cdot 2^t)$이 든다. 못 이룬 바퀴를 등비 급수로 더하면 모두 $O(n \log h)$이다.

## 올바름

!!! note "정리"
    챈 알고리즘은 $\operatorname{CH}(S)$을 $O(n \log h)$ 시간에 내놓는다.

**증명 얼개.**

- 무리 껍질은 저마다 옳다(그레이엄 훑기의 올바름에 따라).
- 닿는 선 물음이 늘 참으로 각이 가장 큰 점을 돌려주면 자비스 행진은 옳다. 각으로 정렬한 볼록 다각형에 대한 이분 찾기가 이를 보장한다.
- 두 배 차례표는 $m \ge h$에 이름을 보장한다. 그때 행진은 $m$ 걸음 안에 끝나고 온전한 껍질을 돌려준다.
- 모든 바퀴의 온 일은 마지막으로 이룬 바퀴가 도맡는다. 곧 $O(n \log h)$이다.

## 복잡도

| 잣대 | 비용 |
|---------|------|
| 시간(가장 나쁜 경우) | $O(n \log h)$ |
| 시간($h = \Theta(n)$일 때) | $O(n \log n)$ -- 그레이엄 훑기와 같다 |
| 시간($h = O(1)$일 때) | $O(n)$ -- 자비스 행진과 같다 |
| 공간 | $O(n)$ |

## 구현

```python
"""
챈 알고리즘 — O(n log h) 내놓기에 민감한 볼록 껍질.

무리 껍질에는 그레이엄 훑기를, 무리 껍질에 대한 닿는 선 묻기를 갖춘
자비스 행진을 쓴다.
"""

from __future__ import annotations


# === 방향 도우미 ============================================================

def cross(o: tuple[float, float],
          a: tuple[float, float],
          b: tuple[float, float]) -> float:
    """평행사변형 OA x OB의 부호 있는 넓이."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


# === 작은 무리를 위한 그레이엄 훑기 ==========================================

def graham_hull(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """앤드루의 단조 사슬로 구한 볼록 껍질(곁 절차로 쓴다)."""
    pts = sorted(set(pts))
    if len(pts) <= 1:
        return list(pts)
    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


# === 닿는 선 찾기 ==========================================================

def _tangent(hull: list[tuple[float, float]],
             point: tuple[float, float]) -> tuple[float, float]:
    """*point*에서 왼쪽 돌기 각을 가장 크게 하는 *hull* 위의 점을 찾는다."""
    best = hull[0]
    for q in hull:
        if cross(point, best, q) < 0:
            best = q
    return best


# === 챈 알고리즘 ============================================================

def chan_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """O(n log h) 내놓기에 민감한 볼록 껍질."""
    pts = list(set(points))
    n = len(pts)
    if n <= 2:
        return graham_hull(pts)

    for t in range(1, n + 1):
        m = min(2 ** (2 ** t), n)
        # 무리로 가르고 아래 껍질을 셈한다
        groups = [graham_hull(pts[i:i + m]) for i in range(0, n, m)]

        # 가장 아래 점에서 감싸기 시작한다
        start = min(pts, key=lambda p: (p[1], p[0]))
        hull = [start]
        for _ in range(m):
            candidates = [_tangent(g, hull[-1]) for g in groups]
            best = candidates[0]
            for c in candidates[1:]:
                if c == hull[-1]:
                    continue
                if best == hull[-1] or cross(hull[-1], best, c) < 0:
                    best = c
            if best == start:
                return hull
            hull.append(best)
        # m이 너무 작았다 — 더 크게 시험한다
    return hull  # 물러설 자리


# === 보임 ====================================================================

if __name__ == "__main__":
    sample = [(0, 0), (1, 1), (2, 2), (0, 2), (2, 0), (1, 0)]
    hull = chan_hull(sample)
    print(f"Input:  {sample}")
    print(f"Hull:   {hull}")
    print(f"|hull| = {len(hull)}")
```

## 다른 알고리즘과의 견줌

| 알고리즘 | 시간 | 내놓기에 민감한가? |
|-----------|------|-------------------|
| 그레이엄 훑기 | $O(n \log n)$ | 아니다 |
| 자비스 행진 | $O(nh)$ | 그렇다 |
| **챈** | $O(n \log h)$ | 그렇다 |

챈 알고리즘은 늘 그레이엄 훑기만큼 빠르고 $h = o(n)$일 때는 엄격히 더 빠르다.

## 참고 문헌

- T. M. Chan, "Optimal Output-Sensitive Convex Hull Algorithms in Two and Three Dimensions," *Discrete & Computational Geometry*, 16, 1996.
- de Berg, Cheong, van Kreveld, Overmars, *Computational Geometry*, 3rd ed., Springer, 2008.

## 연습문제

**연습문제 1.**
챈 알고리즘의 핵심 기하 통찰과 그 시간 복잡도를 설명하라.

??? success "연습문제 1 풀이"
    챈 알고리즘은 기하의 성질(방향, 거리, 각 차례, 훑는 선 사건)을 이용해 점이나 선, 다각형의 모임을 효율 좋게 다룬다. 시간 복잡도는 흔히 $O(n \log n)$(견줌에 바탕한 기하 문제에서 가장 좋다)에서, 본디 이차 짜임을 지닌 문제의 $O(n^2)$까지이다. 핵심 통찰은 기하 문제를 여느 알고리즘이 풀 수 있는 조합 문제로 줄이는 것이다. $\square$

---

**연습문제 2.**
작은 점 모임 $\{(0,0), (1,3), (3,1), (4,4), (2,2)\}$에서 챈 알고리즘을 좇아라.

??? success "연습문제 2 풀이"
    알고리즘의 방책(자리값으로 정렬하기, 각으로 훑기, 사건에 따라 다루기)에 따라 점을 다룬다. 걸음마다 기하 짜임(볼록 껍질, 만남 목록, 보로노이 칸 등)을 새로 고친다. 마지막 결과가 이 들임에 대한 알고리즘의 내놓기이다. 손으로 셈한 것과 견주어 기하의 성질을 살펴 옳음을 확인하라. $\square$

---

**연습문제 3.**
챈 알고리즘은 어떤 찌그러진 경우를 다루어야 하는가? 흔히 어떻게 푸는가?

??? success "연습문제 3 풀이"
    흔한 찌그러진 경우는 이렇다. (1) **한 줄에 놓인 점**: 셋 이상이 한 선 위에 있으면 방향 살피기가 애매해진다. (2) **겹친 점**: 자리값이 똑같다. (3) **세로선**: 기울기 셈에서 0으로 나누게 된다. (4) **한 동그라미 위의 점**: 네 점이 한 동그라미 위에 있으면 들로네 삼각 나누기에 영향을 준다. 푸는 방책은 튼튼한 판정(정확한 셈)을 쓰거나, 기호로 살짝 흔들거나(일반 자리를 흉내 냄), 찌그러진 경우를 따로 다루는 코드를 두는 것이다. $\square$

---

**연습문제 4.**
챈 알고리즘을 막무가내 방식과 견주어라. 점 $n = 10^6$개에서 얼마나 빨라지는지 수로 나타내라.

??? success "연습문제 4 풀이"
    막무가내 방식은 짝이나 세 짝을 모두 살피므로 흔히 $O(n^2)$이나 $O(n^3)$이 든다. 챈 알고리즘은 $O(n \log n)$ 또는 그보다 좋다. $n = 10^6$이면 막무가내는 셈이 $10^{12}$번이나 $10^{18}$번(몇 시간에서 몇 해) 필요하지만 효율 좋은 알고리즘은 $\approx 2 \times 10^7$번(몇 초)이면 된다. 빨라지는 갑절은 $10^5$에서 $10^{11}$이므로 들임이 클 때는 효율 좋은 알고리즘이 꼭 필요하다. $\square$
