# 나누어 이기기로 얻는 볼록 껍질

여러 셈 기하 문제는 점 모음의 바깥 테두리를 가려내는 데서 시작한다. 판에 못을 박고 그 둘레에 고무줄을 두르는 모습을 그려 보라. 그 고무줄이 이루는 꼴이 **볼록 껍질**이다. 이 짜임은 부딪힘 알아내기, 그림 다루기, 지리 앎 체계에 나타난다. 이 쪽에서는 나누어 이기기 볼록 껍질 알고리즘을 펼치고, 짜기가 더 단순하면서 가까이 이어진 앤드루의 단조 사슬과 견준다.

## 정의

The **convex hull** of a point set $P \subset \mathbb{R}^2$ is the smallest convex set containing $P$. A set $S$ is convex if for every pair of points $a, b \in S$, the line segment $\overline{ab}$ lies entirely within $S$. The boundary of the convex hull is a convex polygon whose vertices are a subset of $P$.

## 벡터곱 시험

아래 두 알고리즘 모두 차례가 있는 점 셋이 왼쪽으로 도는지, 오른쪽으로 도는지, 한 줄에 있는지 정하는 데 **벡터곱**에 기댄다. 점 $O$, $A$, $B$에 대해 다음을 정한다:

$$
\text{cross}(O, A, B) = (A_x - O_x)(B_y - O_y) - (A_y - O_y)(B_x - O_x)
$$

This quantity equals the signed area of the parallelogram spanned by vectors $\overrightarrow{OA}$ and $\overrightarrow{OB}$:

- **Positive**: $O \to A \to B$ makes a left (counterclockwise) turn.
- **Negative**: $O \to A \to B$ makes a right (clockwise) turn.
- **0**: 세 점이 한 줄에 있다.

## 나누어 이기기 알고리즘

고전 나누어 이기기 방식은 네 단계로 나아간다.

**Step 1 -- Sort.** Sort all $n$ points by $x$-coordinate, breaking ties by $y$-coordinate. This takes $O(n \log n)$ time and needs to be done only once.

**Step 2 -- Divide.** Split the sorted array at the median into a left half $P_L$ (indices $1$ to $\lfloor n/2 \rfloor$) and a right half $P_R$ (indices $\lfloor n/2 \rfloor + 1$ to $n$).

**3단계 — 이기기.** 볼록 껍질 $H_L$과 $H_R$을 되돌이로 셈한다. 바탕 경우는 점이 하나, 둘, 셋인 모음이며 그 껍질은 아무것도 아니다.

**4단계 — 어울리기.** 두 껍질 사이의 **위 접선**과 **아래 접선**을 찾아 $H_L$과 $H_R$을 껍질 하나로 아우른다. 접선을 찾는 절차는 다음과 같다:

1. $H_L$의 가장 오른쪽 점 $p$과 $H_R$의 가장 왼쪽 점 $q$에서 시작한다.
2. **Upper tangent**: While the line $\overline{pq}$ is not tangent to both hulls, repeatedly move $p$ counterclockwise around $H_L$ (as long as the cross product shows a left turn with the next vertex) and move $q$ clockwise around $H_R$ (as long as the cross product shows a right turn with the next vertex).
3. **아래 접선**: 대칭인 절차를 써서 $p$은 시계 방향으로, $q$은 반시계 방향으로 옮긴다.
4. $H_L$의 아래 접점부터 위 접점까지의 테두리 토막을 잇고, 이어서 $H_R$의 위 접점부터 아래 접점까지의 테두리 토막을 잇는다.

접선을 따라 걷는 동안 꼭짓점마다 많아야 한 번 들르므로 어울리기는 $O(n)$ 시간이 든다. 전체 되돌이 관계식은 다음과 같다:

$$
T(n) = 2T(n/2) + O(n) = O(n \log n)
$$

## 앤드루의 단조 사슬

Andrew's monotone chain achieves the same $O(n \log n)$ bound through a simpler implementation. Rather than recursively merging two hulls, it builds the **upper hull** and **lower hull** independently by scanning sorted points from left to right (lower hull) and right to left (upper hull). Each scan maintains a stack and uses the cross product test to discard points that would create a non-convex turn. The two half-hulls are then concatenated to form the complete hull.

이 방식은 나누어 이기기 전략과 가까이 이어져 있다. 곧 정렬 단계를 함께 쓰고, 반쪽 껍질을 세우는 일이 나누어 이기기의 어울리기가 위아래 테두리를 세우는 방식을 그대로 본뜬다. 단조 사슬은 시간 복잡도는 그대로면서 되돌이 어울리기의 잔손질을 피하므로 실전에서 대개 더 낫게 여긴다.

```python
"""
앤드루의 단조 사슬 알고리즘으로 얻는 볼록 껍질.

x-자리표로 정렬한 점을 훑으며 위 껍질과 아래 껍질을 따로 세워
<= 0을 써서
벡터곱 살피기에서 한 줄에 있는 테두리 점을 뺀다.
넣으려면 < 0으로 바꿔라.
"""

# === 벡터곱 ===

def cross(o: tuple, a: tuple, b: tuple) -> float:
    """벡터 OA와 OB의 벡터곱 셈하기.

    반환값:
        O->A->B가 반시계 방향(왼쪽으로 돎)이면 양수,
        시계 방향(오른쪽으로 돎)이면 음수, 한 줄에 있으면 0이다.
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


# === 점이 껍질 안에 있는지 시험 ===

def point_in_hull(p: tuple, hull_pts: list[tuple]) -> bool:
    """점 p이 볼록 껍질 안이나 위에 있는지 살피기.

    껍질의 꼭짓점은 반시계 방향이어야 한다. 이 시험은
    p이 모든 방향 변의 왼쪽에 있는지 확인한다.

    인수:
        p: 2차원 점 (x, y).
        hull_pts: 반시계 방향으로 늘어놓은 볼록 껍질의 꼭짓점.

    반환값:
        p이 껍질 안이나 테두리 위에 있으면 True.
    """
    n = len(hull_pts)
    for i in range(n):
        if cross(hull_pts[i], hull_pts[(i + 1) % n], p) < 0:
            return False
    return True


# === 볼록 껍질 ===

def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """2차원 점 모음의 볼록 껍질 셈하기.

    다루기 앞에 겹치는 것을 없앤다. 한 줄에 있는 테두리
    점은 뺀다(껍질 다각형의 꼭짓점만 남긴다).

    인수:
        points: (x, y) 자리표의 목록.

    반환값:
        반시계 방향으로 늘어놓은 볼록 껍질의 꼭짓점.
    """
    # 겹치는 것을 없애고 x으로 정렬(같으면 y으로)
    points = sorted(set(points))
    if len(points) <= 1:
        return list(points)

    # 아래 껍질 세우기(왼쪽에서 오른쪽)
    lower: list[tuple[float, float]] = []
    for p in points:
        # 왼쪽으로 돌지 않으면 마지막 점 꺼내기(<=0이면 한 줄에 있는 것도 뺀다)
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # 위 껍질 세우기(오른쪽에서 왼쪽)
    upper: list[tuple[float, float]] = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # 겹치는 끝점을 빼고 이어 붙이기
    return lower[:-1] + upper[:-1]


# === 시연 ===

if __name__ == "__main__":
    pts = [(0, 0), (1, 1), (2, 2), (0, 2), (2, 0), (1, 0), (1, 3)]
    hull = convex_hull(pts)
    print(f"Points: {pts}")
    print(f"Hull vertices: {hull}")
    print(f"Number of hull vertices: {len(hull)}")

    all_inside = all(point_in_hull(p, hull) for p in pts)
    print(f"All points inside hull: {all_inside}")
```

**출력:**

```
Points: [(0, 0), (1, 1), (2, 2), (0, 2), (2, 0), (1, 0), (1, 3)]
Hull vertices: [(0, 0), (1, 0), (2, 0), (2, 2), (1, 3), (0, 2)]
Number of hull vertices: 6
All points inside hull: True
```

안쪽 점 $(1, 1)$과 한 줄에 있는 점 $(2, 2)$은 껍질의 꼭짓점이 아니다. 꼭짓점 여섯 개가 들임 점 일곱 개를 모두 감싸는 볼록 다각형을 이룬다.

## 복잡도

| 항목 | 비용 |
|--------|:----:|
| Time   | $O(n \log n)$ |
| 공간 | $O(n)$ |

The sorting step dominates at $O(n \log n)$. The hull construction itself runs in $O(n)$ amortized time: each point enters the stack exactly once and is popped at most once, so the total number of push and pop operations across the entire scan is at most $2n$.

## 아래 한계

Having established an $O(n \log n)$ algorithm, a natural question is whether any comparison-based algorithm can do better. The answer is no.

Computing the convex hull requires $\Omega(n \log n)$ time in the comparison model. The proof uses a reduction from sorting. Given $n$ numbers $x_1, \dots, x_n$, map each to the point $(x_i, x_i^2)$ on the parabola $y = x^2$. Because a parabola is strictly convex, every mapped point is a vertex of the hull, and the hull visits them in sorted order of $x$. Any convex hull algorithm therefore sorts $n$ numbers, which requires $\Omega(n \log n)$ comparisons.

## 참고 문헌

- Andrew, A. M. (1979). Another efficient algorithm for convex hulls in two dimensions. *Information Processing Letters*, 9(5), 216--219.
- Preparata, F. P. & Shamos, M. I. (1985). *Computational Geometry: An Introduction*. Springer.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), 33장: Computational Geometry.

## 연습문제

**연습문제 1.**
나누어 이기기로 얻는 볼록 껍질의 핵심 생각과 그 시간 복잡도를 설명하여라.

??? success "연습문제 1 풀이"
    Convex Hull via Divide and Conquer applies the divide-and-conquer paradigm: split the problem into smaller subproblems, solve them recursively, and combine the results. The time complexity is determined by the recurrence relation governing the subproblem sizes and the combination cost. The Master Theorem or recursion tree analysis typically gives the closed-form complexity. $\square$

---

**연습문제 2.**
나누어 이기기로 얻는 볼록 껍질의 되돌이 관계식을 쓰고 마스터 정리로 풀어라.

??? success "연습문제 2 풀이"
    The recurrence depends on the specific algorithm's division strategy (number of subproblems $a$, size reduction factor $b$, and combination cost $f(n)$). Apply the Master Theorem: compare $f(n)$ with $n^{\log_b a}$ to determine which case applies. If $f(n) = \Theta(n^{\log_b a})$ (case 2), $T(n) = \Theta(n^{\log_b a} \log n)$. $\square$

---

**연습문제 3.**
나누어 이기기로 얻는 볼록 껍질이 막무가내 방식보다 나은 장면을 설명하여라. 얼마나 빨라지는지 수로 나타내어라.

??? success "연습문제 3 풀이"
    The brute-force approach typically runs in $O(n^2)$ or worse. The divide-and-conquer approach achieves a lower complexity by reducing redundant computation through recursive decomposition. For input size $n = 10^6$, the difference between $O(n^2) = 10^{12}$ and $O(n \log n) = 2 \times 10^7$ operations is a factor of $50{,}000$. $\square$

---

**연습문제 4.**
나누어 이기기로 얻는 볼록 껍질의 바탕 경우는 무엇인가? 그것이 알고리즘 전체의 옳음에 어떤 영향을 주는가?

??? success "연습문제 4 풀이"
    Base cases handle inputs too small to subdivide further (typically $n \leq 1$ or $n \leq 2$). They must return correct results directly. Without proper base cases, the recursion never terminates. Choosing a larger base case (e.g., $n \leq 10$) and switching to a simpler algorithm can improve practical performance by reducing recursion overhead while maintaining the same asymptotic complexity. $\square$
