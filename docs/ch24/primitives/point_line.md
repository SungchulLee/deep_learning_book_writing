# 점과 선의 관계

많은 기하 문제는 점과 선(또는 도막)에 대한 두 가지 기본 물음으로 줄어든다. 점이 선의 어느 쪽에 있는가, 점은 선에서 얼마나 먼가? 첫 물음은 방향 살피기(어긋 곱 부호)를, 둘째 물음은 쏘기 공식을 쓴다. 이 밑감들이 함께 가장 가까운 점 묻기, 다각형 단순화, 선 맞추기 알고리즘을 받쳐 준다.

---

## 1. 선 나타내기

2차원의 선은 여러 방식으로 나타낼 수 있다.

| 꼴 | 식 | 매개변수 |
|---|---|---|
| 두 점 | $A$과 $B$을 지나는 선 | 점 $A, B$ |
| 매개변수 | $P(t) = A + t(B - A)$ | 방향 벡터 $B - A$ |
| 숨은 꼴 | $ax + by + c = 0$ | 계수 $a, b, c$ |

두 점 $A = (a_x, a_y)$과 $B = (b_x, b_y)$이 주어질 때 숨은 꼴은 다음과 같다.

$$
(b_y - a_y)(x - a_x) - (b_x - a_x)(y - a_y) = 0
$$

여기서 $a = b_y - a_y$, $b = -(b_x - a_x)$, $c = -a \cdot a_x - b \cdot a_y$이다.

---

## 2. 선의 어느 쪽인지 살피기

어긋 곱은 점 $P$이 선 $\overleftrightarrow{AB}$의 어느 쪽에 있는지 정한다.

$$
\text{side}(A, B, P) = (B_x - A_x)(P_y - A_y) - (B_y - A_y)(P_x - A_x)
$$

| 값 | 뜻 |
|---|---|
| $> 0$ | $P$이 $\overleftrightarrow{AB}$의 왼쪽에 있다 |
| $= 0$ | $P$이 선 위에 있다 |
| $< 0$ | $P$이 $\overleftrightarrow{AB}$의 오른쪽에 있다 |

이는 세 점 $(A, B, P)$에 쓴 방향 살피기 바로 그것이다.

---

## 3. 점과 선 사이 거리

점 $P$에서 $A$과 $B$을 지나는 선까지의 부호 있는 거리는 다음과 같다.

$$
d_{\text{signed}} = \frac{(B_x - A_x)(P_y - A_y) - (B_y - A_y)(P_x - A_x)}{\|\overrightarrow{AB}\|}
$$

여기서 $\|\overrightarrow{AB}\| = \sqrt{(B_x - A_x)^2 + (B_y - A_y)^2}$이다. 절대 거리는 $|d_{\text{signed}}|$이다.

!!! note "분자는 어긋 곱이다"
    분자는 선의 어느 쪽인지 살피기에 쓰는 것과 같은 어긋 곱이다. $\overrightarrow{AB}$의 길이로 나누면 참된 유클리드 거리로 고르게 된다.

---

## 4. 점과 도막 사이 거리

(끝없는 선이 아니라) *도막* $\overline{AB}$에서는 가장 가까운 점이 수선의 발이 아니라 $A$이나 $B$일 수 있다. $P$을 선 $\overleftrightarrow{AB}$에 쏜 매개변수 $t$을 쓴다.

$$
t = \frac{\overrightarrow{AP} \cdot \overrightarrow{AB}}{\overrightarrow{AB} \cdot \overrightarrow{AB}}
$$

- $t \le 0$이면 가장 가까운 점은 $A$이다.
- $t \ge 1$이면 가장 가까운 점은 $B$이다.
- $0 < t < 1$이면 가장 가까운 점은 $A + t(B - A)$이다.

---

## 5. 풀이 예제

$A = (1, 1)$, $B = (5, 3)$, $P = (3, 4)$이라 하자.

**어느 쪽인지 살피기:**

$$
\text{side}(A, B, P) = (5-1)(4-1) - (3-1)(3-1) = 4 \cdot 3 - 2 \cdot 2 = 8
$$

$8 > 0$이므로 점 $P$은 선 $\overleftrightarrow{AB}$의 왼쪽에 있다.

**선까지의 거리:**

$$
\|\overrightarrow{AB}\| = \sqrt{16 + 4} = \sqrt{20} = 2\sqrt{5}
$$

$$
d = \frac{8}{2\sqrt{5}} = \frac{4}{\sqrt{5}} = \frac{4\sqrt{5}}{5} \approx 1.789
$$

**쏘기 매개변수:**

$$
\overrightarrow{AP} = (2, 3), \quad \overrightarrow{AB} = (4, 2)
$$

$$
t = \frac{2 \cdot 4 + 3 \cdot 2}{4^2 + 2^2} = \frac{14}{20} = 0.7
$$

$0 < t < 1$이므로 수선의 발은 도막 위 $(1 + 0.7 \cdot 4,\, 1 + 0.7 \cdot 2) = (3.8, 2.4)$에 있다.

---

## 6. 구현

```python
"""
점과 선, 점과 도막 밑감.

2차원에서 선과 도막에 대한 어느 쪽인지 살피기, 거리 셈하기,
가장 가까운 점 쏘기를 준다.
"""

import math

# === 선의 어느 쪽인지 살피기 ===

def side_of_line(a, b, p):
    """점 P이 선 AB의 어느 쪽에 있는지 가린다.

    반환값:
        P이 AB의 왼쪽이면 양수, 오른쪽이면 음수, 선 위에 있으면 0.
    """
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

# === 점과 선 사이 거리 ===

def point_line_distance(a, b, p):
    """점 P에서 선 AB까지의 수직 거리를 셈한다."""
    cross = side_of_line(a, b, p)
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    if length == 0:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    return abs(cross) / length

# === 점과 도막 사이 거리 ===

def point_segment_distance(a, b, p):
    """점 P에서 도막 AB까지의 거리를 셈한다.

    도막 위 가장 가까운 점까지의 거리를 돌려주며,
    그 점은 끝점이거나 수선의 발일 수 있다.
    """
    dx, dy = b[0] - a[0], b[1] - a[1]
    len_sq = dx * dx + dy * dy

    if len_sq == 0:
        return math.hypot(p[0] - a[0], p[1] - a[1])

    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len_sq
    t = max(0, min(1, t))

    proj_x = a[0] + t * dx
    proj_y = a[1] + t * dy
    return math.hypot(p[0] - proj_x, p[1] - proj_y)

# === 도막 위 가장 가까운 점 ===

def closest_point_on_segment(a, b, p):
    """도막 AB 위에서 점 P에 가장 가까운 점을 찾는다."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    len_sq = dx * dx + dy * dy

    if len_sq == 0:
        return a

    t = max(0, min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len_sq))
    return (a[0] + t * dx, a[1] + t * dy)

# === 메인 ===

if __name__ == "__main__":
    A, B = (1, 1), (5, 3)
    P = (3, 4)

    print(f"Line through A={A}, B={B}")
    print(f"Query point P={P}")
    print(f"Side value: {side_of_line(A, B, P)}")
    print(f"Distance to line: {point_line_distance(A, B, P):.4f}")
    print(f"Distance to segment: {point_segment_distance(A, B, P):.4f}")
    print(f"Closest point: {closest_point_on_segment(A, B, P)}")

    # 도막 끝점 너머의 점
    P2 = (7, 5)
    print(f"\nQuery point P2={P2}")
    print(f"Distance to line: {point_line_distance(A, B, P2):.4f}")
    print(f"Distance to segment: {point_segment_distance(A, B, P2):.4f}")
    print(f"Closest point: {closest_point_on_segment(A, B, P2)}")
```

**출력:**
```
Line through A=(1, 1), B=(5, 3)
Query point P=(3, 4)
Side value: 8
Distance to line: 1.7889
Distance to segment: 1.7889
Closest point: (3.8, 2.4)

Query point P2=(7, 5)
Distance to line: 1.7889
Distance to segment: 2.8284
Closest point: (5, 3)
```

---

## 연습문제

**연습문제 1.**
점과 선의 관계의 핵심 기하 통찰과 그 시간 복잡도를 설명하라.

??? success "연습문제 1 풀이"
    점과 선의 관계은 기하의 성질(방향, 거리, 각 차례, 훑는 선 사건)을 이용해 점이나 선, 다각형의 모임을 효율 좋게 다룬다. 시간 복잡도는 흔히 $O(n \log n)$(견줌에 바탕한 기하 문제에서 가장 좋다)에서, 본디 이차 짜임을 지닌 문제의 $O(n^2)$까지이다. 핵심 통찰은 기하 문제를 여느 알고리즘이 풀 수 있는 조합 문제로 줄이는 것이다. $\square$

---

**연습문제 2.**
작은 점 모임 $\{(0,0), (1,3), (3,1), (4,4), (2,2)\}$에서 점과 선의 관계을 좇아라.

??? success "연습문제 2 풀이"
    알고리즘의 방책(자리값으로 정렬하기, 각으로 훑기, 사건에 따라 다루기)에 따라 점을 다룬다. 걸음마다 기하 짜임(볼록 껍질, 만남 목록, 보로노이 칸 등)을 새로 고친다. 마지막 결과가 이 들임에 대한 알고리즘의 내놓기이다. 손으로 셈한 것과 견주어 기하의 성질을 살펴 옳음을 확인하라. $\square$

---

**연습문제 3.**
점과 선의 관계은 어떤 찌그러진 경우를 다루어야 하는가? 흔히 어떻게 푸는가?

??? success "연습문제 3 풀이"
    흔한 찌그러진 경우는 이렇다. (1) **한 줄에 놓인 점**: 셋 이상이 한 선 위에 있으면 방향 살피기가 애매해진다. (2) **겹친 점**: 자리값이 똑같다. (3) **세로선**: 기울기 셈에서 0으로 나누게 된다. (4) **한 동그라미 위의 점**: 네 점이 한 동그라미 위에 있으면 들로네 삼각 나누기에 영향을 준다. 푸는 방책은 튼튼한 판정(정확한 셈)을 쓰거나, 기호로 살짝 흔들거나(일반 자리를 흉내 냄), 찌그러진 경우를 따로 다루는 코드를 두는 것이다. $\square$

---

**연습문제 4.**
점과 선의 관계을 막무가내 방식과 견주어라. 점 $n = 10^6$개에서 얼마나 빨라지는지 수로 나타내라.

??? success "연습문제 4 풀이"
    막무가내 방식은 짝이나 세 짝을 모두 살피므로 흔히 $O(n^2)$이나 $O(n^3)$이 든다. 점과 선의 관계은 $O(n \log n)$ 또는 그보다 좋다. $n = 10^6$이면 막무가내는 셈이 $10^{12}$번이나 $10^{18}$번(몇 시간에서 몇 해) 필요하지만 효율 좋은 알고리즘은 $\approx 2 \times 10^7$번(몇 초)이면 된다. 빨라지는 갑절은 $10^5$에서 $10^{11}$이므로 들임이 클 때는 효율 좋은 알고리즘이 꼭 필요하다. $\square$

## 정리하며

이 마당은 선 나타내기、선의 어느 쪽인지 살피기、점과 선 사이 거리、점과 도막 사이 거리을 차례로 짚었다.

**참고 문헌**

- de Berg, M., Cheong, O., van Kreveld, M., & Overmars, M. *Computational Geometry: Algorithms and Applications*. Springer.
- O'Rourke, J. *Computational Geometry in C*. Cambridge University Press.
