# 가장 가까운 점 짝

판 위에 점 $n$개가 있을 때 유클리드 거리가 가장 짧은 짝을 막무가내로 찾으려면 $\binom{n}{2}$ 짝을 모두 살펴야 하므로 $O(n^2)$ 때가 든다. 나누어 다스리는 길은 $O(n \log n)$을 이루는데, 이는 견줌 바탕 알고리즘의 아래 테두리에 딱 맞는다. 고갱이 어려움은 **아우르는 걸음**에 있으며, 여기서 슬기로운 기하 따짐이 금을 가로지르는 짝의 수를 옭아맨다.

## 문제 서술

$\mathbb{R}^2$ 안의 점 묶음 $P = \{p_1, p_2, \dots, p_n\}$이 주어졌을 때 다음을 찾아라.

$$
\min_{i \ne j} d(p_i, p_j) = \min_{i \ne j} \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}
$$

## 나누어 이기기 알고리즘

**1단계: 정렬.** 모든 점을 $x$-자리표로 정렬한다. $y$-자리표로 정렬한 복사본도 함께 지닌다.

**2단계: 나누기.** 가운뎃 $x$-자리표에서 $P$을 두 반쪽 $P_L$과 $P_R$으로 쪼갠다.

**걸음 3: 다스리기.** $P_L$ 안의 가장 가까운 짝(거리 $\delta_L$)과 $P_R$ 안의 가장 가까운 짝(거리 $\delta_R$)을 되부르며 찾는다. $\delta = \min(\delta_L, \delta_R)$이라 하자.

**걸음 4: 아우르기.** 한 점은 $P_L$에, 다른 점은 $P_R$에 있는 짝 가운데 거리가 $\delta$보다 짧은 것이 있는지 살핀다. 알고리즘이 잘 드는지가 여기서 기하 깨침에 달려 있다.

## 띠 논증

가르는 금에서 거리 $\delta$ 안에 있는 점만 더 가까운 짝을 이룰 수 있다. **띠**를 다음과 같이 매긴다.

$$
S = \{p \in P : |p.x - x_{\text{mid}}| < \delta\}
$$

$S$ 안의 점을 $y$ 자리 값으로 줄 세운다. $S$ 안의 점 $p$마다 $y$ 방향으로 $\delta$ 안에 있는 점과만 견준다.

!!! note "성김 보조정리"
    띠 안의 어떤 점 $p$에 대해서도, $p$을 가운데 둔 $\delta \times 2\delta$ 네모 안에 있는 $S$의 다른 점은 많아야 **7개**다. 그러므로 안쪽 되돌이는 점마다 많아야 7개만 살핀다.

밝히기는 채우기 따짐을 쓴다. $\delta \times 2\delta$ 네모는 $(\delta/2) \times (\delta/2)$ 잔네모 여덟 개로 나눌 수 있다. 잔네모마다 점이 많아야 하나 들어 있으므로(같은 쪽의 두 점은 거리가 적어도 $\delta$이기 때문이다) 네모 안의 다른 점은 많아야 $8 - 1 = 7$개다.

곧 아우르기 단계는 $O(|S|)$ 시간(띠 크기에 한 줄로 비례)이 들며 전체 되돌이 관계식은 다음과 같다:

$$
T(n) = 2T(n/2) + O(n) = O(n \log n)
$$

## 구현

```python
"""
나누어 이기기로 찾는 평면 위 가장 가까운 점 짝.

아우르기 단계에서 띠 자리가 성기다는 것을 써먹어
O(n log n) 시간을 이룬다.
"""

import math

# === 가장 가까운 짝 알고리즘 ===

def closest_pair(points: list[tuple[float, float]]) -> float:
    """가장 가까운 점 짝의 거리 찾기.

    인수:
        points: (x, y) 자리표의 목록.

    반환값:
        아무 두 점 사이의 최소 유클리드 거리.
    """
    def dist(p1: tuple, p2: tuple) -> float:
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _solve(px: list, py: list) -> float:
        n = len(px)
        if n <= 3:
            # 작은 경우는 막무가내로
            best = float('inf')
            for i in range(n):
                for j in range(i + 1, n):
                    best = min(best, dist(px[i], px[j]))
            return best

        mid = n // 2
        mid_x = px[mid][0]

        # y 정렬 차례를 지키며 py를 왼쪽과 오른쪽으로 쪼개기
        pyl = [p for p in py if p[0] <= mid_x]
        pyr = [p for p in py if p[0] > mid_x]

        # 가운뎃점에서 같은 값 다루기
        if len(pyl) > mid:
            excess = len(pyl) - mid
            pyr = [p for p in pyl if p[0] == mid_x][-excess:] + pyr
            pyl = [p for p in pyl if p[0] < mid_x] + \
                  [p for p in pyl if p[0] == mid_x][:-excess]

        dl = _solve(px[:mid], pyl)
        dr = _solve(px[mid:], pyr)
        delta = min(dl, dr)

        # y-자리표로 정렬한 띠 세우기
        strip = [p for p in py if abs(p[0] - mid_x) < delta]

        # 띠 안의 짝 살피기(점마다 많아야 7번 견줌)
        for i in range(len(strip)):
            j = i + 1
            while j < len(strip) and strip[j][1] - strip[i][1] < delta:
                delta = min(delta, dist(strip[i], strip[j]))
                j += 1

        return delta

    px = sorted(points, key=lambda p: p[0])
    py = sorted(points, key=lambda p: p[1])
    return _solve(px, py)


# === 시연 ===

if __name__ == "__main__":
    points = [
        (2.0, 3.0), (12.0, 30.0), (40.0, 50.0),
        (5.0, 1.0), (12.0, 10.0), (3.0, 4.0),
    ]
    result = closest_pair(points)
    print(f"Closest pair distance: {result:.4f}")

    # 막무가내로 확인하기
    best = float('inf')
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = math.sqrt((points[i][0]-points[j][0])**2 +
                          (points[i][1]-points[j][1])**2)
            best = min(best, d)
    print(f"Brute force distance: {best:.4f}")
```

**출력:**

```
Closest pair distance: 1.4142
Brute force distance: 1.4142
```

가장 가까운 짝은 $(2, 3)$과 $(3, 4)$이고 거리는 $\sqrt{2} \approx 1.4142$이다. 나누어 다스리는 길과 막무가내로 하는 길이 같은 답을 낸다.

## 복잡도

| 항목 | 비용 |
|--------|:----:|
| Time   | $O(n \log n)$ |
| 공간 | $O(n)$ |

처음 줄 세우기에 $O(n \log n)$이 든다. 되돌이 식 $T(n) = 2T(n/2) + O(n)$은 으뜸 정리에 따라 $O(n \log n)$으로 풀린다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), 33장: Computational Geometry.
- Shamos, M. I., & Hoey, D. (1975). Closest-point problems. *IEEE Symposium on FOCS*, pp. 151--162.

## 연습문제

**연습문제 1.**
가장 가까운 점 짝의 핵심 생각과 그 시간 복잡도를 설명하여라.

??? success "연습문제 1 풀이"
    가장 가까운 점 짝 찾기는 나누어 다스리기 틀을 쓴다. 문제를 더 작은 잔문제로 쪼개고, 되부르며 풀고, 그 결과를 아우른다. 때 복잡도는 잔문제의 크기와 아우르는 값을 다스리는 되돌이 식이 정한다. 흔히 으뜸 정리나 되부름 나무 살피기로 닫힌 꼴의 복잡도를 얻는다. $\square$

---

**연습문제 2.**
가장 가까운 점 짝의 되돌이 관계식을 쓰고 마스터 정리로 풀어라.

??? success "연습문제 2 풀이"
    되돌이 식은 그 알고리즘이 어떻게 나누는지에 달려 있다(잔문제의 수 $a$, 크기를 줄이는 값 $b$, 아우르는 값 $f(n)$). 으뜸 정리를 쓴다. $f(n)$을 $n^{\log_b a}$과 견주어 어느 갈래인지 가린다. $f(n) = \Theta(n^{\log_b a})$이면(둘째 갈래) $T(n) = \Theta(n^{\log_b a} \log n)$이다. $\square$

---

**연습문제 3.**
가장 가까운 점 짝이 막무가내 방식보다 나은 장면을 설명하여라. 얼마나 빨라지는지 수로 나타내어라.

??? success "연습문제 3 풀이"
    막무가내로 하는 길은 흔히 $O(n^2)$ 이상이 든다. 나누어 다스리는 길은 되부르며 쪼개어 군더더기 셈을 줄이므로 복잡도가 더 낮다. 들임 크기가 $n = 10^6$이면 $O(n^2) = 10^{12}$과 $O(n \log n) = 2 \times 10^7$의 차이는 $50{,}000$ 곱절이다. $\square$

---

**연습문제 4.**
가장 가까운 점 짝의 바탕 경우는 무엇인가? 그것이 알고리즘 전체의 옳음에 어떤 영향을 주는가?

??? success "연습문제 4 풀이"
    밑 자리는 더 나눌 수 없을 만큼 작은 들임을 다룬다(흔히 $n \leq 1$이나 $n \leq 2$). 이때는 옳은 결과를 곧바로 돌려주어야 한다. 밑 자리가 제대로 없으면 되부름이 끝나지 않는다. 밑 자리를 더 크게 잡고($n \leq 10$ 따위) 더 단순한 알고리즘으로 갈아타면 같은 점근 복잡도를 지키면서 되부름 덤을 줄여 참으로 더 빠르게 할 수 있다. $\square$
