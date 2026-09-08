# 거꿀 애커만 함수

계급으로 합치기와 길 줄이기를 쓴 합치기-찾기는 연산마다 고르게 친 값이 $O(\alpha(n))$이며, 여기서 $\alpha$은 **거꿀 애커만 함수**이다. 이 함수는 아주 느리게 자라서 $2^{2^{2^{65536}}}$까지의 어떤 $n$에 대해서도 $\alpha(n) \leq 4$이다. 이는 볼 수 있는 우주의 원자 수를 훌쩍 넘는 수이다. 실전에서는 $\alpha(n)$이 상수나 마찬가지여서 합치기-찾기 연산은 사실상 고르게 친 $O(1)$이 된다.

---

## 1. 애커만 함수

애커만 함수 $A(i, j)$은 되돌이로 정한다:

$$
A(i, j) = \begin{cases}
j + 1 & \text{if } i = 0 \\
A(i-1, 1) & \text{if } i > 0 \text{ and } j = 0 \\
A(i-1, A(i, j-1)) & \text{if } i > 0 \text{ and } j > 0
\end{cases}
$$

이 함수는 엄청나게 빨리 자란다. 몇몇 값이 그 터짐을 보여 준다:

| $i$ | $A(i, 1)$ | 자라는 무늬 |
|-----|-----------|----------------|
| 0 | 2 | 덧셈: $j + 1$ |
| 1 | 3 | $2j + 3$(선형) |
| 2 | 7 | $2^{j+3} - 3$(지수) |
| 3 | 61 | 2의 탑(테트레이션) |
| 4 | $2^{2^{2^{65536}}} - 3$ | 헤아릴 수 없음 |

---

## 2. 거꿀 애커만 함수

거꿀 애커만 함수 $\alpha(n)$은 다음과 같이 정한다:

$$
\alpha(n) = \min\{i \geq 1 : A(i, \lfloor \log_2 n \rfloor) \geq \log_2 n\}
$$

헐겁게 말하면 $\alpha(n)$은 결과가 문턱값 아래로 떨어질 때까지 $\log^*$(되풀이 로그)을 몇 번 씌워야 하는지이다. $A$이 워낙 빨리 자라므로 $\alpha$은 믿기 어려울 만큼 느리게 자란다:

| $n$ | $\alpha(n)$ |
|-----|-------------|
| $1$에서 $2$ | $1$ |
| $3$에서 $7$ | $2$ |
| $8$에서 $2047$ | $3$ |
| $2048$에서 $A(3,1) \approx 10^{19728}$ | $4$ |

---

## 3. 합치기-찾기 살피기에서 하는 일

타잔(1975)은 (계급으로 합치기와 길 줄이기를 쓸 때) 원소 $n$개에 대한 합치기-찾기 연산 $m$번이 모두 $O(m \cdot \alpha(n))$ 시간이 듦을 증명했다. 그 증명은 애커만 함수에 바탕을 둔 퍼텐셜 함수 논증으로 길 줄이기가 시간이 갈수록 나무 짜임을 얼마나 납작하게 만드는지 좇는다.

핵심 통찰은 길 줄이기가 뒤이은 찾기 연산을 싸게 만든다는 것이다. 찾기가 끝나면 그 길 위의 모든 마디가 뿌리를 곧바로 가리킨다. 애커만 함수는 이 납작해짐이 여러 연산에 걸쳐 나아가는 빠르기를 담아낸다.

!!! note "팽팽한 아래 경계"
    프레드먼과 색스(1989)는 가리개에 바탕을 둔 어떤 합치기-찾기 구현에도 맞아떨어지는 $\Omega(m \cdot \alpha(n))$ 아래 경계를 증명했다. 곧 이 셈 모형 안에서는 고르게 친 $O(\alpha(n))$ 경계를 더 낫게 할 수 없다는 뜻이다.

---

## 4. 구현

```python
"""
길 줄이기와 계급으로 합치기를 쓴 합치기-찾기.

연산마다 고르게 친 O(alpha(n))을 이루며, 여기서 alpha은
거꿀 애커만 함수이다. 또한 다음 함수도 담았다:
거꿀 애커만 함수이다. 보이기 위해 alpha(n)을 셈하는 함수도 담았다.
"""

import math

# === 애커만 함수(묶인 셈하기) ===

def ackermann(i: int, j: int, limit: int = 100000) -> int:
    """넘침을 막는 안전 한계를 두고 A(i, j) 셈하기.

    결과를, 셈이 한계를 넘을 것 같으면 그 한계를 돌려준다.
    """
    if i == 0:
        return j + 1
    if j == 0:
        if i <= 4:
            return ackermann(i - 1, 1, limit)
        return limit
    inner = ackermann(i, j - 1, limit)
    if inner >= limit:
        return limit
    return ackermann(i - 1, inner, limit)

def inverse_ackermann(n: int) -> int:
    """A(i, i) >= n인 가장 작은 i을 찾아 alpha(n) 셈하기."""
    if n <= 2:
        return 0
    for i in range(1, 100):
        if ackermann(i, i) >= n:
            return i
    return 99

# === 길 줄이기와 계급으로 합치기를 쓴 합치기-찾기 ===

class UnionFind:
    """연산이 고르게 친 O(alpha(n))인 합치기-찾기 자료 짜임."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.num_components = n

    def find(self, x: int) -> int:
        """온전한 길 줄이기로 x의 뿌리 찾기."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> bool:
        """계급으로 합치기. 합침이 일어났으면 True를 돌려준다."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.num_components -= 1
        return True

    def connected(self, a: int, b: int) -> bool:
        """a과 b이 같은 조각에 있는지 살핀다."""
        return self.find(a) == self.find(b)

# === 시연 ===

if __name__ == "__main__":
    # 여러 값에 대한 alpha(n) 보이기
    print("Inverse Ackermann function values:")
    for n in [1, 2, 4, 8, 16, 64, 1000, 10**6, 10**12]:
        alpha = inverse_ackermann(n)
        print(f"  alpha({n:>15,}) = {alpha}")

    print()

    # 합치기-찾기 보이기
    uf = UnionFind(8)
    operations = [(0, 1), (2, 3), (4, 5), (6, 7),
                  (0, 2), (4, 6), (0, 4)]
    for a, b in operations:
        merged = uf.union(a, b)
        print(f"union({a},{b}) -> merged={merged}, "
              f"components={uf.num_components}")

    print()
    print(f"0 and 7 connected: {uf.connected(0, 7)}")
    print(f"Parent array after path compression: {uf.parent}")
```

**출력:**
```
Inverse Ackermann function values:
  alpha(              1) = 0
  alpha(              2) = 0
  alpha(              4) = 2
  alpha(              8) = 2
  alpha(             16) = 3
  alpha(             64) = 3
  alpha(          1,000) = 3
  alpha(      1,000,000) = 3
  alpha(  1,000,000,000,000) = 4

union(0,1) -> merged=True, components=7
union(2,3) -> merged=True, components=6
union(4,5) -> merged=True, components=5
union(6,7) -> merged=True, components=4
union(0,2) -> merged=True, components=3
union(4,6) -> merged=True, components=2
union(0,4) -> merged=True, components=1

0 and 7 connected: True
Parent array after path compression: [0, 0, 0, 2, 0, 4, 4, 6]
```

---

## 5. 실전에서의 뜻

실전에서 다룰 만한 들임 크기라면 $\alpha(n) \leq 4$이므로 합치기-찾기 연산은 사실상 $O(1)$이다. 그래서 합치기-찾기는 쓸 수 있는 자료 짜임 가운데 가장 효율적인 축에 든다:

| 연산 수($m$) | 원소 수($n$) | 전체 시간 |
|-------------------|----------------|------------|
| $10^6$ | $10^6$ | $\leq 4 \times 10^6$ |
| $10^9$ | $10^9$ | $\leq 4 \times 10^9$ |
| 아무거나 | 실전에서 다룰 만한 아무 $n$ | $\leq 4m$ |

---

## 연습문제

**연습문제 1.**
애커만 함수 $A(m, n)$을 정의하고 왜 그토록 빨리 자라는지 설명하여라.

??? success "연습문제 1 풀이"
    $A(0, n) = n + 1$이고, $m > 0$에 대해 $A(m, 0) = A(m-1, 1)$이며, $m, n > 0$에 대해 $A(m, n) = A(m-1, A(m, n-1))$이다. 층 $m$마다 앞 층을 되풀이하므로 이 함수는 빠르게 자란다. 곧 $A(1, n) = n + 2$, $A(2, n) = 2n + 3$, $A(3, n) \approx 2^{n+3} - 3$이고, $A(4, n)$은 높이 $n$의 2의 탑이며, $A(5, n)$은 유한한 높이의 어떤 탑도 넘어선다. 겹쳐진 짜임 때문에 층마다 앞 층의 초지수 자람을 되풀이해 쓰게 된다. $\square$

---

**연습문제 2.**
거꿀 애커만 함수 $\alpha(n)$을 정의하고 왜 "실전에서는 상수"인지 설명하여라.

??? success "연습문제 2 풀이"
    $\alpha(n) = \min\{k : A(k, k) \geq n\}$이다. $A(4, 4)$이 (2를 65536번 쌓은 탑인) 어마어마한 수이므로 실전에서 떠올릴 만한 모든 $n$에 대해($2^{2^{2^{65536}}}$쯤까지) $\alpha(n) \leq 4$이다. 참 셈하기에서 나올 수 있는 어떤 $n$에 대해서도(우주의 원자 수인 $n = 10^{80}$이라 해도) $\alpha(n) \leq 4$이다. 그래서 $O(\alpha(n))$을 "실전에서는 $O(1)$"로 여긴다. $\square$

---

**연습문제 3.**
합치기-찾기를 살필 때 거꿀 애커만 함수가 왜 나타나는가?

??? success "연습문제 3 풀이"
    (계급으로 합치기와 길 줄이기를 쓸 때) 원소 $n$개에 대한 합치기-찾기 연산 $m$번의 고르게 친 복잡도는 $O(m \cdot \alpha(n))$이다. 타잔은 계급과 길 줄이기 층에 바탕을 둔 퍼텐셜 함수를 세워 이를 증명했다. 길 줄이기가 애커만 자람의 역을 비추는 계급 늘어놓음을 만들기 때문에 거꿀 애커만 함수가 나타난다. 곧 한 마디가 줄어들 수 있는 "층"의 수가 $\alpha(n)$으로 묶인다. 이는 팽팽한 경계이다. 곧 $\Omega(m \cdot \alpha(n))$ 시간이 드는 연산 늘어놓음이 있다. $\square$

---

**연습문제 4.**
$\alpha(n)$을 $\log^* n$(되풀이 로그)과 견주어라. 어느 쪽이 더 작은가?

??? success "연습문제 4 풀이"
    $\log^* n$은 $\leq 1$에 이를 때까지 $\log_2$을 몇 번 씌우는지이다. 이를테면 $\log^*(2^{65536}) = 5$이다. 모든 $n$에 대해 $\alpha(n) \leq \log^*(n)$이고 $\alpha$이 엄격히 더 느리게 자란다. $n = 2^{65536}$이면 $\log^* n = 5$이지만 $\alpha(n) = 4$이다. 실제 들임에서는 둘 다 "실전에서 상수"이지만 이론으로는 $\alpha$이 더 느리게 자란다. 예전 살피기에서는 타잔의 더 팽팽한 $O(m \cdot \alpha(n))$ 경계가 나오기 전에 합치기-찾기가 $O(m \log^* n)$임을 보였다. $\square$

## 정리하며

이 마당은 애커만 함수、거꿀 애커만 함수、합치기-찾기 살피기에서 하는 일、구현을 차례로 짚었다.

**참고 문헌**

- Tarjan, R. E. (1975). Efficiency of a good but not linear set union algorithm. *Journal of the ACM*, 22(2), 215-225.
- Fredman, M., & Saks, M. (1989). The cell probe complexity of dynamic data structures. *Proceedings of STOC*, 345-354.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 19장. MIT Press.
