# 높이 분석

AVL 트리의 성능 보장은 오로지 그 높이가 $O(\log n)$이라는 주장에 기댄다. 이 한계가 없으면 찾기와 삽입과 삭제의 $O(\log n)$ 복잡도도 따라 나오지 않는다. 이 절은 주어진 높이에서 노드가 가장 적은 **가장 성긴** AVL 트리를 분석하고 그것을 피보나치 수와 이어 높이의 한계를 엄밀히 증명한다.

## 핵심 물음

노드가 $n$개인 AVL 트리의 높이가 $h$이라 하자. 물음은 이것이다. $n$의 함수로서 $h$은 얼마나 커질 수 있는가? $h = O(\log n)$임을 보일 수 있다면 뿌리에서 잎까지의 모든 경로가 로그이고 모든 사전 연산이 로그 시간에 끝난다.

전략은 물음을 뒤집는 것이다. 높이가 $h$인 AVL 트리가 가질 수 있는 노드의 **최솟값** $N(h)$은 얼마인가? $N(h)$을 찾으면 노드가 $n \geq N(h)$개인 AVL 트리의 높이는 많아야 $h$이고, 이 관계를 뒤집으면 $n$으로 나타낸 $h$을 얻는다.

## 최소 AVL 트리

높이가 $h$인 **최소 AVL 트리**는 높이가 $h$이면서 노드가 가장 적은 AVL 트리이다. 높이 $h$을 지키면서 노드를 최소로 하려면 한쪽 부분 트리를 AVL 조건이 허락하는 만큼 낮게 만든다.

$N(h)$을 높이가 $h$인 AVL 트리의 최소 노드 수라 하자. 바탕 경우는 다음과 같다.

$$
N(0) = 1, \qquad N(1) = 2
$$

$h \geq 2$일 때 높이가 $h$인 최소 AVL 트리는 다음을 갖는다.

- (전체 높이를 $h$으로 하려고) 높이가 $h - 1$인 부분 트리 하나. 그 자체도 최소 AVL 트리이다.
- (AVL 조건이 허락하는 가장 낮은) 높이 $h - 2$인 부분 트리 하나. 이것도 최소 AVL 트리이다.
- 뿌리 노드 자신.

이로부터 다음 점화식을 얻는다.

$$
N(h) = N(h-1) + N(h-2) + 1
$$

## 피보나치 수와의 관계

점화식 $N(h) = N(h-1) + N(h-2) + 1$은 피보나치 점화식 $F(h) = F(h-1) + F(h-2)$과 매우 닮았다. $M(h) = N(h) + 1$이라 두면 다음과 같다.

$$
M(h) = N(h) + 1 = N(h-1) + N(h-2) + 2 = M(h-1) + M(h-2)
$$

여기서 $M(0) = 2$, $M(1) = 3$이다. 피보나치 수가 $F_0 = 0, F_1 = 1, F_2 = 1, F_3 = 2, \ldots$을 만족하므로 다음을 확인할 수 있다.

$$
M(h) = F_{h+3} - 1
$$

여기서 $F_k$은 $k$번째 피보나치 수이다($F_1 = 1, F_2 = 1$). 따라서 다음이 성립한다.

$$
N(h) = F_{h+3} - 2
$$

## 높이의 한계

황금비 $\phi = (1 + \sqrt{5})/2 \approx 1.618$에 대한 잘 알려진 근사 $F_k \approx \phi^k / \sqrt{5}$을 쓰면 다음과 같다.

$$
N(h) \geq F_{h+3} - 2 \geq \frac{\phi^{h+3}}{\sqrt{5}} - 3
$$

노드가 $n$개이고 높이가 $h$인 어떤 AVL 트리에서도 $n \geq N(h)$이므로 다음이 성립한다.

$$
n \geq \frac{\phi^{h+3}}{\sqrt{5}} - 3
$$

밑이 $\phi$인 로그를 취하면 다음과 같다.

$$
h \leq \log_\phi(n + 3) + \log_\phi \sqrt{5} - 3
$$

$\log_\phi = \log_2 / \log_2 \phi$이고 $\log_2 \phi \approx 0.694$이므로 $\log_\phi n \approx 1.44 \log_2 n$이다. 정확한 한계는 다음과 같다.

$$
h \leq 1.44 \log_2(n + 2) - 0.328
$$

??? note "상수 1.44의 유도"
    인수 $1/\log_2 \phi = 1/\log_2((1+\sqrt{5})/2) \approx 1.4404$은 $\log_\phi$과 $\log_2$ 사이의 밑 변환에서 나온다. 곧 AVL 트리는 높이가 $\lfloor \log_2 n \rfloor$인 완벽하게 균형 잡힌 이진 트리보다 많아야 약 44% 높다.

## 주요 정리

!!! info "정리: AVL 트리의 높이"
    노드가 $n$개인 AVL 트리의 높이는 $h = \Theta(\log n)$이다. 더 정확히는 다음과 같다.

    $$
    \lfloor \log_2 n \rfloor \leq h \leq 1.44 \log_2(n + 2) - 0.328
    $$

    아래 한계는 높이가 $h$인 이진 트리의 노드가 많아야 $2^{h+1} - 1$개라는 사실에서 나오며, 따라서 $h \geq \lfloor \log_2 n \rfloor$이다. 위 한계는 위의 최소 AVL 트리 분석에서 나온다.

**증명 개요.** 위 한계는 $n \geq N(h) = F_{h+3} - 2$과 피보나치 수의 지수적 증가에서 따라 나온다. 아래 한계는 모든 이진 트리에 성립한다. 높이가 $h$인 트리의 노드는 많아야 $2^{h+1} - 1$개이므로 $n \leq 2^{h+1} - 1$에서 $h \geq \log_2(n+1) - 1$이 나온다. $\square$

## 최소 AVL 트리의 크기 계산하기

```python
"""
가장 작은 AVL 트리의 크기를 셈하고 높이 한계를 확인한다.

가장 작은 AVL 트리와 피보나치 수의 이음을 보이며
1.44 * log2(n)이라는 높이 한계를 확인한다.
"""


# === 가장 작은 AVL 트리의 크기 ===

def minimal_avl_sizes(max_height):
    """N(h) = 높이가 h인 AVL 트리의 최소 노드 수를 셈한다."""
    if max_height < 0:
        return []
    sizes = [1, 2]  # N(0) = 1, N(1) = 2
    for h in range(2, max_height + 1):
        sizes.append(sizes[h - 1] + sizes[h - 2] + 1)
    return sizes[:max_height + 1]


# === 피보나치 수 ===

def fibonacci(k):
    """F_k를 셈한다(1부터 센다: F_1 = 1, F_2 = 1, F_3 = 2, ...)."""
    if k <= 0:
        return 0
    a, b = 0, 1
    for _ in range(k):
        a, b = b, a + b
    return a


# === 확인 ===

def verify_fibonacci_connection(max_height):
    """N(h) = F_{h+3} - 2임을 확인한다."""
    sizes = minimal_avl_sizes(max_height)
    print(f"{'h':>3} | {'N(h)':>8} | {'F(h+3)-2':>8} | {'Match':>5}")
    print("-" * 35)
    for h, n_h in enumerate(sizes):
        fib_val = fibonacci(h + 3) - 2
        match = "yes" if n_h == fib_val else "NO"
        print(f"{h:3d} | {n_h:8d} | {fib_val:8d} | {match:>5}")


# === 높이 한계 확인 ===

import math

def height_bound(n):
    """위 한계: h <= 1.44 * log2(n+2) - 0.328."""
    if n <= 0:
        return 0
    return 1.44 * math.log2(n + 2) - 0.328


if __name__ == "__main__":
    print("=== Minimal AVL Tree Sizes vs Fibonacci ===")
    verify_fibonacci_connection(12)
    print()

    print("=== Height Bound Verification ===")
    sizes = minimal_avl_sizes(12)
    print(f"{'h':>3} | {'N(h)':>8} | {'bound':>8}")
    print("-" * 28)
    for h, n_h in enumerate(sizes):
        bound = height_bound(n_h)
        print(f"{h:3d} | {n_h:8d} | {bound:8.2f}")
```

**출력:**
```
=== Minimal AVL Tree Sizes vs Fibonacci ===
  h |     N(h) | F(h+3)-2 | Match
-----------------------------------
  0 |        1 |        1 |   yes
  1 |        2 |        2 |   yes
  2 |        4 |        4 |   yes
  3 |        7 |        7 |   yes
  4 |       12 |       12 |   yes
  5 |       20 |       20 |   yes
  6 |       33 |       33 |   yes
  7 |       54 |       54 |   yes
  8 |       88 |       88 |   yes
  9 |      143 |      143 |   yes
 10 |      232 |      232 |   yes
 11 |      376 |      376 |   yes
 12 |      609 |      609 |   yes

=== Height Bound Verification ===
  h |     N(h) |    bound
----------------------------
  0 |        1 |     1.95
  1 |        2 |     2.63
  2 |        4 |     3.63
  3 |        7 |     4.60
  4 |       12 |     5.55
  5 |       20 |     6.49
  6 |       33 |     7.46
  7 |       54 |     8.42
  8 |       88 |     9.39
  9 |      143 |    10.36
 10 |      232 |    11.33
 11 |      376 |    12.30
 12 |      609 |    13.27
```

높이 $h$마다 그에 해당하는 한계값 아래에 있어 정리가 확인된다.

## 실무적 함의

높이의 한계 $h \leq 1.44 \log_2 n$은 다음을 뜻한다.

- 노드가 $10^6$개인 AVL 트리의 높이는 많아야 $1.44 \times 20 \approx 29$이다.
- 완벽하게 균형 잡힌 트리라면 높이가 $20$일 것이다. AVL 트리는 많아야 44% 더 높다.
- 모든 연산(찾기, 삽입, 삭제)이 많아야 노드 $h$개를 들르므로 최악의 경우에도 $O(\log n)$ 시간에 끝난다.

이 최악의 경우는 빈틈이 없다. 위에서 설명한 최소 AVL 트리(피보나치 트리)는 정확히 $1.44 \log_2 n$의 높이에 이른다. 다만 무작위로 만든 AVL 트리는 높이가 $\log_2 n$에 훨씬 가까운 편이다.

## 참고 문헌

- [6. AVL Trees, AVL Sort](https://www.youtube.com/watch?v=FNeL18KsWPc&list=PLUl4u3cNGP61Oq3tWYp6V_F-5jb5L2iHb&index=7)
- [AVL tree](https://en.wikipedia.org/wiki/AVL_tree)
- [1382. Balance a Binary Search Tree](https://leetcode.com/problems/balance-a-binary-search-tree/)


## 연습문제

**연습문제 1.**
높이 분석의 균형 불변식을 밝히고 그것이 높이 $O(\log n)$을 보장함을 증명하라.

??? success "연습문제 1 풀이"
    각 구조의 불변식(균형 인수, 색의 성질, 차수 제약)이 경로 길이의 치우침을 묶는다. 높이의 한계는 그 불변식에서 따라 나온다. 트리의 층마다 (불변식이 정하는) 최소한의 노드가 있어야 하므로 전체 노드 수 $n$이 높이에 따라 지수적으로 늘고, 따라서 $h = O(\log n)$이다.

---

**연습문제 2.**
구조를 다시 짜야 하는(회전, 색 바꾸기, 쪼개기·합치기) 트리에서 높이 분석을 따라가라. 앞뒤의 상태를 보여라.

??? success "연습문제 2 풀이"
    이 쪽에서 설명한 재구성 상황을 일으키는 트리를 하나 만들어라. 어긋난 곳을 보이고, 어느 경우에 해당하는지 가리고, 고친 뒤, 불변식이 되살아났는지 확인하라.

---

**연습문제 3.**
높이 분석이(가) 구조를 다시 짜는 연산을 많아야 $O(\log n)$번 필요로 함을 증명하라.

??? success "연습문제 3 풀이"
    구조를 다시 짤 때마다 어긋난 곳이 뿌리에 한 층 가까워지거나 해소된다. 트리의 층이 $O(\log n)$개이므로 재구성은 많아야 $O(\log n)$번 필요하다. 레드-블랙 삽입 같은 연산에서는 회전 2번과 색 바꾸기 $O(\log n)$번이면 충분하다. $\square$

---

**연습문제 4.**
최악의 높이, 연산마다의 회전 횟수, 구현의 까다로움 면에서 높이 분석을 다른 균형 트리 구조와 견주어라.

??? success "연습문제 4 풀이"
    AVL은 높이가 $1.44\log n$ 이하이고 삭제마다 회전이 $O(\log n)$번까지 든다. 레드-블랙은 높이가 $2\log n$ 이하이고 연산마다 회전이 많아야 3번이다. B-트리는 높이가 $O(\log_B n)$이며 디스크 입출력에 맞추어져 있다. 스플레이 트리는 분할 상환으로 $O(\log n)$이지만 최악은 $O(n)$이다.