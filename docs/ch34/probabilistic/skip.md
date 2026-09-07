# 건너뛰기 목록

고른 이진 찾기 나무(AVL, 붉은-검은)는 $O(\log n)$ 연산을 보장하지만 얽힌 다시 고르기 논리가 있어야 한다. **건너뛰기 목록**은 얼개 불변량 대신 마구잡이를 써서 같은 어림 매임을 이룬다. 원소마다 낌새 $p$(흔히 $1/2$)으로 더 높은 "빠른 길"에 올려, 이음 목록 위에서 두 갈래 찾기 같은 훑기를 이루는 이음 목록의 켜를 만든다.

## 얼개

건너뛰기 목록은 매긴 이음 목록 $L_0, L_1, \ldots, L_h$의 모임이다.

- $L_0$은 원소 $n$개를 모두 담는다.
- $L_i$의 원소마다 서로 매이지 않고 낌새 $p$으로 $L_{i+1}$에 올라간다.
- 파수 마디 $-\infty$과 $+\infty$이 층마다 나타난다.

층 $i$의 어림 원소 개수는 $n p^i$이고 어림 높이는 다음과 같다.

$$
E[h] = \log_{1/p} n = O(\log n)
$$

## 찾기

열쇠 $k$을 찾으려면 가장 위 층의 가장 왼쪽 파수에서 시작한다.

1. 다음 원소가 $k$보다 작으면 오른쪽으로 옮긴다.
2. 다음 원소가 $\ge k$이면 한 층 내려간다.
3. 층 0에 이를 때까지 되풀이한다.
4. 층 0의 원소가 $k$과 같으면 그것을 돌려주고, 아니면 $k$은 없다.

**어림 때**: 층마다 내려가기 앞서 어림잡아 많아야 $1/p$번 견준다. 층이 $O(\log n)$개이므로 다음과 같다.

$$
E[T_{\text{찾기}}] = O\!\left(\frac{\log n}{p}\right) = O(\log n) \quad (p \text{이 상수일 때})
$$

## 넣기

열쇠 $k$을 넣으려면:

1. $k$의 자리를 찾으며 층마다 앞 마디를 적어 둔다.
2. 동전을 던져 새 마디의 높이를 정한다. $\ell = 0$으로 두고, 동전이 앞면이면(낌새 $p$) $\ell$을 올린다.
3. 적어 둔 앞 마디 뒤에 끼워 넣어 마디를 층 $0$부터 $\ell$까지에 넣는다.

$$
E[T_{\text{넣기}}] = O(\log n), \quad E[S_{\text{넣기}}] = O\!\left(\frac{1}{1-p}\right) = O(1)
$$

## 지우기

열쇠 $k$을 지우려면:

1. $k$을 찾으며 층마다 앞 마디를 적어 둔다.
2. 손가락질을 다잡아 $k$이 나타나는 층마다 그것을 없앤다.
3. 비지 않은 가장 위 층이 낮아졌으면 높이를 줄인다.

$$
E[T_{\text{지우기}}] = O(\log n)
$$

## 어림 자리

원소마다 어림 높이가 $1/(1-p)$이므로 온 어림 자리는 다음과 같다.

$$
E[S] = \frac{n}{1-p} = O(n) \quad (p \text{이 상수일 때})
$$

$p = 1/2$이면 어림 자리가 손가락질 $2n$개다.

## 복잡도 간추림

| 연산 | 어림 때 | 가장 나쁠 때 |
|---|---|---|
| 찾기 | $O(\log n)$ | $O(n)$ |
| 넣기 | $O(\log n)$ | $O(n)$ |
| 지우기 | $O(\log n)$ | $O(n)$ |
| 자리 | $O(n)$ | $O(n \log n)$ |

가장 나쁜 경우(모든 원소가 가장 높은 층까지 올라감)는 셈에 넣지 않아도 될 만큼 작은 낌새로 생긴다.

## 구현

```python
"""
건너뛰기 목록 -- 마구잡이 매긴 자료 얼개.

다시 고르기 대신 확률로 올려, 찾기, 넣기, 지우기에 어림
O(log n) 때를 이룬다.
"""

from __future__ import annotations
import random
import math


# === 건너뛰기 목록 마디 =======================================================

class SkipNode:
    """여러 층에 앞 손가락질을 지닌 마디."""

    def __init__(self, key: float, level: int):
        self.key = key
        self.forward: list[SkipNode | None] = [None] * (level + 1)


# === 건너뛰기 목록 ============================================================

class SkipList:
    """어림 O(log n) 연산을 주는 매긴 확률 자료 얼개."""

    def __init__(self, max_level: int = 16, p: float = 0.5):
        self.max_level = max_level
        self.p = p
        self.level = 0
        self.header = SkipNode(float("-inf"), max_level)
        self.size = 0

    def _random_level(self) -> int:
        """동전을 던져 아무 층을 짓는다."""
        lvl = 0
        while random.random() < self.p and lvl < self.max_level:
            lvl += 1
        return lvl

    def search(self, key: float) -> bool:
        """*key*이 건너뛰기 목록에 있으면 True를 돌려준다."""
        current = self.header
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
        current = current.forward[0]
        return current is not None and current.key == key

    def insert(self, key: float) -> None:
        """*key*을 건너뛰기 목록에 넣는다."""
        update = [None] * (self.max_level + 1)
        current = self.header
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current

        current = current.forward[0]
        if current and current.key == key:
            return  # 겹침

        new_level = self._random_level()
        if new_level > self.level:
            for i in range(self.level + 1, new_level + 1):
                update[i] = self.header
            self.level = new_level

        new_node = SkipNode(key, new_level)
        for i in range(new_level + 1):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node
        self.size += 1

    def to_list(self) -> list[float]:
        """모든 원소를 매긴 차례대로 돌려준다."""
        result = []
        current = self.header.forward[0]
        while current:
            result.append(current.key)
            current = current.forward[0]
        return result


# === 메인 =====================================================================

if __name__ == "__main__":
    random.seed(42)
    sl = SkipList()
    for val in [3, 6, 7, 9, 12, 19, 17, 26, 21, 25]:
        sl.insert(val)

    print(f"Elements: {sl.to_list()}")
    print(f"Levels used: {sl.level}")
    print(f"Search 19: {sl.search(19)}")
    print(f"Search 15: {sl.search(15)}")
```

**출력:**

```
Elements: [3, 6, 7, 9, 12, 17, 19, 21, 25, 26]
Levels used: 3
Search 19: True
Search 15: False
```

여러 층 얼개에 걸쳐 원소가 매긴 차례대로 지켜지고, 찾기가 있는 열쇠와 없는 열쇠를 옳게 가른다.

## 참고 문헌

- Pugh, W. "Skip Lists: A Probabilistic Alternative to Balanced Trees." *CACM*, 1990
- [Advanced Data Structures (Brass)](https://www.cambridge.org/core/books/advanced-data-structures/D56E2269D7CEE969A3B8105D3541F601)

## 연습문제

**연습문제 1.**
건너뛰기 목록의 찾기 알고리즘을 밝혀라. 여러 층 얼개는 어떻게 어림 $O(\log n)$ 찾기 때를 주는가?

??? success "연습문제 1 풀이"
    열쇠 $k$ 찾기: 왼쪽 위 모서리(가장 높은 층, 머리 파수)에서 시작한다. 층마다 다음 마디의 열쇠가 $k$보다 작은 동안 이음 목록을 따라 오른쪽으로 옮긴다. 다음 열쇠가 $\ge k$이면(또는 목록이 끝나면) 한 층 내려간다. 층 0에 이를 때까지 되풀이한다. 층 0에서 오른쪽 마디의 열쇠가 $k$과 같으면 그것을 돌려주고, 아니면 $k$은 없다. 여러 층 얼개는 두 갈래 찾기처럼 움직인다. (원소마다 낌새 $1/2$으로 올라가므로) 층마다 대략 원소의 절반을 건너뛴다. 층마다 어림 견줌 횟수는 $O(1)$이다(어림잡아 내려가기 앞서 2번 견준다). 층이 $O(\log n)$개이므로 온 어림 견줌은 $O(\log n)$이다. $\square$

---

**연습문제 2.**
원소가 $n$개이고 올릴 낌새가 $p = 1/2$인 건너뛰기 목록의 어림 자리 쓰임을 살펴라. 온 층에 걸친 어림 손가락질 개수는 얼마인가?

??? success "연습문제 2 풀이"
    층 0의 원소마다 손가락질이 하나다. 층 $j$까지 올라간 원소는 손가락질이 $j + 1$개다(층 0부터 $j$까지 하나씩). 층 $j$ 이상에 있을 낌새는 $(1/2)^j$이다. 원소 하나의 어림 온 손가락질은 $\sum_{j=0}^{\infty} (1/2)^j = 2$이다. 원소 $n$개에 걸쳐 어림 온 손가락질은 $2n$이다. 이는 건너뛰기 목록이 그냥 이음 목록의 대략 두 배 자리를 쓰고, 층이 하나 늘 때마다 손가락질이 대략 $n/2^j$개 더 든다는 뜻이다. 자리는 어림잡아 작은 상수와 함께 $O(n)$이다. 가장 나쁠 때(모든 원소가 온 층에 올라감) 자리는 $O(n \log n)$이지만 이 일이 생길 낌새는 지수로 작다. $\square$

---

**연습문제 3.**
건너뛰기 목록을 AVL 나무, 붉은-검은 나무와 어림/가장 나쁠 때 때, 자리, 만들기 품, 캐시 성능에서 견주어라.

??? success "연습문제 3 풀이"
    | 성질 | 건너뛰기 목록 | AVL 나무 | 붉은-검은 나무 |
    |---|---|---|---|
    | 찾기(어림) | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ |
    | 찾기(가장 나쁠 때) | $O(n)$ (드묾) | $O(\log n)$ | $O(\log n)$ |
    | 넣기/지우기 | 어림 $O(\log n)$ | $O(\log n)$ | $O(\log n)$ |
    | 자리 | 어림 $O(n)$ | $O(n)$ | $O(n)$ |
    | 만들기 | 쉬움 | 어중간 | 얽힘 |
    | 캐시 성능 | 나쁨(손가락질) | 어중간 | 어중간 |

    건너뛰기 목록은 만들기가 쉬운 데서 이긴다(돌리기도, 빛깔/고름 지키기도 없다). AVL과 붉은-검은 나무는 가장 나쁠 때의 보장과 (마디를 함께 마련하면 이어져 있으므로) 캐시 성능에서 이긴다. 한꺼번에 쓰도록 만들 때에는 (온 세상을 다시 고르는 일이 없으므로) 건너뛰기 목록을 즐겨 쓴다. $\square$

---

**연습문제 4.**
원소가 $n$개이고 올릴 낌새가 $p$인 건너뛰기 목록의 어림 높이가 $O(\log_{1/p} n)$임을 증명하여라.

??? success "연습문제 4 풀이"
    높이는 아무 원소의 가장 높은 층이다. 원소 $i$이 층 $\ge j$에 이를 낌새는 $p^j$이다. 합집합 매임으로 $P(\text{높이} \ge j) \le n \cdot p^j$이다. $n \cdot p^j \le 1$으로 두면 $j \ge \log_{1/p} n$을 얻는다. 더 또렷이 $P(\text{높이} \ge c \log_{1/p} n) \le n \cdot p^{c \log_{1/p} n} = n \cdot n^{-c} = n^{1-c}$이다. $c = 2$이면 높이가 $2 \log_{1/p} n$을 넘을 낌새가 $\le 1/n$이다. 따라서 어림 높이는 $O(\log_{1/p} n)$이다. $p = 1/2$이면 $O(\log_2 n)$이다. $p = 1/4$이면 $O(\log_4 n) = O(\log_2 n / 2)$으로 층이 절반이지만 층마다 더 많이 훑는다. $\square$

---

**연습문제 5.**
차례 통계 연산, 곧 $k$번째로 작은 원소를 어림 $O(\log n)$ 때에 찾기를 받쳐 주는 건너뛰기 목록을 설계하여라. 어떤 덧붙임이 있어야 하는가?

??? success "연습문제 5 풀이"
    앞 손가락질마다 그것이 건너뛰는 원소의 개수(다다르는 마디를 넣는다)인 **너비**를 덧붙인다. 층 0에서는 너비가 모두 1이다. 더 높은 층에서 너비는 그것이 건너뛰는 아래 층 손가락질들의 너비 합이다. $k$번째 원소를 찾으려면 왼쪽 위에서 시작한다. 층마다 다음 손가락질의 너비가 $\le k$이면 $k$에서 그 너비를 빼고 오른쪽으로 옮긴다. 아니면 아래로 내려간다. $k = 0$이면(또는 겨눈 자리에 이르면) 지금 마디가 답이다. 이는 어림 $O(\log n)$ 때다(찾기와 같다). 넣기와 지우기는 넣거나 지우는 길을 따라 너비를 고쳐야 한다. 넣는 층 위에서 건너뛰는 손가락질의 너비에서 1을 빼고 새 마디의 손가락질에 1을 더한다. 이 덧붙임은 Redis의 매긴 집합(`ZRANGEBYSCORE`과 `ZRANK` 명령)에 쓰인다. $\square$
