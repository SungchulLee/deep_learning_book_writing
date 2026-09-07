# 마구잡이 건너뛰기 목록

정렬한 이음 목록은 찾기를 $O(n)$ 시간에 한다. 고른 이진 찾기 나무는 $O(\log n)$을 이루지만 복잡한 다시 고르기가 필요하다. **건너뛰기 목록**은 정해진 다시 고르기 대신 마구잡이를 써서 찾기, 넣기, 지우기에 같은 기댓값 $O(\log n)$ 시간을 준다. 낱개마다 마구잡이로 위 층으로 올려 "빠른 길"을 만들어 찾기가 목록의 큰 조각을 건너뛰게 한다.

## 구조

건너뛰기 목록은 이음 목록 여러 층으로 이루어진다. 층 0(맨 아래)은 $n$개 낱개를 모두 정렬한 차례로 담는다. 위 층은 저마다 아래 층 낱개의 아무 부분 모임을 담는다.

낱개마다 아무 **높이**를 매긴다. 곧 공정한 동전을 거듭 던져 처음 뒷면이 나오기 전까지 앞면의 수를 센다. 높이가 $h$인 낱개는 층 $0, 1, \ldots, h$에 나타난다.

!!! note "기댓값 높이"
    공정한 동전이면(올릴 확률 $p = 1/2$):

    - 낱개마다 기댓값 높이: $1/(1-p) = 2$
    - 층 $i$의 기댓값 낱개 수: $n/2^i$
    - 기댓값 최대 높이: $O(\log n)$

## 찾기

열쇠 $x$을 찾으려면:

1. 가장 높은 층의 머리 마디에서 시작한다.
2. 다음 마디의 열쇠가 $> x$이 될 때까지(또는 끝에 이를 때까지) 지금 층을 따라 오른쪽으로 간다.
3. 한 층 아래로 내려가 되풀이한다.
4. 층 0에서 지금 마디의 열쇠가 $= x$인지 살핀다.

이는 이분 찾기와 비슷하다. 곧 층마다 남은 찾기 공간이 기댓값으로 반이 된다.

## 기대 탐색 시간

**정리.** 낱개 $n$개인 건너뛰기 목록에서 기댓값 찾기 시간은 $O(\log n)$이다.

**밝힘 밑그림.** 목표에서 머리로 찾기 길을 *거꾸로* 살핀다. 거슬러 가는 걸음마다 한 층 위로 가거나($1/2$의 확률로. 지금 마디가 올려졌기 때문이다) 같은 층에서 왼쪽으로 간다($1/2$의 확률로). 층 $i$에서 왼쪽으로 가는 기댓값 횟수는 많아야 $1/p = 2$이다. 층이 $O(\log n)$개이므로 온 기댓값 길이는 $O(\log n)$이다.

더 자세히는 기댓값 견줌 횟수가 다음과 같다.

$$
E[\text{comparisons}] = \frac{\log_2 n}{p} + \frac{1}{1-p} = O(\log n)
$$

## 삽입

1. 자리를 찾는다(층마다 고칠 가리개를 좇으면서).
2. 새 낱개의 아무 높이 $h$을 만든다.
3. 낱개를 층 $0, 1, \ldots, h$에 넣으며 층마다 이음 목록에 끼워 넣는다.

$h$이 지금 최대 층을 넘으면 건너뛰기 목록에 새 층을 더한다.

## 삭제

1. 낱개를 찾는다(층마다 앞 마디를 좇으면서).
2. 낱개가 나타나는 층마다 그것을 뺀다.
3. 가장 높은 층이 비면 최대 층을 줄인다.

## 구현

```python
"""
마구잡이 건너뛰기 목록: 고른 이진 찾기 나무의 확률 대안.

찾기, 넣기, 지우기를 기대 시간 O(log n)에 받쳐 준다.
"""

import random


# === 건너뛰기 목록 마디 ===

class SkipNode:
    """건너뛰기 목록의 마디."""

    def __init__(self, key, level):
        self.key = key
        self.forward = [None] * (level + 1)


# === 건너뛰기 리스트 ===

class SkipList:
    """마구잡이 건너뛰기 목록 자료 짜임.

    찾기, 넣기, 지우기가 모두 기대 시간 O(log n)에 돈다.
    """

    def __init__(self, max_level=16, p=0.5):
        self.max_level = max_level
        self.p = p
        self.level = 0
        self.header = SkipNode(-float("inf"), max_level)
        self.size = 0

    def random_level(self):
        """동전 던지기로 아무 층을 만든다."""
        lvl = 0
        while random.random() < self.p and lvl < self.max_level:
            lvl += 1
        return lvl

    def search(self, key):
        """건너뛰기 목록에서 열쇠를 찾는다.

        찾으면 True, 아니면 False을 돌려준다.
        """
        current = self.header
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
        current = current.forward[0]
        return current is not None and current.key == key

    def insert(self, key):
        """건너뛰기 리스트에 키를 넣는다."""
        update = [None] * (self.max_level + 1)
        current = self.header

        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current

        current = current.forward[0]

        if current is None or current.key != key:
            new_level = self.random_level()

            if new_level > self.level:
                for i in range(self.level + 1, new_level + 1):
                    update[i] = self.header
                self.level = new_level

            new_node = SkipNode(key, new_level)
            for i in range(new_level + 1):
                new_node.forward[i] = update[i].forward[i]
                update[i].forward[i] = new_node

            self.size += 1

    def delete(self, key):
        """건너뛰기 목록에서 열쇠를 지운다.

        열쇠를 찾아 지웠으면 True를 돌려준다.
        """
        update = [None] * (self.max_level + 1)
        current = self.header

        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current

        current = current.forward[0]

        if current and current.key == key:
            for i in range(self.level + 1):
                if update[i].forward[i] != current:
                    break
                update[i].forward[i] = current.forward[i]

            while self.level > 0 and self.header.forward[self.level] is None:
                self.level -= 1

            self.size -= 1
            return True
        return False

    def display(self):
        """건너뛰기 목록의 모든 층을 찍는다."""
        for i in range(self.level, -1, -1):
            nodes = []
            node = self.header.forward[i]
            while node:
                nodes.append(str(node.key))
                node = node.forward[i]
            print(f"  Level {i}: {' -> '.join(nodes)}")


# === 메인 ===

if __name__ == "__main__":
    random.seed(42)
    sl = SkipList()

    # 원소 삽입
    for val in [3, 6, 7, 9, 12, 19, 17, 26, 21, 25]:
        sl.insert(val)

    print(f"Skip list ({sl.size} elements):")
    sl.display()

    # 탐색
    for key in [7, 10, 21]:
        print(f"Search {key}: {sl.search(key)}")

    # 지우기
    sl.delete(19)
    print(f"\nAfter deleting 19 ({sl.size} elements):")
    sl.display()
```

**출력:**
```
Skip list (10 elements):
  Level 2: 6 -> 17
  Level 1: 6 -> 9 -> 17 -> 21 -> 25
  Level 0: 3 -> 6 -> 7 -> 9 -> 12 -> 17 -> 19 -> 21 -> 25 -> 26
7 찾기: True
10 찾기: False
21 찾기: True

After deleting 19 (9 elements):
  Level 2: 6 -> 17
  Level 1: 6 -> 9 -> 17 -> 21 -> 25
  Level 0: 3 -> 6 -> 7 -> 9 -> 12 -> 17 -> 21 -> 25 -> 26
```

## 복잡도 요약

| 셈 | 기댓값 시간 | 가장 나쁜 경우 |
|---|---|---|
| 찾기 | $O(\log n)$ | $O(n)$ |
| 삽입 | $O(\log n)$ | $O(n)$ |
| 삭제 | $O(\log n)$ | $O(n)$ |
| 자리 | 기댓값 $O(n)$ | $O(n \log n)$ |

## 건너뛰기 목록과 고른 이진 찾기 나무

| 특징 | 건너뛰기 목록 | 고른 이진 찾기 나무 |
|---|---|---|
| 짜기 | 단순하다 | 돌리기가 복잡하다 |
| 보장 | 기댓값 $O(\log n)$ | 가장 나쁜 경우 $O(\log n)$ |
| 함께 돌기 | 자물쇠 없는 변형이 쉽다 | 나란히 하기 어렵다 |
| 저장턱 움직임 | 나쁘다(가리개 좇기) | 배열이면 더 낫다 |

!!! tip "언제 건너뛰기 목록을 고를까"
    건너뛰기 목록은 함께 도는 짜기에서 빛난다. 자물쇠 없는 건너뛰기 목록은 자물쇠 없는 고른 나무보다 훨씬 단순하다. Redis, LevelDB, 자바의 ConcurrentSkipListMap이 모두 건너뛰기 목록을 쓴다.

## 참고 문헌

- Pugh, W. "Skip Lists: A Probabilistic Alternative to Balanced Trees." *CACM*, 1990.
- Motwani, R. & Raghavan, P. *Randomized Algorithms*. Cambridge University Press.

## 연습문제

**연습문제 1.**
마구잡이 건너뛰기 목록의 핵심 마구잡이 재주와 그것이 정해진 방식보다 나은 까닭을 설명하라.

??? success "연습문제 1 풀이"
    마구잡이 건너뛰기 목록은 마구잡이를 써서 정해진 알고리즘이 마주칠 수 있는 가장 나쁜 들임을 피한다. 아무렇게나 고르므로 알고리즘의 솜씨가 들임의 짜임이 아니라 제 동전 던지기에 달린다. 그래서 모든 들임에 대해 참인 센 기댓값 시간이나 높은 확률의 보장을 흔히 얻으며, 짓궂거나 병리적인 경우를 걱정할 까닭이 없어진다. $\square$

---

**연습문제 2.**
마구잡이 건너뛰기 목록의 기댓값 시간 복잡도는 얼마인가? 가장 나쁜 경우의 복잡도는 얼마인가?

??? success "연습문제 2 풀이"
    기댓값 시간 복잡도는 흔히 $O(n)$이나 $O(n \log n)$이며 높은 확률로 이룬다. 가장 나쁜 경우는 다항식만큼 더 나쁠 수 있지만(예컨대 $O(n^2)$) 그럴 확률은 무시할 만큼 작다. 기댓값과 가장 나쁜 경우의 틈이 마구잡이의 값이며, 가장 나쁜 움직임이 일어날 확률은 들임 크기에 따라 지수로 줄어든다. $\square$

---

**연습문제 3.**
마구잡이 건너뛰기 목록은 라스베이거스 알고리즘인가 몬테카를로 알고리즘인가? 그 차이를 설명하라.

??? success "연습문제 3 풀이"
    **라스베이거스**: 늘 옳은 결과를 내며 도는 시간이 아무 변수이다(기댓값이 다항식). **몬테카를로**: 늘 다항식 시간에 돌지만 결과가 어떤 가둔 확률로 틀릴 수 있다. 마구잡이 건너뛰기 목록은 옳음을 보장하느냐 도는 시간을 보장하느냐에 따라 이 가운데 하나에 든다. 이 가름이 어긋날 확률을 어떻게 다룰지 정한다. $\square$

---

**연습문제 4.**
마구잡이 건너뛰기 목록에서 마구잡이를 없애거나 솜씨가 나쁠 확률을 줄이는 법을 설명하라.

??? success "연습문제 4 풀이"
    방책은 다음과 같다. (1) **거듭 해 보기**: 알고리즘을 여러 번 돌려 가장 좋거나 많은 쪽 결과를 택하면 어긋날 확률이 지수로 줄어든다. (2) **마구잡이 없애기**: 조건부 기댓값이나 흩는 함수 무리로 아무 고르기를 정해진 고르기로 바꾼다. (3) **키우기**: 몬테카를로 알고리즘에서는 $k$번 되풀이해 어긋남을 $2^{-k}$으로 줄인다. (4) **비슷 마구잡이 만들개**: 알고리즘이 보기에 "마구잡이처럼 보이는" 정해진 차례를 쓴다. $\square$
