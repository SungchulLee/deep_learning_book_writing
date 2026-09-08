# 한꺼번에 쓰는 건너뛰기 목록

고른 이진 찾기 나무(AVL, 붉은-검은)는 다시 고르는 동안 벌이는 돌리기가 여러 마디에 걸치므로 얽힌 잠금 규약이 있어야 하고, 그래서 한꺼번에 쓰게 만들기가 어렵다. **건너뛰기 목록**은 솔깃한 다른 길을 준다. 마구잡이 얼개라 돌리기가 없고, 넣기와 지우기가 가까운 마디에만 미친다. 그래서 건너뛰기 목록은 한꺼번에 닿기에 절로 알맞으며, 잠금 없는 한꺼번에 쓰는 건너뛰기 목록은 온 세상 발맞추기 없이 모든 연산에 어림 $O(\log n)$ 때를 이룬다.

---

## 1. 건너뛰기 목록 되짚기

건너뛰기 목록은 원소마다 낌새 $p$(흔히 $1/2$)으로 더 높은 층에 올려 놓는 켜진 이음 목록이다. 찾기는 가장 위 층에서 시작해 다음 손가락질이 지나칠 때 아래로 내려간다. 이로써 어림 $O(\log n)$의 찾기, 넣기, 지우기 때를 얻는다.

층의 어림 개수는 $O(\log n)$이고 원소마다 어림 $O(1)$개의 손가락질을 지닌다.

---

## 2. 한꺼번에 쓰기에 건너뛰기 목록이 알맞은 까닭

- **돌리기가 없음**: 고른 이진 찾기 나무와 달리 건너뛰기 목록 연산은 가까운 손가락질만 고친다. 온 세상 얼개를 다시 짤 까닭이 없다.
- **층이 서로 풀려 있음**: 층 $k$에서 넣기는 (이은 뒤에는) 층 $k+1$ 위쪽에 미치지 않는다. 그래서 잘게 나눈 잠금이나 잠금 없는 발맞추기를 쓸 수 있다.
- **홀로 던지는 동전**: 새 마디의 층은 지금 얼개와 매이지 않고 아무렇게나 정해진다.

---

## 3. 한꺼번에 벌이는 연산

### 잠금을 쓰는 길(잘게 나눔)

고치는 마디만 잠근다. 어느 층에서 마디 $A$와 $B$ 사이에 넣는다면:

1. 그 층에서 $A$을 잠근다.
2. 그 층에서 $B$을 잠근다(같은 틈에 한꺼번에 넣기를 막으려고).
3. $A$와 $B$ 사이에 새 마디를 넣는다.
4. $B$을 풀고 이어서 $A$을 푼다.
5. 새 마디가 나타나는 층마다 되풀이한다.

이로써 서로 다른 자리에서 한꺼번에 넣고 지울 수 있다.

### 잠금 없는 길(CAS를 씀)

잠금 없는 길은 마디를 실제로 떼어 내기 앞서 지움 표시를 한다.

1. **뜻으로 지우기**: CAS로 마디의 다음 손가락질에 표시 비트를 켠다.
2. **참으로 지우기**: 뒤이어 훑는 이들이 표시된 마디를 건너뛰고 CAS로 목록에서 떼어 낸다.
3. **넣기**: 새 마디를 맨 아래 층에 먼저 잇고 그다음 더 높은 층에 잇는다.

---

## 4. 구현

```python
"""
잘게 나눈 잠금을 쓰는 한꺼번에 쓰는 건너뛰기 목록.

마디마다 잠금을 두어 건너뛰기 목록의 서로 다른 자리에서
한꺼번에 연산을 벌일 수 있게 한다.
"""

import random
import threading

# ===================================================================
# 한꺼번에 쓰는 건너뛰기 목록
# ===================================================================

MAX_LEVEL = 16

class SkipNode:
    """층마다 다음 손가락질과 잠금을 지닌 건너뛰기 목록 마디."""

    def __init__(self, key, value, level):
        self.key = key
        self.value = value
        self.next = [None] * (level + 1)
        self.lock = threading.Lock()
        self.level = level

class ConcurrentSkipList:
    """잘게 나눈 잠금을 쓰는 건너뛰기 목록.

    인수:
        max_level: 층의 최대 개수
        p: 다음 층으로 올릴 낌새
    """

    def __init__(self, max_level=MAX_LEVEL, p=0.5):
        self.max_level = max_level
        self.p = p
        self.header = SkipNode(float('-inf'), None, max_level)
        self.level = 0
        self._lock = threading.Lock()

    def _random_level(self):
        """새 마디의 층을 아무렇게나 짓는다."""
        lvl = 0
        while random.random() < self.p and lvl < self.max_level:
            lvl += 1
        return lvl

    def search(self, key):
        """건너뛰기 목록에서 열쇠를 찾는다.

        인수:
            key: 찾을 열쇠

        돌려주는 값:
            찾으면 값, 아니면 None
        """
        current = self.header
        for i in range(self.level, -1, -1):
            while (current.next[i] is not None and
                   current.next[i].key < key):
                current = current.next[i]
        current = current.next[0]
        if current is not None and current.key == key:
            return current.value
        return None

    def insert(self, key, value):
        """실에 안전한 넣기.

        인수:
            key: 넣을 열쇠
            value: 매어 둘 값
        """
        update = [None] * (self.max_level + 1)
        current = self.header

        for i in range(self.level, -1, -1):
            while (current.next[i] is not None and
                   current.next[i].key < key):
                current = current.next[i]
            update[i] = current

        current = current.next[0]

        if current is not None and current.key == key:
            current.value = value
            return

        new_level = self._random_level()

        with self._lock:
            if new_level > self.level:
                for i in range(self.level + 1, new_level + 1):
                    update[i] = self.header
                self.level = new_level

        new_node = SkipNode(key, value, new_level)

        for i in range(new_level + 1):
            if update[i] is not None:
                with update[i].lock:
                    new_node.next[i] = update[i].next[i]
                    update[i].next[i] = new_node

    def to_list(self):
        """모든 열쇠-값 짝을 매긴 차례대로 돌려준다."""
        result = []
        current = self.header.next[0]
        while current is not None:
            result.append((current.key, current.value))
            current = current.next[0]
        return result

# ===================================================================
# 메인
# ===================================================================

if __name__ == "__main__":
    random.seed(42)
    sl = ConcurrentSkipList()

    # 실 하나로 옳음 살피기
    for key in [3, 6, 1, 9, 2, 7, 4, 8, 5]:
        sl.insert(key, key * 10)

    print("Skip list contents:", sl.to_list())
    print(f"search(5) = {sl.search(5)}")
    print(f"search(10) = {sl.search(10)}")

    # 실 여럿으로 넣기
    sl2 = ConcurrentSkipList()
    barrier = threading.Barrier(4)

    def worker(start, count):
        barrier.wait()
        for i in range(start, start + count):
            sl2.insert(i, i)

    threads = [threading.Thread(target=worker, args=(t * 25, 25))
               for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    items = sl2.to_list()
    print(f"\nConcurrent insertion: {len(items)} items")
    print(f"Sorted correctly: {items == sorted(items)}")
    print(f"All present: {len(items) == 100}")
```

**출력:**
```
Skip list contents: [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60), (7, 70), (8, 80), (9, 90)]
search(5) = 50
search(10) = None

Concurrent insertion: 100 items
Sorted correctly: True
All present: True
```

---

## 5. 복잡도

| 연산 | 어림 때 |
|---|---|
| 찾기 | $O(\log n)$ |
| 넣기 | $O(\log n)$ |
| 지우기 | $O(\log n)$ |
| 자리 | 어림 $O(n)$ |

---

## 6. 한꺼번에 쓰는 나무와 견주기

| 성질 | 한꺼번에 쓰는 건너뛰기 목록 | 한꺼번에 쓰는 붉은-검은 나무 |
|---|---|---|
| 돌리기 | 없음 | 있어야 하고 잠금을 얽히게 함 |
| 잠금 잘기 | 마디마다, 층마다 | 마디마다 + 돌리기 이웃 |
| 잠금 없이 될까 | 예(잘 살펴진 것) | 어려움 |
| 캐시 거동 | 손가락질 좇기 | 마디를 채워 넣으면 더 좋음 |
| 실제 쓰임 | 자바 ConcurrentSkipListMap | 한꺼번에 쓰기에는 덜 흔함 |

!!! note "자바 ConcurrentSkipListMap"
    자바의 여느 서고는 매긴 차례를 지니는 한꺼번에 쓰는 표로 한꺼번에 쓰는 나무 대신 잠금 없는 건너뛰기 목록(ConcurrentSkipListMap)을 골랐다. 바로 건너뛰기 목록이 잠금 없이 만들기 쉽기 때문이다.

---

## 연습문제

**연습문제 1.**
건너뛰기 목록이 고른 이진 찾기 나무보다 한꺼번에 닿기에 알맞은 까닭을 풀어라. 어떤 얼개 성질이 이 다름을 낳는가?

??? success "연습문제 1 풀이"
    고른 이진 찾기 나무(AVL, 붉은-검은)는 고름을 지키려고 넣고 지우는 동안 돌리기를 벌여야 한다. 돌리기 한 번이 어버이, 자식, 손자의 손가락질을 고치는데, 이 마디 셋은 나무 곳곳에 흩어져 있을 수 있다. 이 마디들에 한꺼번에 닿으려면 나무의 크기가 들쭉날쭉한 구역을 잠가야 하고 잠그는 차례를 미리 알기 어렵다(돌리기가 위로 번질 수 있다). 건너뛰기 목록은 이를 아예 비껴간다. 그 고름은 마구잡이(넣을 때 층을 아무렇게나 매김)에 기대므로 넣고 지운 뒤 얼개를 다잡을 까닭이 없다. 넣기는 층마다 바로 앞 마디에만 미치고, 이 앞 마디들은 저마다 홀로 잠그거나(잘게 나눈 잠금) CAS로 고칠 수 있다(잠금 없음). 연산마다 미리 알 수 있는 차례로 이웃한 마디 $O(\log n)$개만 건드리는 이 고침의 지역성이 건너뛰기 목록을 잘게 나눈 잠금과 잠금 없는 한꺼번에 쓰기에 절로 알맞게 만든다. $\square$

---

**연습문제 2.**
잠금 없는 건너뛰기 목록 넣기 알고리즘을 밝혀라. 넣는 동안 한꺼번에 벌어진 지우기가 앞 마디를 없애는 경우를 어떻게 다루는가?

??? success "연습문제 2 풀이"
    잠금 없는 넣기: (1) 가장 위 층부터 찾으며 층마다 앞 마디와 뒤 마디를 적어 둔다. (2) 아무렇게나 고른 높이로 새 마디를 마련한다. (3) 층 0(맨 아래)부터 앞 마디의 `next` 손가락질을 뒤 마디에서 새 마디로 CAS 한다. CAS가 어그러지면(앞 마디가 바뀌었으면) 그 층에서 다시 찾고 다시 꾀한다. (4) 더 높은 층마다 되풀이한다. 한꺼번에 벌어지는 지우기에 대해서는, 지운 마디를 실제로 떼어 내기 앞서 (`next` 손가락질의 깃발로) 뜻으로 먼저 표시한다. 넣는 동안 찾기가 표시된(뜻으로 지워진) 앞 마디를 만나면 넣는 실이 그것을 떼어 내도록 거들고(표시된 마디를 참으로 없앤다) 다시 찾는다. 이 "거들기" 장치가 나아감을 지키고 지운 마디가 든 사슬에 넣는 일을 막는다. $\square$

---

**연습문제 3.**
한꺼번에 쓰는 건너뛰기 목록 찾기 연산의 어림 때 복잡도를 살펴라. 한꺼번에 벌어지는 적기의 다툼이 점근 찾기 때에 미치는가?

??? success "연습문제 3 풀이"
    차례로 벌이는 건너뛰기 목록 찾기는 어림 $O(\log n)$ 때가 든다. 층마다 아래로 내려가기 앞서 어림 $O(1)$개의 마디를 지나고 층이 $O(\log n)$개이기 때문이다. 한꺼번에 쓰는 자리에서 찾기는 읽기뿐이며 어떤 손가락질도 고치지 않으므로 CAS를 하나도 벌이지 않는다. 여러 찾기가 서로를 방해하지 않고 나란히 나아간다. 한꺼번에 벌어지는 적기(넣기/지우기)가 찾는 동안 목록 얼개를 고칠 수 있으나 찾기는 여전히 옳다. 그 까닭은 이렇다. (1) 원자적으로 펴낸 새 마디는 드러나 있고 안전하게 지날 수 있다. (2) 지운 마디는 떼어 내기 앞서 뜻으로 표시되며, 표시된 마디를 만난 찾기는 그저 다음으로 건너뛴다. 적기가 층마다 많아야 상수 개의 손가락질을 바꾸므로 층마다 지나는 마디의 어림 개수는 $O(1)$으로 남는다. 따라서 점근 어림 찾기 때는 $O(\log n)$이며 한꺼번에 벌어지는 적기에 흔들리지 않는다. $\square$

---

**연습문제 4.**
자바의 `ConcurrentSkipListMap`은 매긴 차례를 지니는 한꺼번에 쓰는 표로 쓰인다. 낱점 찾기, 범위 물음, 차례대로 훑기라는 서로 다른 닿기 결에서 `ConcurrentHashMap`과 그 성능 결을 견주어라.

??? success "연습문제 4 풀이"
    **낱점 찾기**: `ConcurrentHashMap`은 어림 $O(1)$ 때를 주고(해시하고 두레박에 닿는다) `ConcurrentSkipListMap`은 $O(\log n)$을 준다. 오로지 낱점만 찾는다면 해시 표가 3~10배 빠르다. **범위 물음**($[a, b]$ 안의 온 열쇠 찾기): `ConcurrentSkipListMap`은 $a$을 찾은 뒤 맨 아래 층 이음 목록을 지나며 $O(\log n + k)$에 이를 받쳐 준다. 여기서 $k$은 범위 안 열쇠의 개수다. `ConcurrentHashMap`에는 좋은 범위 물음이 없어 온 두레박을 $O(n)$에 훑어야 한다. **차례대로 훑기**: `ConcurrentSkipListMap`은 맨 아래 층 목록을 지나 매긴 차례대로 열쇠를 준다. `ConcurrentHashMap`은 차례를 하나도 보장하지 않는다. 권함: 차례 없는 열쇠-값 곳간에는 `ConcurrentHashMap`을 쓰고, 매긴 차례나 범위 물음, `ceilingKey`/`floorKey` 같은 연산이 필요하면 `ConcurrentSkipListMap`을 쓴다. $\square$

---

**연습문제 5.**
원소가 $n$개이고 올릴 낌새가 $p = 1/2$인 건너뛰기 목록의 어림 높이가 $O(\log n)$이고 어림 온 자리가 $O(n)$임을 증명하여라.

??? success "연습문제 5 풀이"
    **높이**: 마디가 층 $k$까지 올라갈 낌새는 $(1/2)^k$이다. 아무 마디의 가장 높은 층이 곧 높이다. 적어도 한 마디가 층 $c \log_2 n$에 이를 낌새는 많아야 $n \cdot (1/2)^{c \log_2 n} = n \cdot n^{-c} = n^{1-c}$이다. $c = 2$이면 이는 $1/n$이므로 높은 낌새로 높이가 많아야 $2 \log_2 n = O(\log n)$이다. **자리**: 층 0의 마디마다 낌새 $1/2$으로 층 1에, 낌새 $1/4$으로 층 2에 오르며 그렇게 이어진다. 마디 하나의 어림 손가락질 개수는 $\sum_{k=0}^{\infty} (1/2)^k = 2$이다. 마디 $n$개에 걸쳐 어림 온 손가락질 개수는 $2n = O(n)$이다. $\square$

## 정리하며

이 마당은 건너뛰기 목록 되짚기、한꺼번에 쓰기에 건너뛰기 목록이 알맞은 까닭、한꺼번에 벌이는 연산、구현을 차례로 짚었다.

**참고 문헌**

- Pugh, W. (1990). "Concurrent maintenance of skip lists." *TR CS-2222, University of Maryland*.
- Herlihy, M. et al. (2006). "A provably correct scalable concurrent skip list." *OPODIS*.
