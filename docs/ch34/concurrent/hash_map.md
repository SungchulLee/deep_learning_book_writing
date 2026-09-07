# 한꺼번에 쓰는 해시 표

여느 해시 표는 어림잡아 $O(1)$의 찾기, 넣기, 지우기를 주지만 이 연산들은 실에 안전하지 않다. 실 여럿이 같은 해시 표에 한꺼번에 닿으면 자료 겨룸이 망가짐을 낳는다. **한꺼번에 쓰는 해시 표**는 같은 $O(1)$ 겉면을 주면서도 한꺼번에 닿을 때의 옳은 거동을 보장한다. 설계에서 종요로운 어려움은 발맞추기의 너비와 길이를 가장 작게 하여 높은 처리량을 이루는 것이다.

## 한꺼번에 다루는 꾀

### 온통 하나뿐인 잠금

가장 쉬운 길은 모든 연산을 뮤텍스 하나로 감싸는 것이다. 옳지만 모든 닿기를 한 줄로 늘어세워 나란함의 이로움을 모두 없앤다.

### 줄무늬 잠금

해시 표를 저마다 잠금을 지닌 토막(줄무늬) $k$개로 가른다. 열쇠 $x$에 대한 연산은 오직 토막 $h(x) \bmod k$의 잠금만 얻는다. 서로 다른 토막에 대한 연산은 나란히 나아간다.

**처리량**: 줄무늬가 $k$개이고 실이 $p$개일 때, 열쇠가 고르게 흩어지면 어림 처리량이 $\min(p, k)$에 따라 늘어난다.

### 잠금 없는 길

넣기와 지우기에 원자적인 견주어 바꾸기(CAS)를 쓴다. 잠금을 쥐지 않으므로 실이 서로를 막지 않는다. 처리량이 가장 높지만 옳게 만들기가 훨씬 어렵다.

## 줄무늬 해시 표 구현

```python
"""
줄무늬 잠금을 쓰는 한꺼번에 쓰는 해시 표.

잠금 여럿(줄무늬)을 써서 해시 표의 서로 다른 토막에 나란히
닿게 한다. 서로 다른 줄무늬에 대한 연산은 한꺼번에 나아간다.
"""

import threading
from collections import defaultdict

# ===================================================================
# 한꺼번에 쓰는 해시 표
# ===================================================================

class ConcurrentHashMap:
    """실에 안전하도록 줄무늬 잠금을 쓰는 해시 표.

    인수:
        num_stripes: 잠금 줄무늬의 개수
        initial_capacity: 처음 두레박 개수
    """

    def __init__(self, num_stripes=16, initial_capacity=64):
        self.num_stripes = num_stripes
        self.capacity = initial_capacity
        self.buckets = [[] for _ in range(self.capacity)]
        self.locks = [threading.Lock() for _ in range(num_stripes)]
        self.size = 0

    def _stripe(self, key):
        """열쇠의 줄무늬 번호를 돌려준다."""
        return hash(key) % self.num_stripes

    def _bucket_index(self, key):
        """열쇠의 두레박 번호를 돌려준다."""
        return hash(key) % self.capacity

    def get(self, key, default=None):
        """실에 안전한 get.

        인수:
            key: 찾을 열쇠
            default: 열쇠가 없을 때 돌려줄 값

        돌려주는 값:
            갈무리된 값 또는 default
        """
        stripe = self._stripe(key)
        with self.locks[stripe]:
            idx = self._bucket_index(key)
            for k, v in self.buckets[idx]:
                if k == key:
                    return v
            return default

    def put(self, key, value):
        """실에 안전한 put.

        인수:
            key: 넣거나 고칠 열쇠
            value: 열쇠에 매어 둘 값
        """
        stripe = self._stripe(key)
        with self.locks[stripe]:
            idx = self._bucket_index(key)
            for i, (k, v) in enumerate(self.buckets[idx]):
                if k == key:
                    self.buckets[idx][i] = (key, value)
                    return
            self.buckets[idx].append((key, value))
            self.size += 1

    def delete(self, key):
        """실에 안전한 delete.

        인수:
            key: 없앨 열쇠

        돌려주는 값:
            열쇠를 찾아 없앴으면 True, 아니면 False
        """
        stripe = self._stripe(key)
        with self.locks[stripe]:
            idx = self._bucket_index(key)
            for i, (k, v) in enumerate(self.buckets[idx]):
                if k == key:
                    self.buckets[idx].pop(i)
                    self.size -= 1
                    return True
            return False

# ===================================================================
# 메인
# ===================================================================

if __name__ == "__main__":
    hmap = ConcurrentHashMap(num_stripes=4)

    # 실 하나로 옳음 살피기
    hmap.put("name", "Alice")
    hmap.put("age", 30)
    hmap.put("city", "NYC")
    print(f"get('name') = {hmap.get('name')}")
    print(f"get('age')  = {hmap.get('age')}")
    print(f"get('city') = {hmap.get('city')}")
    print(f"get('zip')  = {hmap.get('zip', 'N/A')}")

    # 실 여럿으로 넣기
    results = {}
    barrier = threading.Barrier(4)

    def worker(thread_id, count):
        barrier.wait()
        for i in range(count):
            key = f"t{thread_id}_k{i}"
            hmap.put(key, thread_id * 100 + i)
        results[thread_id] = count

    threads = [threading.Thread(target=worker, args=(t, 100))
               for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"\nMulti-threaded insertion:")
    print(f"  Threads: 4, items per thread: 100")
    print(f"  Total items: {hmap.size}")

    # 모든 항목이 있는지 살핀다
    all_found = all(
        hmap.get(f"t{t}_k{i}") is not None
        for t in range(4) for i in range(100)
    )
    print(f"  All items found: {all_found}")
```

**출력:**
```
get('name') = Alice
get('age')  = 30
get('city') = NYC
get('zip')  = N/A

Multi-threaded insertion:
  Threads: 4, items per thread: 100
  Total items: 403
  All items found: True
```

## 복잡도

| 연산 | 어림 때 | 나눠 갚음 |
|---|---|---|
| `get` | $O(1 + n/m)$ | 채움률이 좋으면 $O(1)$ |
| `put` | $O(1 + n/m)$ | $O(1)$ 나눠 갚음 |
| `delete` | $O(1 + n/m)$ | $O(1)$ 나눠 갚음 |

여기서 $n$은 항목 개수이고 $m$은 두레박 개수다. 채움률 $\alpha = n/m < 1$이면 모든 연산이 $O(1)$이다.

## 크기 바꾸기

채움률이 문턱(흔히 0.75)을 넘으면 표의 크기를 바꾸어야 한다. 한꺼번에 쓰는 자리에서 크기 바꾸기는 만만치 않다.

- **온 세상 멈추고 바꾸기**: 모든 줄무늬 잠금을 얻고 표를 곱절로 늘린 뒤 모든 항목을 다시 해시하고 잠금을 놓는다. 쉽지만 멈춤이 생긴다.
- **조금씩 바꾸기**: 옛 표와 새 표를 함께 지닌다. 여느 연산 동안 항목을 게으르게 옮긴다. 더 얽히지만 멈춤이 없다.

## 꾀 견주기

| 꾀 | 처리량 | 품 | 막힘 |
|---|---|---|---|
| 온통 하나뿐인 잠금 | 낮음 | 쉬움 | 있음 |
| 줄무늬 잠금 | 어중간~높음 | 어중간 | 줄무늬마다 |
| 잠금 없음(CAS) | 가장 높음 | 높음 | 없음 |
| 읽기-적기 잠금 | 어중간 | 어중간 | 적는 이가 막음 |

!!! tip "알맞은 꾀 고르기"
    읽기가 많은 일감에는 읽기-적기 잠금이나 잠금 없는 설계가 가장 높은 처리량을 준다. 읽기와 적기가 고른 일감에서는 줄무늬 잠금이 성능과 만들기 품 사이에서 좋은 타협을 준다.

## 참고 문헌

- Herlihy, M. and Shavit, N. *The Art of Multiprocessor Programming*, 13장(한꺼번에 하는 해싱).
- Lea, D. (2003). "Overview of package java.util.concurrent." (자바 ConcurrentHashMap 설계).

## 연습문제

**연습문제 1.**
자바 `ConcurrentHashMap`이 쓰는 줄무늬 잠금 설계를 풀어라. 그것은 어떻게 온통 하나뿐인 잠금보다 높은 처리량을 이루는가?

??? success "연습문제 1 풀이"
    줄무늬 잠금은 해시 표를 저마다 잠금으로 지키는 토막(줄무늬) $S$개로 가른다. 열쇠의 줄무늬는 열쇠 해시를 $S$으로 나눈 나머지로 정해진다. 서로 다른 줄무늬에 있는 열쇠에 대한 연산은 다툼 없이 나란히 나아간다. 줄무늬가 $S$개이고 실이 $T$개일 때 두 실이 같은 잠금에서 다툴 낌새는 연산 짝마다 대략 $1/S$으로, 온통 하나뿐인 잠금의 1.0에 견주어 작다. (열쇠가 고르게 흩어진다고 여기면) 처리량이 실 개수가 $S$에 이를 때까지 거의 곧게 늘어난다. 자바의 처음 `ConcurrentHashMap`은 기본으로 토막 16개를 썼으며, 자바 8에서는 붙박인 토막 개수를 아예 없앤 더 잘게 나눈 두레박별 CAS 길로 갈음되었다. $\square$

---

**연습문제 2.**
두레박 손가락질에 CAS를 쓰는 잠금 없는 해시 표에서 ABA 문제를 밝혀라. 푸는 길을 내놓아라.

??? success "연습문제 2 풀이"
    ABA 문제는 CAS가 헛되이 이루어질 때 생긴다. 실 1이 손가락질 값 A을 읽고 멈춘다. 실 2가 손가락질을 A에서 B으로, 다시 A으로 바꾼다(보기로 마디를 지우고 같은 주소에 새 마디를 넣는다). 실 1이 다시 돌 때 손가락질이 다시 A이므로 CAS가 이루어지지만 그 밑의 자료는 바뀌었다. 해시 표에서는 이 때문에 실이 새 마디를 낡거나 놓아준 마디에 이을 수 있다. 푸는 길은 이렇다. (1) **표를 단 손가락질**: 한결같이 늘어나는 판 세개를 손가락질의 안 쓰는 비트에 채워 넣는다(또는 곱절 너비 CAS를 쓴다). CAS가 손가락질과 판을 함께 견주므로 ABA가 드러난다. (2) **위험 손가락질**: 어떤 실도 가리키지 않을 때까지 기억 거둬들이기를 미루어, 놓아준 주소가 다시 쓰이지 못하게 한다. (3) **시대에 바탕을 둔 거둬들이기**: 미룬 놓아주기를 시대별로 묶어, 한꺼번에 읽는 어떤 이도 되쓴 주소를 보지 않게 한다. $\square$

---

**연습문제 3.**
어떤 한꺼번에 쓰는 해시 표가 곧게 더듬는 열린 주소 매기기를 쓴다. 지우기가 왜 골칫거리인지, 무덤 표시가 이를 어떻게 푸는지 풀어라.

??? success "연습문제 3 풀이"
    곧게 더듬기에서는 $h(k)$에서 시작해 $k$을 찾거나 빈 자리를 만날 때까지 잇단 자리를 훑어 열쇠 $k$을 찾는다. 열쇠를 지우면서 그 자리를 비었다고 표시하면, 지운 자리를 지나 더듬던 다른 열쇠를 뒤에 찾을 때 새로 빈 자리에서 멈추어 그 열쇠가 없다고 그르게 매듭짓는다. 무덤이 이를 푼다. 자리를 비우는 대신 "지워짐"이라고 표시한다. 찾기는 무덤을 차 있는 것으로 다루어(지나쳐 계속 훑는다) 나아가고, 넣기는 무덤을 쓸 수 있는 것으로 다루어(그 자리를 되쓸 수 있다) 넣는다. 한꺼번에 쓰는 자리에서는 무덤을 원자적으로 놓아야 하고, 찾는 실은 훑는 동안 어떤 자리가 차 있음에서 무덤으로 넘어가는 경우를 다루어야 한다. 나쁜 점은 때가 흐르며 무덤이 더듬는 사슬을 길게 만든다는 것이다. 이를 거둬들이려면 (적기 잠금 아래) 이따금 다시 해시해야 한다. $\square$

---

**연습문제 4.**
두레박이 $n$개이고 채움률이 $\alpha = n_{\text{항목}}/n$이며 해시 함수가 $k$개인(뻐꾸기 해싱) 한꺼번에 쓰는 해시 표가, 한꺼번에 읽는 이의 수와 상관없이 어림 $O(1)$ 찾기 때를 지님을 증명하여라.

??? success "연습문제 4 풀이"
    뻐꾸기 해싱에서 열쇠는 있을 수 있는 자리 $k$개 $h_1(\text{열쇠}), h_2(\text{열쇠}), \ldots, h_k(\text{열쇠})$ 가운데 하나에 갈무리된다. 찾기는 이 $k$개 자리를 살펴 어느 하나에서 열쇠를 찾으면 돌려준다. $k$이 상수이므로(흔히 2나 3이다) 찾기는 $\alpha$이나 표 크기와 상관없이 $k = O(1)$번 기억에 닿는다. 읽기는 곁수가 없으므로 한꺼번에 읽는 이들은 서로를 방해하지 않는다. 잠금이 없어도 저마다 홀로 $k$개 자리에 닿는다. 걱정거리는 오직 하나, 읽는 동안 한꺼번에 적는 이가 열쇠를 제 자리 둘 사이에서 옮겨 그릇된 아님을 낳는 것이다. 이는 읽는 이가 다시 꾀하거나($k$개 자리를 다시 모두 살핀다) 판 세개를 써서 푼다. 살피는 자리의 수가 늘 꼭 $k$이고 한꺼번에 읽는 이의 수와 매이지 않으므로 어림 $O(1)$ 매임이 성립한다. $\square$

---

**연습문제 5.**
한꺼번에 쓰는 해시 표 설계 셋 -- 온통 하나뿐인 잠금, 줄무늬 잠금, 잠금 없는 CAS -- 의 처리량 결을 읽기가 많은 일감(읽기 95%)과 적기가 많은 일감(적기 50%)에서 견주어라. 각 자리에 어느 설계가 가장 좋은가?

??? success "연습문제 5 풀이"
    읽기가 많을 때(읽기 95%): 온통 하나뿐인 잠금은 모든 연산을 한 줄로 늘어세우므로 다툼 때문에 처리량이 실 개수에 거꾸로 견준다. 줄무늬 잠금은 서로 다른 줄무늬에서 나란히 읽게 하지만 한 줄무늬 안에서는 읽는 이들을 여전히 한 줄로 늘어세운다. 잠금 없는 CAS(또는 RCU를 더한 것)는 발맞추기 덧듦 없이 온전히 나란히 읽게 하여 거의 곧은 늘어남을 이룬다. 가장 좋은 것은 잠금 없음이나 RCU다. 적기가 많을 때(적기 50%): 온통 하나뿐인 잠금의 처리량은 무너진다. 줄무늬 잠금은 어중간한 나란함을 준다. 서로 다른 줄무늬에 적는 일은 나란히 나아가고, 줄무늬가 $S = 64$개이면 연산 짝마다 다툴 낌새가 약 $1.5\%$이다. 잠금 없는 CAS는 다툼 아래 다시 꾀하는 CAS가 잦아 앓지만(어그러진 CAS마다 다시 꾀하는 한 판을 버린다) 맞물려 멈춤과 우선순위 뒤집힘을 비껴간다. 가장 좋은 것은 쉬움과 어림할 수 있는 성능을 바라면 줄무늬 잠금이고, 다시 꾀하는 비율을 다스릴 수 있다면 가장 큰 처리량을 바라 잠금 없음이다. $\square$
