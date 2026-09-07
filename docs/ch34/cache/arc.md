# 맞춰 가는 갈아 끼우기 캐시

LRU는 가장 오래 안 쓴 항목을 내보내므로 늦음이 이끄는 일감에서는 잘 돌지만, 자주 닿는 항목에 최근 닿지 않았을 때에는 어그러진다. LFU는 가장 드물게 쓴 항목을 내보내므로 잦기는 잘 다루지만 바뀌는 닿기 결에 더디게 움직인다. **맞춰 가는 갈아 끼우기 캐시**(ARC)는 늦음과 잦기 사이의 저울을 움직이며 맞추어 둘의 힘을 함께 얻고, 손으로 벼리지 않아도 일감에 절로 맞춰 간다.

## 설계 훑어보기

ARC는 최근 닿기 결과 잦은 닿기 결을 함께 좇는 줄 넷을 지닌다.

- **$T_1$**: 최근에 닿은 쪽(캐시에 들어온 뒤 한 번 닿음). 늦음 줄이다.
- **$T_2$**: 자주 닿은 쪽(캐시에 들어온 뒤 적어도 두 번 닿음). 잦기 줄이다.
- **$B_1$**: $T_1$에서 최근 내보내진 넋 항목. 이제 들어가지 못하는, 최근에 쓴 쪽을 좇는다.
- **$B_2$**: $T_2$에서 최근 내보내진 넋 항목. 이제 들어가지 못하는, 자주 쓴 쪽을 좇는다.

캐시는 $T_1 \cup T_2$에 항목을 지니며 $|T_1| + |T_2| \le c$이다. 여기서 $c$은 캐시 용량이다. 넋 줄 $B_1$과 $B_2$은 (값 없이) 열쇠만 갈무리하므로 덧드는 기억이 아주 적다.

## 맞춰 가는 매개변수

ARC는 $T_1$의 목표 크기를 정하는 매개변수 $p$ 하나(처음에 0)를 쓴다.

- **$B_1$에서 맞음**(최근 내보내진 늦음 항목을 다시 부름): $p$을 $\delta_1 = \max(1, |B_2| / |B_1|)$만큼 올린다. 늦음 몫이 넓어진다.
- **$B_2$에서 맞음**(최근 내보내진 잦기 항목을 다시 부름): $p$을 $\delta_2 = \max(1, |B_1| / |B_2|)$만큼 내린다. 잦기 몫이 넓어진다.

매개변수 $p$은 $0 \le p \le c$으로 매인다.

## 알고리즘

쪽 $x$을 캐시에 부를 때:

1. **$T_1$이나 $T_2$에서 맞음**: $x$을 $T_2$의 가장 늦게 쓴 자리로 옮긴다(이제 잦은 항목이다). 캐시 맞음.
2. **$B_1$에서 맞음**: 넋 맞음이다. 그 쪽은 늦음 줄에서 최근 내보내졌다. $p$을 올린다. $|T_1| + |T_2| = c$이면 $p$에 따라 $T_1$이나 $T_2$에서 내보낸다. $x$을 가져와 $T_2$에 넣는다.
3. **$B_2$에서 맞음**: 넋 맞음이다. 그 쪽은 잦기 줄에서 최근 내보내졌다. $p$을 내린다. 필요하면 내보낸다. $x$을 가져와 $T_2$에 넣는다.
4. **아주 빗나감**: $x$이 어느 줄에도 없다. 필요하면 내보낸다. $x$을 $T_1$에 넣는다.

### 내보내기 규칙(갈아 끼우기)

내보내야 할 때 $p$에 따라 제물을 고른다.

- $|T_1| > p$이면 $T_1$에서 가장 오래 안 쓴 항목을 내보낸다($B_1$으로 옮긴다).
- 아니면 $T_2$에서 가장 오래 안 쓴 항목을 내보낸다($B_2$으로 옮긴다).

## 구현

```python
"""
맞춰 가는 갈아 끼우기 캐시(ARC).

늦음 줄 T1, 잦기 줄 T2, 넋 줄 B1과 B2을 지닌다. 넋 맞음에
따라 늦음과 잦기 사이의 저울을 맞춰 간다.
"""

from collections import OrderedDict

# ===================================================================
# ARC 캐시
# ===================================================================

class ARCCache:
    """용량이 c인 맞춰 가는 갈아 끼우기 캐시.

    인수:
        capacity: 캐시에 담을 항목의 최대 개수(T1 + T2)
    """

    def __init__(self, capacity):
        self.c = capacity
        self.p = 0  # 맞춰 가는 매개변수
        self.t1 = OrderedDict()  # 늦음
        self.t2 = OrderedDict()  # 잦기
        self.b1 = OrderedDict()  # 넋 늦음
        self.b2 = OrderedDict()  # 넋 잦기
        self.hits = 0
        self.misses = 0

    def get(self, key):
        """캐시에서 열쇠를 찾는다. 빗나가면 None을 돌려준다."""
        if key in self.t1:
            val = self.t1.pop(key)
            self.t2[key] = val
            self.hits += 1
            return val
        if key in self.t2:
            self.t2.move_to_end(key)
            self.hits += 1
            return self.t2[key]
        self.misses += 1
        return None

    def put(self, key, value):
        """열쇠-값 짝을 넣거나 고친다."""
        if key in self.t1:
            self.t1.pop(key)
            self.t2[key] = value
            return
        if key in self.t2:
            self.t2[key] = value
            self.t2.move_to_end(key)
            return

        if key in self.b1:
            # B1에서 넋 맞음: 늦음을 아껴 준다
            delta = max(1, len(self.b2) // max(1, len(self.b1)))
            self.p = min(self.c, self.p + delta)
            self.b1.pop(key)
            self._replace(key)
            self.t2[key] = value
            return

        if key in self.b2:
            # B2에서 넋 맞음: 잦기를 아껴 준다
            delta = max(1, len(self.b1) // max(1, len(self.b2)))
            self.p = max(0, self.p - delta)
            self.b2.pop(key)
            self._replace(key)
            self.t2[key] = value
            return

        # 아주 빗나감
        total = len(self.t1) + len(self.b1)
        if total >= self.c:
            if len(self.t1) < self.c:
                self.b1.popitem(last=False)
            else:
                self.t1.popitem(last=False)
        self._replace(key)
        self.t1[key] = value

        # 넋 줄의 길이를 막는다
        while len(self.b1) > self.c:
            self.b1.popitem(last=False)
        while len(self.b2) > self.c:
            self.b2.popitem(last=False)

    def _replace(self, key):
        """캐시가 꽉 찼으면 항목 하나를 내보낸다."""
        if len(self.t1) + len(self.t2) < self.c:
            return
        if self.t1 and (len(self.t1) > self.p or
                        (key in self.b2 and len(self.t1) == self.p)):
            evicted_key, evicted_val = self.t1.popitem(last=False)
            self.b1[evicted_key] = None
        elif self.t2:
            evicted_key, evicted_val = self.t2.popitem(last=False)
            self.b2[evicted_key] = None

# ===================================================================
# 메인
# ===================================================================

if __name__ == "__main__":
    cache = ARCCache(capacity=3)

    requests = ["A", "B", "C", "A", "D", "B", "A", "E", "B", "A"]

    print("ARC Cache simulation (capacity=3):")
    for req in requests:
        result = cache.get(req)
        if result is None:
            cache.put(req, req)
            print(f"  {req}: MISS -> inserted")
        else:
            print(f"  {req}: HIT")

    total = cache.hits + cache.misses
    print(f"\nHits: {cache.hits}/{total} "
          f"({100*cache.hits/total:.0f}%)")
    print(f"p (adaptation): {cache.p}")
```

**출력:**
```
ARC Cache simulation (capacity=3):
  A: MISS -> inserted
  B: MISS -> inserted
  C: MISS -> inserted
  A: HIT
  D: MISS -> inserted
  B: HIT
  A: HIT
  E: MISS -> inserted
  B: HIT
  A: HIT

Hits: 4/10 (40%)
p (adaptation): 1
```

## 복잡도

ARC의 모든 연산은 해시 표와 두 겹 이음줄(`OrderedDict`으로)을 써서 $O(1)$ 나눠 갚는 때에 돈다.

| 연산 | 때 | 자리 |
|---|---|---|
| `get` | $O(1)$ | -- |
| `put` | $O(1)$ 나눠 갚음 | -- |
| 온 자리 | -- | 캐시에 $O(c)$ + 넋 줄에 $O(c)$ |

## LRU, LFU와 견주기

| 성질 | LRU | LFU | ARC |
|---|---|---|---|
| 일감에 맞춰 감 | 아니오 | 아니오 | 예 |
| 훑기를 버팀 | 아니오 | 예 | 예 |
| 늦음이 옮겨감을 다룸 | 예 | 아니오 | 예 |
| 자리 덧듦 | $O(c)$ | $O(c)$ | $O(2c)$ (넋 줄) |
| 만들기 품 | 쉬움 | 어중간 | 어중간 |

!!! note "훑기 버팀"
    서로 다른 항목을 줄줄이 훑으면 LRU 캐시 전체가 씻겨 나가 쓸모 있던 항목이 망가진다. ARC는 훑은 항목이 $T_1$에 들어가고 $T_2$(잦은 항목)은 건드리지 않으므로 이를 버틴다. 넋 줄이 이런 결이 생겼음을 알아채고 그에 맞추어 $p$을 고친다.

## 참고 문헌

- Megiddo, N. and Modha, D. S. (2003). "ARC: A self-tuning, low overhead replacement cache." *FAST*.

## 연습문제

**연습문제 1.**
ARC가 지니는 줄 넷을 밝히고 저마다의 몫을 풀어라. 매개변수 $p$은 무엇을 다스리는가?

??? success "연습문제 1 풀이"
    ARC는 줄 넷을 지닌다. (1) $T_1$ -- 한 번만 본, 최근에 닿은 항목(늦음 줄, 참 캐시). (2) $T_2$ -- 적어도 두 번 본, 최근에 닿은 항목(잦기 줄, 참 캐시). (3) $B_1$ -- $T_1$에서 내보내진 넋 항목(자료 없이 딸림 정보만). (4) $B_2$ -- $T_2$에서 내보내진 넋 항목(딸림 정보만). 캐시는 오직 $T_1 \cup T_2$에만 자료를 갈무리하며 $|T_1| + |T_2| \le c$(캐시 용량)이다. 매개변수 $p$($0 \le p \le c$)은 $T_1$의 목표 크기다. $|T_1| > p$이면 ARC는 $T_1$에서 내보내기를 즐기고, $|T_1| < p$이면 $T_2$에서 내보내기를 즐긴다. ARC는 $p$을 움직이며 맞춰 간다. $B_1$에서 맞으면 $p$이 오르고(늦음을 아껴 준다) $B_2$에서 맞으면 $p$이 내린다(잦기를 아껴 준다). $\square$

---

**연습문제 2.**
캐시 용량이 $c = 3$인 ARC에서 닿기 차례 $[A, B, C, D, A, B, E, A]$을 따라가라. 닿을 때마다 $T_1$, $T_2$, $B_1$, $B_2$, $p$의 상태를 보여라.

??? success "연습문제 2 풀이"
    처음: $T_1 = T_2 = B_1 = B_2 = \emptyset$, $p = 0$. A에 닿음: 빗나감, $T_1 = [A]$. B에 닿음: 빗나감, $T_1 = [A, B]$. C에 닿음: 빗나감, $T_1 = [A, B, C]$. D에 닿음: 빗나감, 캐시가 꽉 찼으므로 $T_1$의 가장 오래 안 쓴 것(A)을 $B_1$으로 내보낸다. $T_1 = [B, C, D]$, $B_1 = [A]$. A에 닿음: $B_1$에서 맞음 -- $p$을 $\max(1, |B_2|/|B_1|) = 1$만큼 올려 $p = 1$이다. A을 $T_2$의 가장 늦게 쓴 자리로 옮기고 $T_1$의 가장 오래 안 쓴 것(B)을 $B_1$으로 내보낸다. $T_1 = [C, D]$, $T_2 = [A]$, $B_1 = [B]$. B에 닿음: $B_1$에서 맞음 -- $p$을 2로 올린다. B을 $T_2$으로 옮기고 $T_1$의 가장 오래 안 쓴 것(C)을 $B_1$으로 내보낸다. $T_1 = [D]$, $T_2 = [A, B]$, $B_1 = [C]$. E에 닿음: 빗나감, $T_1$에서 내보낸다($|T_1| = 1 \le p = 2$이지만 쓸 수 있는 것이 D뿐이다). $T_1 = [E]$, $B_1 = [C, D]$, $T_2 = [A, B]$. A에 닿음: $T_2$에서 맞음, $T_2$의 가장 늦게 쓴 자리로 옮긴다. $T_2 = [B, A]$. $\square$

---

**연습문제 3.**
ARC의 맞춰 가는 장치가 모여듦을 증명하여라. 곧 일감이 온전히 늦음에 이끌리면(LRU가 가장 좋으면) $p$이 $c$으로 다가가고, 온전히 잦기에 이끌리면(LFU가 가장 좋으면) $p$이 $0$으로 다가감을 보여라.

??? success "연습문제 3 풀이"
    온전히 늦음이 이끄는 일감에서는 항목에 한 번 닿고 곧 쓴 뒤 다시 쓰지 않는다. $T_1$에서 내보내진 항목에는 다시 닿지 않으므로 $B_1$은 결코 맞음을 내지 않는다. 사이를 두고 다시 닿는 항목은 $T_2$에 나타나지만 판을 치는 결에서는 $T_2$이 더디게 찬다. $B_1$에서 넋이 맞으면 $p$이 오르고, $T_2$과 $B_2$이 성기게 차므로 $B_2$의 넋 맞음은 드물다. 때가 흐르면 $p$이 $c$으로 올라 캐시 자리의 거의 모두를 $T_1$(늦음 줄)에 내주며, 이는 LRU를 흉내 낸다. 거꾸로 잦기가 이끄는 일감에서는 인기 있는 항목이 $T_1 \to B_1 \to T_2$으로 돌고, $B_2$에 지난날 인기 있던 항목의 넋이 쌓인다. $B_2$의 맞음이 $p$을 내려 용량을 $T_1$에서 $T_2$으로 옮기고 끝내 $p \to 0$이 된다. 넋 목록이 어느 내보내기 방침이 이로웠을지에 대해 치우치지 않은 되먹임을 주므로 이 맞춰 감은 든든하다. $\square$

---

**연습문제 4.**
ARC와 LRU의 자리 덧듦을 견주어라. 실제로 만들 때 넋 목록의 크기를 자주 매어 두는 까닭은 무엇인가?

??? success "연습문제 4 풀이"
    LRU는 두 겹 이음줄 하나와 해시 표 하나를 지녀 온통 $O(c)$ 자리를 쓴다. ARC는 $T_1$과 $T_2$에 같은 얼개를(자료를 담는 온 용량 $c$) 지니고 여기에 넋 목록 $B_1$과 $B_2$을 더한다. 처음 꼴에서는 $|T_1| + |B_1| \le c$이고 $|T_2| + |B_2| \le c$이므로 넋 목록이 저마다 항목 $c$개까지 담을 수 있어, 캐시 항목 $c$개 말고도 딸림 정보 항목 $2c$개가 든다. 넋 항목은 (자료 없이) 열쇠만 갈무리하므로 덧듦은 대략 $2c \times (\text{열쇠 크기})$이다. 캐시가 크거나 열쇠가 크면 이는 딸림 정보 비용을 곱절로 만든다. 실제로 만들 때에는 기억을 줄이려고 넋 크기를 $c$의 한 조각(보기로 $c/4$)으로 매어 두고, 그 값으로 맞춰 감이 조금 더뎌짐을 받아들인다. 넋 항목은 거의 모든 일감에서 온전한 $2c$ 덧듦 없이도 훑기 버팀을 이루게 한다. $\square$

---

**연습문제 5.**
LRU는 나쁘게 돌지만 ARC는 높은 맞음률을 지키는 일감을 설계하여라. ARC가 앞서는 장치를 풀어라.

??? success "연습문제 5 풀이"
    크기 $c = 100$인 캐시와 두 마디를 오가는 일감을 여겨 보자. (1) 서로 다른 항목 200개에 줄줄이 닿는 "훑기" 마디(보기로 파일을 차례대로 읽기). (2) 붙박인 항목 50개에 되풀이해 닿는 "일하는 모임" 마디. LRU에서는 200 > 100이므로 훑기 마디가 일하는 모임의 항목을 모두 내보낸다. 일하는 모임 마디가 다시 시작되면 항목 50개를 다시 실을 때까지 닿을 때마다 빗나간다. ARC에서는 훑기 항목이 $T_1$에 들어갔다가 곧 $B_1$으로 내보내진다. 훑기 항목은 저마다 하나뿐이므로 $B_1$에서 다시 맞지 않고 따라서 $p$이 오르지 않는다. 일하는 모임의 항목은 (여러 번 보이므로) $T_2$에 쌓인다. 훑기 마디가 시작되면 ARC는 주로 $T_1$에서 내보내어 일하는 모임을 $T_2$에 지킨다. 훑는 동안 $B_2$에 일하는 모임의 넋이 쌓이면서 맞춰 가는 매개변수 $p$이 내려가 $T_2$을 지킨다. ARC는 두 마디 모두에서 거의 가장 좋은 맞음률을 이룬다. $\square$
