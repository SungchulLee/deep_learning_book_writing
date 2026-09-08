# 가장 드물게 쓴 캐시

LRU가 가장 오래도록 닿지 않은 항목을 내보내는 데 견주어, 어떤 일감은 최근이 아니더라도 여러 번 닿은 항목을 아껴 주는 방침을 바란다. **가장 드물게 쓴**(LFU) 방침은 닿은 셈이 가장 적은 항목을 내보낸다. 가장 작은 잦기를 여럿이 함께 지니면 LFU는 그 가운데 가장 오래 안 쓴 것을 내보내어 비긴 것을 가른다. 자료 얼개를 꼼꼼히 설계하면 LFU의 모든 연산이 $O(1)$ 때에 돈다.

---

## 1. 설계

$O(1)$ LFU 캐시는 서로 얽힌 자료 얼개 셋을 쓴다.

1. **열쇠-값 표**: 열쇠마다 그 값과 잦기와 자리를 맞댄다.
2. **잦기 표**: 닿은 셈 $f$마다 잦기가 $f$인 온 열쇠의 차례 있는 모임(두 겹 이음 목록)을 맞댄다.
3. **가장 작은 잦기 좇개**: 캐시에 담긴 항목 가운데 지금 가장 작은 잦기를 적어 두는 정수 하나 $f_{\min}$.

### 연산

**Get(열쇠)**:

1. 열쇠-값 표에서 열쇠를 찾는다. 없으면 빗나감을 돌려준다.
2. 열쇠의 잦기를 $f$에서 $f + 1$로 올린다.
3. 열쇠를 잦기 두레박 $f$에서 두레박 $f + 1$로 옮긴다.
4. 두레박 $f$이 비었고 $f = f_{\min}$이면 $f_{\min}$을 올린다.
5. 값을 돌려준다.

**Put(열쇠, 값)**:

1. 열쇠가 있으면 값을 고치고 Get과 똑같이 잦기를 올린다.
2. 캐시가 꽉 찼으면 $f_{\min}$ 두레박에서 가장 오래 안 쓴 항목을 내보낸다.
3. 새 열쇠를 잦기 1로 두레박 1에 넣는다. $f_{\min} = 1$로 둔다.

---

## 2. 구현

```python
"""
O(1) 가장 드물게 쓴(LFU) 캐시.

열쇠 찾기에는 해시 표를, 잦기-열쇠 맞댐에는 (잦기마다 LRU
차례를 지키려고 OrderedDict을 쓴) 표를 써서 O(1) get과 put을
이룬다.
"""

from collections import OrderedDict, defaultdict

# ===================================================================
# LFU 캐시
# ===================================================================

class LFUCache:
    """O(1) 연산을 주는 가장 드물게 쓴 캐시.

    인수:
        capacity: 항목의 최대 개수
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.key_to_val = {}
        self.key_to_freq = {}
        self.freq_to_keys = defaultdict(OrderedDict)
        self.min_freq = 0

    def get(self, key):
        """열쇠로 값을 얻는다. 빗나가면 -1을 돌려준다."""
        if key not in self.key_to_val:
            return -1
        self._increment_freq(key)
        return self.key_to_val[key]

    def put(self, key, value):
        """열쇠-값 짝을 넣거나 고친다."""
        if self.capacity <= 0:
            return

        if key in self.key_to_val:
            self.key_to_val[key] = value
            self._increment_freq(key)
            return

        # 용량에 다다랐으면 내보낸다
        if len(self.key_to_val) >= self.capacity:
            # min_freq 두레박에서 가장 오래 안 쓴 항목을 내보낸다
            evict_key, _ = self.freq_to_keys[self.min_freq].popitem(
                last=False)
            del self.key_to_val[evict_key]
            del self.key_to_freq[evict_key]

        # 새 열쇠를 넣는다
        self.key_to_val[key] = value
        self.key_to_freq[key] = 1
        self.freq_to_keys[1][key] = None
        self.min_freq = 1

    def _increment_freq(self, key):
        """열쇠를 잦기 f에서 f+1로 옮긴다."""
        freq = self.key_to_freq[key]
        self.key_to_freq[key] = freq + 1

        # 지금 잦기 두레박에서 뺀다
        del self.freq_to_keys[freq][key]
        if not self.freq_to_keys[freq]:
            del self.freq_to_keys[freq]
            if self.min_freq == freq:
                self.min_freq += 1

        # 다음 잦기 두레박에 넣는다
        self.freq_to_keys[freq + 1][key] = None

# ===================================================================
# 메인
# ===================================================================

if __name__ == "__main__":
    cache = LFUCache(3)

    # 항목을 넣는다
    cache.put("A", 1)
    cache.put("B", 2)
    cache.put("C", 3)

    # 잦기를 올리려고 A와 B에 닿는다
    cache.get("A")  # freq(A)=2
    cache.get("A")  # freq(A)=3
    cache.get("B")  # freq(B)=2

    # D를 넣는다 -- C를 내보낸다(freq=1로 가장 드묾)
    cache.put("D", 4)

    print("LFU Cache (capacity=3):")
    print(f"  get(A) = {cache.get('A')}")  # 1 (freq=4)
    print(f"  get(B) = {cache.get('B')}")  # 2 (freq=3)
    print(f"  get(C) = {cache.get('C')}")  # -1 (내보내짐)
    print(f"  get(D) = {cache.get('D')}")  # 4 (freq=2)

    # E를 넣는다 -- D를 내보낸다(이제 freq=2로 가장 드묾)
    cache.put("E", 5)
    print(f"  get(D) = {cache.get('D')}")  # -1 (내보내짐)
    print(f"  get(E) = {cache.get('E')}")  # 5
```

**출력:**
```
LFU Cache (capacity=3):
  get(A) = 1
  get(B) = 2
  get(C) = -1
  get(D) = 4
  get(D) = -1
  get(E) = 5
```

---

## 3. 복잡도

| 연산 | 때 | 자리 |
|---|---|---|
| `get` | $O(1)$ | -- |
| `put` | $O(1)$ 나눠 갚음 | -- |
| 온 자리 | -- | $O(c)$ |

$O(1)$ 매임은 해시 표 찾기와 잦기 두레박 안에서 벌이는 두 겹 이음 목록 연산에 기댄다.

---

## 4. LRU와 견주기

| 성질 | LRU | LFU |
|---|---|---|
| 내보내는 잣대 | 가장 오래 안 쓴 것 | 가장 드물게 쓴 것 |
| 아껴 주는 것 | 때 지역성 | 닿는 잦기 |
| 약점 | 훑기 더럽힘 | 낡은 높은 잦기 항목 |
| 복잡도 | $O(1)$ | 꼼꼼히 설계하면 $O(1)$ |

!!! warning "LFU의 캐시 더럽힘"
    지난날에는 자주 닿았으나 이제는 쓸모없는 항목이 높은 잦기 셈을 쌓아 두고 내보내지지 않으려 버틴다. 이 "캐시 더럽힘" 문제가 늙히기 장치나 [ARC](arc.md) 같은 섞음 방침을 불러낸다.

---

## 연습문제

**연습문제 1.**
$O(1)$ get과 put을 주는 LFU를 만드는 데 필요한 자료 얼개를 밝혀라. 각 얼개는 어떤 몫을 맡는가?

??? success "연습문제 1 풀이"
    얼개 셋이 있어야 한다. (1) 열쇠를 마디로 맞대는 **해시 표**. 캐시에 담긴 아무 항목이나 $O(1)$에 찾게 한다. (2) 잦기를 두 겹 이음 목록(잦기 두레박)으로 맞대는 **해시 표**. 두레박마다 그 닿은 셈을 지닌 온 항목을 넣은 차례대로 담는다. (3) 캐시에서 지금 가장 작은 잦기를 갈무리하는 **가장 작은 잦기 좇개**. `get(key)`에서는 마디를 찾아 지금 잦기 두레박에서 빼고 잦기를 올린 뒤 다음 잦기 두레박에 넣으며, 앞 두레박이 비었으면 가장 작은 잦기를 고친다. 꽉 찬 상태의 `put(key, value)`에서는 가장 작은 잦기 두레박에서 가장 오래 안 쓴 항목(그 두 겹 이음 목록의 꼬리)을 내보내고 새 항목을 두려고 가장 작은 잦기를 1로 되돌린다. 모든 연산이 손가락질 다루기와 해시 찾기이므로 저마다 $O(1)$이다. $\square$

---

**연습문제 2.**
용량이 3인 LFU 캐시에서 닿기 차례 put(A,1), put(B,2), put(C,3), get(B), put(D,4)을 따라가라. 어느 항목이 왜 내보내지는가?

??? success "연습문제 2 풀이"
    put(A,1) 뒤: 캐시 = {A:1}, 잦기: A=1, min_freq=1. put(B,2) 뒤: 캐시 = {A:1, B:2}, 잦기: A=1, B=1, min_freq=1. put(C,3) 뒤: 캐시 = {A:1, B:2, C:3}, 잦기: A=1, B=1, C=1, min_freq=1. get(B) 뒤: B의 잦기가 2로 오른다. 캐시 = {A:1, B:2, C:3}, 잦기: A=1, C=1, B=2, min_freq=1. put(D,4) 뒤: 캐시가 꽉 찼으므로 min_freq=1에서 가장 오래 안 쓴 항목을 내보낸다. 잦기 1 두레박은 넣은 차례대로 [A, C]을 담는다. 가장 오래된 것은 A이므로 A를 내보낸다. D를 잦기 1로 넣는다. 마지막 캐시 = {B:2, C:3, D:4}, 잦기: C=1, D=1, B=2, min_freq=1. $\square$

---

**연습문제 3.**
LFU 캐시의 "잦기 더럽힘" 문제를 밝히고 누그러뜨릴 길을 내놓아라.

??? success "연습문제 3 풀이"
    잦기 더럽힘은 지난날 크게 닿았으나 이제는 걸맞지 않은 항목이 높은 잦기 셈을 쌓을 때 생긴다. 이런 "낡은 인기" 항목은 그 셈이 새로 넣은 항목(잦기 1로 시작한다)을 훌쩍 넘으므로 거의 내보낼 수 없다. 그래서 싱싱하고 지금 걸맞은 항목이 캐시에 남지 못한다. 누그러뜨릴 길은 이렇다. (1) **늙히기/삭임**: 잦기 셈을 이따금(보기로 닿기 $T$번마다) 반으로 줄여 낡은 항목이 우위를 잃게 한다. (2) **창 LFU**: 온 삶 동안이 아니라 마지막 닿기 $W$번의 미끄러지는 창 안에서만 잦기를 좇는다. (3) **섞음 방침**: LRU와 LFU를 엮어(LRFU나 ARC처럼) 잦기와 늦음의 저울을 맞추어, 오직 잦기에만 바탕을 둔 판단이 판을 치지 못하게 한다. $\square$

---

**연습문제 4.**
잦기 두레박을 모두 훑지 않고도 get과 put 두 연산 동안 가장 작은 잦기 변수를 $O(1)$에 고칠 수 있음을 증명하여라.

??? success "연습문제 4 풀이"
    `min_freq` 변수는 오직 두 자리에서만 바뀐다. (1) **put(새 항목)**: 새 항목은 늘 잦기 1로 들어오므로 `min_freq = 1`이다. 이는 $O(1)$이다. (2) **get(있는 항목)**: 항목이 잦기 $f$에서 $f+1$로 옮겨 간다. $f =$ `min_freq`이고 뺀 뒤 잦기 $f$ 두레박이 비면 `min_freq`이 늘어야 한다. 여기서 요긴한 것은 그것이 (더 높이가 아니라) 꼭 $f + 1$로 는다는 점이다. 방금 옮긴 항목이 이제 잦기 $f + 1$에 있으므로 $f+1$ 두레박이 비어 있지 않음이 보장되기 때문이다. 훑을 까닭이 없다. 앞 두레박이 비었는지만 살펴 그렇다면 `min_freq`을 1만큼 올리면 된다. $f >$ `min_freq`이거나 앞 두레박이 비어 있지 않으면 `min_freq`은 그대로다. 두 경우 모두 $O(1)$이다. $\square$

---

**연습문제 5.**
지프 잦기 분포를 따르는(항목 $i$에 닿을 낌새가 $1/i$에 견주는) 항목 $n$개의 일감에서 LFU와 LRU를 견주어라. 크기 $k \ll n$인 캐시에서 어느 방침이 더 높은 맞음률을 이루며 그 까닭은 무엇인가?

??? success "연습문제 5 풀이"
    지프 분포에서는 적은 수의 항목이 닿기의 거의 모두를 차지한다(항목 1에 가장 자주 닿고, 항목 2는 대략 그 절반이며, 그렇게 이어진다). LFU가 여기에 더 알맞은 까닭은 가장 자주 닿는 항목을 곧바로 가려내어 지니기 때문이다. 데우는 동안이 지나면 LFU의 캐시는 (대략) 항목 1부터 $k$까지를 담아 가장 좋은 맞음률 $\sum_{i=1}^{k} 1/i \,/\, \sum_{i=1}^{n} 1/i$을 이룬다. LRU도 지프 일감에서 잘 도는데, 인기 있는 항목에 최근 닿았을 낌새가 크기 때문이다. 그러나 드문 항목에 어쩌다 닿을 때 인기 있는 항목이 밀려나기 쉽다. 지프 매개변수가 크게 치우치면($\alpha > 1$) 인기 있는 항목과 그렇지 않은 항목의 사이가 벌어지므로 LFU의 우위가 커진다. $\alpha$이 0에 가까우면(고른 것에 가까우면) 두 방침이 비슷하게 돈다. $\square$

## 정리하며

이 마당은 설계、구현、복잡도、LRU와 견주기을 차례로 짚었다.

**참고 문헌**

- Shah, K., Mitra, A., and Matani, D. (2010). "An O(1) algorithm for implementing the LFU cache eviction scheme." *Technical Report*.
