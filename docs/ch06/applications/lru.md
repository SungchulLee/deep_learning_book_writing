# 캐시 (LRU)

캐시는 값비싼 재계산이나 느린 입출력을 피하려고 최근에 쓴 데이터의 일부를 담아 둔다. 캐시가 용량에 다다르면 자리를 내려고 항목을 **내보내야** 한다. **가장 오래 안 쓴 것 먼저(LRU)** 정책은 가장 오래 손대지 않은 항목을 내보내며, 최근에 쓴 항목은 곧 다시 쓰일 가능성이 크다는 시간 지역성의 원리를 이용한다.

## 설계 문제

LRU 캐시는 두 연산을 각각 $O(1)$ 시간에 지원해야 한다.

- **get(key)**: 딸린 값을 돌려주고 그 항목을 가장 최근에 쓴 것으로 표시한다.
- **put(key, value)**: 항목을 넣거나 갱신하고, 캐시가 용량을 넘으면 가장 오래 안 쓴 항목을 내보낸다.

해시 테이블만으로도, 연결 리스트만으로도 두 연산을 동시에 $O(1)$으로 할 수 없다. 핵심은 둘을 결합하는 것이다.

## 해시 맵과 이중 연결 리스트

LRU 캐시는 두 자료 구조를 함께 쓴다.

1. **이중 연결 리스트**: 항목을 최근에 쓴 순서로 관리한다. 머리가 가장 최근에 쓴 것이고 꼬리가 가장 오래 안 쓴 것이다. 노드를 머리로 옮기거나 꼬리를 없애는 데 $O(1)$이 든다.
2. **해시 맵**: 키를 연결 리스트의 해당 노드에 대응시켜 키로 $O(1)$ 조회를 할 수 있게 한다.

이 조합으로 모든 연산이 $O(1)$ 시간이 된다.

| 연산 | 해시 맵의 구실 | 연결 리스트의 구실 |
|---|---|---|
| get(key) | $O(1)$에 노드를 찾는다 | $O(1)$에 머리로 옮긴다 |
| put(key, value) | $O(1)$에 노드를 넣거나 찾는다 | $O(1)$에 머리에 더하거나 꼬리를 내보낸다 |
| 내보내기 | $O(1)$에 맵에서 없앤다 | $O(1)$에 꼬리를 없앤다 |

## 공간 복잡도

용량이 $C$인 LRU 캐시는 해시 맵과 연결 리스트에 각각 많아야 $C$개의 항목을 담는다. 항목마다 연결 리스트 노드(키, 값, 포인터 둘)와 해시 맵 항목(키, 노드 포인터)이 필요하다. 전체 공간은 다음과 같다.

$$
O(C)
$$

## 내보내기 정책의 비교

| 정책 | 설명 | 강점 | 약점 |
|---|---|---|---|
| LRU | 가장 오래 안 쓴 것을 내보낸다 | 시간 지역성에 강하다 | 훑기 성 접근에 약하다 |
| FIFO | 가장 먼저 넣은 것을 내보낸다 | 간단하다 | 접근 양상을 무시한다 |
| LFU | 가장 적게 쓴 것을 내보낸다 | 접근이 치우칠 때 좋다 | 변화에 느리다 |
| 무작위 | 아무 항목이나 내보낸다 | 부담이 없다 | 예측할 수 없다 |

LRU는 단순함과 효과의 균형이 좋아 실무에서 가장 널리 쓰이는 정책이다.

## OrderedDict를 쓰는 파이썬 구현

파이썬의 `collections.OrderedDict`은 내부에서 해시 맵과 이중 연결 리스트를 결합하므로 간결한 LRU 캐시에 알맞다.

```python
"""
OrderedDict으로 구현한 LRU 캐시.

get과 put이 O(1)인 가장 오래 안 쓰인 것 내보내기 정책을
보인다.
"""

from collections import OrderedDict


# === LRU 캐시 ===

class LRUCache:
    """get과 put이 O(1)인 LRU 캐시."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        """열쇠의 값을 돌려주고 없으면 -1을 돌려준다. 최근에 쓴 것으로 표시한다."""
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        """열쇠-값 쌍을 넣거나 갱신한다. 꽉 차면 가장 오래 안 쓰인 것을 내보낸다."""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # 가장 오래된 것 없애기


# === 시연 ===

if __name__ == "__main__":
    cache = LRUCache(capacity=3)

    # 캐시 채우기
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    print(f"get('a'): {cache.get('a')}")  # 'a' 접근, 이제 가장 최근

    # 'd' 삽입 — 가장 오래 쓰이지 않은 'b'를 내보낸다
    cache.put("d", 4)
    print(f"get('b'): {cache.get('b')}")  # -1, 내보내짐
    print(f"get('c'): {cache.get('c')}")  # 3, 아직 있음
    print(f"get('d'): {cache.get('d')}")  # 4, 있음
```

**출력:**
```
get('a'): 1
get('b'): -1
get('c'): 3
get('d'): 4
```

## 바닥부터 구현하기

내부 작동을 이해하려고 날 이중 연결 리스트와 사전으로 만든 LRU 캐시를 보인다.

```python
"""
이중 연결 리스트와 해시 맵으로 밑바닥부터 만든 LRU 캐시.

위의 OrderedDict 기반 구현 밑에 깔린
자료 구조 설계를 드러낸다.
"""


# === 이중 연결 리스트 노드 ===

class DLLNode:
    """이중 연결 리스트의 노드."""

    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key=None, value=None):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


# === LRU 캐시 (밑바닥부터) ===

class LRUCacheDLL:
    """이중 연결 리스트와 해시 맵을 직접 쓰는 LRU 캐시."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}  # 열쇠 -> DLLNode
        # 보초 노드가 경계 상황을 간단하게 만든다
        self.head = DLLNode()  # 가장 최근 쪽 끝
        self.tail = DLLNode()  # 가장 오래된 쪽 끝
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        """연결 리스트에서 노드를 O(1)에 없앤다."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_head(self, node):
        """머리 보초 바로 뒤에 노드를 O(1)에 넣는다."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key):
        """열쇠의 값을 돌려주고, 없으면 -1을 돌려준다."""
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._remove(node)
        self._add_to_head(node)
        return node.value

    def put(self, key, value):
        """열쇠-값 쌍을 넣거나 갱신한다."""
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            self._remove(node)
            self._add_to_head(node)
        else:
            node = DLLNode(key, value)
            self.cache[key] = node
            self._add_to_head(node)
            if len(self.cache) > self.capacity:
                lru = self.tail.prev
                self._remove(lru)
                del self.cache[lru.key]


# === 시연 ===

if __name__ == "__main__":
    cache = LRUCacheDLL(capacity=2)
    cache.put(1, 10)
    cache.put(2, 20)
    print(f"get(1): {cache.get(1)}")
    cache.put(3, 30)  # 열쇠 2를 내보낸다
    print(f"get(2): {cache.get(2)}")
    print(f"get(3): {cache.get(3)}")
```

**출력:**
```
get(1): 10
get(2): -1
get(3): 30
```

## 참고 문헌

- [Introduction to Algorithms (CLRS), Chapter 11](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)


## 연습문제

**연습문제 1.**
캐시 (LRU)에 대해, 적재율이 $\alpha = 0.75$일 때 삽입과 조회의 기대 시간과 최악의 경우 시간을 계산하라.

??? success "연습문제 1 풀이"
    기대 시간은 충돌 해결 전략에 달렸으며 균등 해싱을 가정한다. 체이닝에서는 기대 시간이 $O(1 + \alpha) = O(1.75)$이다. 개방 주소법에서는 탐색에 실패할 때 기대 탐사 횟수가 $\approx 1/(1-\alpha) = 4$이다. 최악의 경우는 모든 키가 같은 칸으로 해시될 때의 $O(n)$이다.

---

**연습문제 2.**
캐시 (LRU)을(를) 써서 키 10, 22, 31, 4, 15, 28, 17을 크기가 7인 해시 테이블에 넣어라. 최종 테이블의 상태를 보여라.

??? success "연습문제 2 풀이"
    해시 함수 $h(k) = k \bmod 7$을 적용하고 이 쪽의 방법으로 충돌을 처리한다. 키마다 해시를 계산하고 충돌을 해결한 뒤 키를 놓는다. 최종 테이블의 내용을 보인다.

---

**연습문제 3.**
캐시 (LRU)은(는) 딥러닝의 임베딩 테이블에서 어떻게 쓰이는가? 토큰 $V = 50{,}000$개의 어휘를 $m = 30{,}000$개의 버킷에 대응시킬 때 충돌의 양상을 분석하라.

??? success "연습문제 3 풀이"
    $V/m \approx 1.67$이므로 비둘기집 원리에 의해 충돌이 반드시 생긴다. 버킷마다 평균 1.67개의 토큰이 같은 임베딩을 나누어 쓴다. 충돌률과 그것이 모델의 품질에 미치는 영향은 해시 함수의 품질과 임베딩의 차원에 달렸다(차원이 높을수록 충돌을 더 잘 견딘다).

---

**연습문제 4.**
$\alpha > 0.75$일 때 해시 테이블의 크기를 다시 잡으면 삽입의 상각 비용이 $O(1)$으로 유지됨을 증명하라.

??? success "연습문제 4 풀이"
    크기를 다시 잡는 사이(용량 $m$에서 $2m$까지)에 삽입이 $m/4$번 일어난다(적재율이 $0.375$에서 $0.75$로 간다). 크기 조정에는 $O(m)$이 든다. 삽입 하나당 상각된 크기 조정 비용은 $O(m)/(m/4) = O(4) = O(1)$이다. 여기에 (균등 해싱 아래) 삽입마다의 기대 비용 $O(1)$을 더하면 전체 상각 비용은 $O(1)$이다. $\square$