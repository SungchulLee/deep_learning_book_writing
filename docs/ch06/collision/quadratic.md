# 이차 탐사

선형 탐사는 걸음이 언제나 1이어서 찬 영역이 이어져 자라므로 일차 뭉침에 시달린다. 이차 탐사는 탐사 번호의 이차 함수를 오프셋으로 써서 탐사를 테이블 전체에 흩뿌리고 뭉치를 깬다. 대신 **이차 뭉침**이라는 더 약한 형태의 뭉침이 생기고 모든 칸을 방문한다는 보장이 없다.

---

## 1. 탐사 열

해시 함수 $h'(k)$과 테이블 크기 $m$에 대해 이차 탐사 열은 다음과 같다.

$$
h(k, i) = \bigl(h'(k) + c_1 i + c_2 i^2\bigr) \bmod m, \qquad i = 0, 1, 2, \ldots
$$

여기서 $c_1$과 $c_2$은 상수이고 $c_2 \neq 0$이다. 흔히 $c_1 = 0$, $c_2 = 1$으로 두어 다음의 더 간단한 형태를 쓴다.

$$
h(k, i) = \bigl(h'(k) + i^2\bigr) \bmod m
$$

부호를 번갈아 쓰는 변형도 널리 쓰인다. $i$번째 탐사가 $h'(k) + \lceil i/2 \rceil^2 \cdot (-1)^i$을 살핀다.

---

## 2. 이차 뭉침

같은 첫 칸에서 충돌하는 서로 다른 두 키 $k_1 \neq k_2$($h'(k_1) = h'(k_2)$)은 오프셋이 키가 아니라 $i$에만 달렸으므로 **같은** 이차 탐사 열을 따른다. 이 현상을 **이차 뭉침**이라 한다.

첫 해시값이 다른 키는 전혀 다른 열을 탐사하므로 이차 뭉침은 일차 뭉침보다 덜 심하다. 첫 해시값이 $m$가지이므로 이차 탐사는 서로 다른 탐사 열을 $m$개 만든다. 선형 탐사와 같은 수이지만 더 잘 퍼진다.

---

## 3. 방문 보장

선형 탐사와 달리 이차 탐사 열은 $m$개 칸을 모두 방문한다는 보장이 없다. 그러나 테이블의 크기 $m$이 소수이고 테이블이 많아야 절반 찼다면 처음 $\lfloor m/2 \rfloor$번의 탐사가 서로 다른 칸에 닿음이 보장된다.

!!! note "절반 방문 정리"
    $m$이 소수이고 $\alpha \le 1/2$이면, $i = 0, 1, \ldots, \lfloor m/2 \rfloor$에 대한 이차 탐사 열 $h'(k) + i^2 \pmod{m}$은 서로 다른 값 $\lfloor m/2 \rfloor + 1$개를 낸다.

??? note "증명 개요"
    $0 \le i < j \le \lfloor m/2 \rfloor$에 대해 $h'(k) + i^2 \equiv h'(k) + j^2 \pmod{m}$이라 하자. 그러면 $i^2 \equiv j^2 \pmod{m}$이므로 $m \mid (j-i)(j+i)$이다. $m$이 소수이고 $0 < j - i < m$, $0 < j + i < m$이므로 어느 인수도 $m$으로 나누어떨어지지 않아 모순이다. 따라서 $\lfloor m/2 \rfloor + 1$번의 탐사가 모두 서로 다르다.

모든 칸의 방문을 보장하는 다른 방법은 $m = 2^p$에 삼각수 탐사 열을 쓰는 것이다. 오프셋을 $0, 1, 3, 6, 10, \ldots$(곧 $i = 0, 1, \ldots, m-1$에 대해 $i(i+1)/2$)으로 두면 $m$개 인덱스 전체의 순열이 된다.

---

## 4. 기대 성능

이차 탐사의 성능은 (뭉침이 심한) 선형 탐사와 (균등 해싱에 가까운) 이중 해싱 사이에 있다. 기대 탐사 횟수의 간단한 닫힌 식은 없지만 실험 결과는 다음과 같다.

| 적재율 $\alpha$ | 이차 (실패) | 선형 (실패) | 이중 (실패) |
|---|---|---|---|
| 0.50 | ${\sim}2.2$ | 2.5 | 2.0 |
| 0.75 | ${\sim}4.6$ | 8.5 | 4.0 |
| 0.90 | ${\sim}11.4$ | 50.5 | 10.0 |

모든 적재율에서 이차 탐사가 선형 탐사보다 균등 해싱이라는 이상에 가깝다.

---

## 5. 파이썬 구현

```python
"""
이차 탐사 해시 표 구현.

이차 오프셋으로 일차 군집을 누그러뜨리는
충돌 해결을 보인다.
"""

# === 이차 탐사 표 ===

class QuadraticProbingTable:
    """이차 탐사를 쓰는 해시 표: h(k,i) = (h'(k) + i^2) mod m."""

    _EMPTY = None
    _DELETED = "<DELETED>"

    def __init__(self, capacity=11):
        # 용량이 소수이면 처음 m/2번의 탐사가 서로 다른 칸에 닿는다
        self.capacity = capacity
        self.size = 0
        self.table = [self._EMPTY] * capacity

    def _probe(self, key):
        """이차 탐사 순서를 만든다."""
        start = hash(key) % self.capacity
        for i in range(self.capacity):
            yield (start + i * i) % self.capacity

    def insert(self, key, value):
        """이차 탐사로 열쇠-값 쌍을 넣는다."""
        for idx in self._probe(key):
            entry = self.table[idx]
            if entry is self._EMPTY or entry is self._DELETED:
                self.table[idx] = (key, value)
                self.size += 1
                return
            if entry[0] == key:
                self.table[idx] = (key, value)  # 갱신
                return
        raise RuntimeError("Could not insert — table may be too full")

    def search(self, key):
        """열쇠를 찾아 값을 돌려주고, 없으면 None을 돌려준다."""
        for idx in self._probe(key):
            entry = self.table[idx]
            if entry is self._EMPTY:
                return None
            if entry is not self._DELETED and entry[0] == key:
                return entry[1]
        return None

    def delete(self, key):
        """묘비 표시로 열쇠를 지운다."""
        for idx in self._probe(key):
            entry = self.table[idx]
            if entry is self._EMPTY:
                return False
            if entry is not self._DELETED and entry[0] == key:
                self.table[idx] = self._DELETED
                self.size -= 1
                return True
        return False

    def load_factor(self):
        """지금 적재율을 돌려준다."""
        return self.size / self.capacity

# === 시연 ===

if __name__ == "__main__":
    ht = QuadraticProbingTable(capacity=11)

    words = ["hello", "world", "foo", "bar", "baz"]
    for i, w in enumerate(words):
        ht.insert(w, i * 10)

    print(f"Load factor: {ht.load_factor():.2f}")
    for w in words:
        print(f"search('{w}'): {ht.search(w)}")

    ht.delete("foo")
    print(f"After delete, search('foo'): {ht.search('foo')}")
    print(f"search('bar'): {ht.search('bar')}")  # 여전히 닿을 수 있음
```

**출력:**
```
Load factor: 0.45
search('hello'): 0
search('world'): 10
search('foo'): 20
search('bar'): 30
search('baz'): 40
After delete, search('foo'): None
search('bar'): 30
```

---

## 연습문제

**연습문제 1.**
이차 탐사에 대해, 적재율이 $\alpha = 0.75$일 때 삽입과 조회의 기대 시간과 최악의 경우 시간을 계산하라.

??? success "연습문제 1 풀이"
    기대 시간은 충돌 해결 전략에 달렸으며 균등 해싱을 가정한다. 체이닝에서는 기대 시간이 $O(1 + \alpha) = O(1.75)$이다. 개방 주소법에서는 탐색에 실패할 때 기대 탐사 횟수가 $\approx 1/(1-\alpha) = 4$이다. 최악의 경우는 모든 키가 같은 칸으로 해시될 때의 $O(n)$이다.

---

**연습문제 2.**
이차 탐사을(를) 써서 키 10, 22, 31, 4, 15, 28, 17을 크기가 7인 해시 테이블에 넣어라. 최종 테이블의 상태를 보여라.

??? success "연습문제 2 풀이"
    해시 함수 $h(k) = k \bmod 7$을 적용하고 이 쪽의 방법으로 충돌을 처리한다. 키마다 해시를 계산하고 충돌을 해결한 뒤 키를 놓는다. 최종 테이블의 내용을 보인다.

---

**연습문제 3.**
이차 탐사은(는) 딥러닝의 임베딩 테이블에서 어떻게 쓰이는가? 토큰 $V = 50{,}000$개의 어휘를 $m = 30{,}000$개의 버킷에 대응시킬 때 충돌의 양상을 분석하라.

??? success "연습문제 3 풀이"
    $V/m \approx 1.67$이므로 비둘기집 원리에 의해 충돌이 반드시 생긴다. 버킷마다 평균 1.67개의 토큰이 같은 임베딩을 나누어 쓴다. 충돌률과 그것이 모델의 품질에 미치는 영향은 해시 함수의 품질과 임베딩의 차원에 달렸다(차원이 높을수록 충돌을 더 잘 견딘다).

---

**연습문제 4.**
$\alpha > 0.75$일 때 해시 테이블의 크기를 다시 잡으면 삽입의 상각 비용이 $O(1)$으로 유지됨을 증명하라.

??? success "연습문제 4 풀이"
    크기를 다시 잡는 사이(용량 $m$에서 $2m$까지)에 삽입이 $m/4$번 일어난다(적재율이 $0.375$에서 $0.75$로 간다). 크기 조정에는 $O(m)$이 든다. 삽입 하나당 상각된 크기 조정 비용은 $O(m)/(m/4) = O(4) = O(1)$이다. 여기에 (균등 해싱 아래) 삽입마다의 기대 비용 $O(1)$을 더하면 전체 상각 비용은 $O(1)$이다. $\square$

## 정리하며

이 마당은 탐사 열、이차 뭉침、방문 보장、기대 성능을 차례로 짚었다.

**참고 문헌**

- [Introduction to Algorithms (CLRS), Chapter 11](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
