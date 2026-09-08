# 이중 해싱

선형 탐사는 일차 뭉침에, 이차 탐사는 이차 뭉침에 시달린다. 이중 해싱은 두 번째 해시 함수로 탐사의 걸음 크기를 정하여 탐사 열이 키 자체에 달리게 함으로써 둘 다 없앤다. 그래서 이중 해싱은 실무에서 가장 좋은 개방 주소법 전략 가운데 하나이다.

---

## 1. 탐사 열

이중 해싱은 해시 함수 $h_1$과 $h_2$을 쓴다. 키 $k$의 탐사 열은 다음과 같다.

$$
h(k, i) = \bigl(h_1(k) + i \cdot h_2(k)\bigr) \bmod m, \qquad i = 0, 1, 2, \ldots
$$

여기서 $m$은 테이블의 크기이다. 첫 탐사는 $h_1(k)$으로 가고 이후에는 매번 $h_2(k)$만큼 나아간다. 걸음 크기가 $k$에 달렸으므로 같은 첫 칸에서 충돌한 서로 다른 키가 서로 다른 탐사 열을 따른다.

---

## 2. 두 번째 해시 함수의 요건

이중 해싱이 테이블의 모든 칸을 살피려면 $h_2(k)$이 다음 두 조건을 만족해야 한다.

1. **0이 아닐 것**: 모든 키 $k$에 대해 $h_2(k) \neq 0$이어야 한다. 그렇지 않으면 탐사 열이 첫 칸에 머문다.
2. **$m$과 서로소일 것**: 탐사 열이 $\{0, 1, \ldots, m-1\}$의 완전한 순열을 만들려면 $h_2(k)$이 $m$과 서로소여야 한다.

두 조건을 모두 보장하는 흔한 전략은 $m$을 소수로 잡고 다음과 같이 정의하는 것이다.

$$
h_2(k) = 1 + (k \bmod (m - 1))
$$

$m$이 소수이고 $1 \le h_2(k) \le m - 1$이므로 모든 키에 대해 $\gcd(h_2(k), m) = 1$이다.

다른 방법은 (나머지 연산을 빠르게 하려고) $m = 2^p$으로 두고 $h_2(k)$이 언제나 홀수를 돌려주게 하는 것이다.

$$
h_2(k) = 2 \cdot (k \bmod \lfloor m/2 \rfloor) + 1
$$

---

## 3. 이중 해싱이 뭉침을 피하는 까닭

**일차 뭉침**(선형 탐사): 해시값이 달라도 같은 칸에서 충돌한 키들이 그 뒤로 똑같은 탐사 열을 따른다.

**이차 뭉침**(이차 탐사): 이차 오프셋이 키가 아니라 탐사 번호 $i$에만 달렸으므로 $h_1(k)$이 같은 키들이 같은 탐사 열을 따른다.

**이중 해싱**: 걸음 크기 $h_2(k)$이 키마다 다르므로 $h_1(k)$에서 충돌한 키도 다음 탐사에서 곧바로 갈라진다. 서로 다른 탐사 열의 수는 다음과 같다.

$$
\Theta(m^2)
$$

선형 탐사나 이차 탐사의 $\Theta(m)$과 견주어 보라.

---

## 4. 기대 탐사 횟수

균등 해싱 가정 아래 적재율이 $\alpha = n/m$인 테이블의 기대 탐사 횟수는 다음과 같다.

**탐색 실패:**

$$
E[\text{probes}] \le \frac{1}{1 - \alpha}
$$

**탐색 성공:**

$$
E[\text{probes}] \le \frac{1}{\alpha} \ln \frac{1}{1 - \alpha}
$$

이 식들은 이상적인 균등 해싱의 한계와 같으며, 이중 해싱이 이론적 최적에 가깝게 다가감을 보여 준다.

| 적재율 $\alpha$ | 실패한 탐침 | 성공한 탐침 |
|---|---|---|
| 0.50 | 2.0 | 1.39 |
| 0.75 | 4.0 | 1.85 |
| 0.90 | 10.0 | 2.56 |
| 0.95 | 20.0 | 3.15 |

---

## 5. 삽입과 삭제

**삽입**: 빈 칸이나 묘비를 만날 때까지 탐사 열 $h(k, 0), h(k, 1), \ldots$을 따라간 뒤 그 자리에 키를 놓는다.

**탐색**: 키를 찾거나, 빈 칸에 이르거나(키가 없다는 뜻), 칸 $m$개를 모두 살필 때까지 같은 탐사 열을 따라간다.

**삭제**: 모든 개방 주소법이 그렇듯 그냥 없애면 탐사 사슬이 끊긴다. 탐색이 지워진 칸을 지나 이어지도록 **묘비**(삭제 표시)를 놓아야 한다. 묘비는 삽입할 때 다시 쓸 수 있지만 탐색에서는 여전히 찬 칸으로 센다.

---

## 6. 파이썬 구현

```python
"""
열린 주소법을 위한 이중 해싱 구현.

해시 함수 두 개로 탐사 순서를 만들어 일차 군집과
이차 군집을 모두 피한다.
"""

# === 이중 해싱 표 ===

class DoubleHashTable:
    """이중 해싱을 쓰는 열린 주소법 해시 표."""

    _EMPTY = None
    _DELETED = "<DELETED>"

    def __init__(self, capacity=11):
        # 가장 좋은 결과를 얻으려면 용량이 소수여야 한다
        self.capacity = capacity
        self.size = 0
        self.table = [self._EMPTY] * capacity

    def _h1(self, key):
        return hash(key) % self.capacity

    def _h2(self, key):
        # 0이 아니고 용량과 서로소가 되게 한다 (용량은 소수)
        return 1 + (hash(key) % (self.capacity - 1))

    def _probe(self, key):
        """주어진 열쇠의 탐사 순서를 만든다."""
        start = self._h1(key)
        step = self._h2(key)
        for i in range(self.capacity):
            yield (start + i * step) % self.capacity

    def insert(self, key, value):
        """이중 해싱으로 열쇠-값 쌍을 넣는다."""
        for idx in self._probe(key):
            entry = self.table[idx]
            if entry is self._EMPTY or entry is self._DELETED:
                self.table[idx] = (key, value)
                self.size += 1
                return
            if entry[0] == key:
                self.table[idx] = (key, value)  # 갱신
                return
        raise RuntimeError("Hash table is full")

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

# === 시연 ===

if __name__ == "__main__":
    ht = DoubleHashTable(capacity=11)

    data = [("alice", 100), ("bob", 200), ("carol", 300),
            ("dave", 400), ("eve", 500)]
    for k, v in data:
        ht.insert(k, v)

    print(f"Size: {ht.size}")
    for k, _ in data:
        print(f"search('{k}'): {ht.search(k)}")

    ht.delete("carol")
    print(f"After delete, search('carol'): {ht.search('carol')}")
    print(f"search('dave'): {ht.search('dave')}")  # 여전히 닿을 수 있음
```

**출력:**
```
Size: 5
search('alice'): 100
search('bob'): 200
search('carol'): 300
search('dave'): 400
search('eve'): 500
After delete, search('carol'): None
search('dave'): 400
```

---

## 연습문제

**연습문제 1.**
이중 해싱에 대해, 적재율이 $\alpha = 0.75$일 때 삽입과 조회의 기대 시간과 최악의 경우 시간을 계산하라.

??? success "연습문제 1 풀이"
    기대 시간은 충돌 해결 전략에 달렸으며 균등 해싱을 가정한다. 체이닝에서는 기대 시간이 $O(1 + \alpha) = O(1.75)$이다. 개방 주소법에서는 탐색에 실패할 때 기대 탐사 횟수가 $\approx 1/(1-\alpha) = 4$이다. 최악의 경우는 모든 키가 같은 칸으로 해시될 때의 $O(n)$이다.

---

**연습문제 2.**
이중 해싱을(를) 써서 키 10, 22, 31, 4, 15, 28, 17을 크기가 7인 해시 테이블에 넣어라. 최종 테이블의 상태를 보여라.

??? success "연습문제 2 풀이"
    해시 함수 $h(k) = k \bmod 7$을 적용하고 이 쪽의 방법으로 충돌을 처리한다. 키마다 해시를 계산하고 충돌을 해결한 뒤 키를 놓는다. 최종 테이블의 내용을 보인다.

---

**연습문제 3.**
이중 해싱은(는) 딥러닝의 임베딩 테이블에서 어떻게 쓰이는가? 토큰 $V = 50{,}000$개의 어휘를 $m = 30{,}000$개의 버킷에 대응시킬 때 충돌의 양상을 분석하라.

??? success "연습문제 3 풀이"
    $V/m \approx 1.67$이므로 비둘기집 원리에 의해 충돌이 반드시 생긴다. 버킷마다 평균 1.67개의 토큰이 같은 임베딩을 나누어 쓴다. 충돌률과 그것이 모델의 품질에 미치는 영향은 해시 함수의 품질과 임베딩의 차원에 달렸다(차원이 높을수록 충돌을 더 잘 견딘다).

---

**연습문제 4.**
$\alpha > 0.75$일 때 해시 테이블의 크기를 다시 잡으면 삽입의 상각 비용이 $O(1)$으로 유지됨을 증명하라.

??? success "연습문제 4 풀이"
    크기를 다시 잡는 사이(용량 $m$에서 $2m$까지)에 삽입이 $m/4$번 일어난다(적재율이 $0.375$에서 $0.75$로 간다). 크기 조정에는 $O(m)$이 든다. 삽입 하나당 상각된 크기 조정 비용은 $O(m)/(m/4) = O(4) = O(1)$이다. 여기에 (균등 해싱 아래) 삽입마다의 기대 비용 $O(1)$을 더하면 전체 상각 비용은 $O(1)$이다. $\square$

## 정리하며

이 마당은 탐사 열、두 번째 해시 함수의 요건、이중 해싱이 뭉침을 피하는 까닭、기대 탐사 횟수을 차례로 짚었다.

**참고 문헌**

- [Introduction to Algorithms (CLRS), Chapter 11](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
