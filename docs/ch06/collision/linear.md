# 선형 탐사

선형 탐사는 가장 간단한 개방 주소법이다. 충돌이 일어나면 다음 칸, 또 그다음 칸을 차례로 살피며 빈 칸을 찾을 때까지 테이블을 돌아 감는다. 이렇게 차례대로 접근하므로 캐시에 아주 친화적이어서, 이론적으로 뭉침에 약함에도 실무에서는 가장 빠른 해시 테이블 전략인 경우가 많다.

---

## 1. 탐사 열

해시 함수 $h'(k)$과 크기 $m$의 테이블이 주어지면 선형 탐사는 다음 탐사 열을 정의한다.

$$
h(k, i) = \bigl(h'(k) + i\bigr) \bmod m, \qquad i = 0, 1, 2, \ldots, m - 1
$$

첫 탐사는 칸 $h'(k)$으로 가고, 이후에는 $m$으로 나눈 나머지 기준으로 다음 칸으로 옮겨 간다. 걸음 크기가 언제나 1이므로 탐사 열은 되풀이되기 전에 모든 칸을 정확히 한 번씩 방문한다.

---

## 2. 일차 뭉침

선형 탐사의 주된 단점은 **일차 뭉침**이다. 찬 칸이 이어진 덩어리(**뭉치**)가 생기면, 그 안 어느 자리로든 해시되는 새 키가 뭉치를 한 칸 늘린다. 뭉치가 길수록 새 키를 흡수할 가능성이 커져 더욱 길어진다.

형식적으로, 뭉치의 길이가 $\ell$이면 다음 삽입이 이 뭉치를 늘릴 확률은 다음에 비례한다.

$$
\frac{\ell + 1}{m}
$$

뭉치 안의 $\ell$개 칸이나 바로 뒤의 한 칸으로 해시되는 키가 모두 뭉치에 붙기 때문이다. 이러한 양의 되먹임 고리가 일차 뭉침의 결정적인 특징이다.

---

## 3. 기대 탐사 횟수

커누스의 분석(1963)은 적재율이 $\alpha = n/m$인 테이블에 대해 균등 해싱 가정 아래의 기대 탐사 횟수를 준다.

**탐색 실패 (또는 삽입):**

$$
E[\text{probes}] \approx \frac{1}{2}\left(1 + \frac{1}{(1-\alpha)^2}\right)
$$

**탐색 성공:**

$$
E[\text{probes}] \approx \frac{1}{2}\left(1 + \frac{1}{1-\alpha}\right)
$$

이들은 균등 해싱의 대응 식($\frac{1}{1-\alpha}$과 $\frac{1}{\alpha}\ln\frac{1}{1-\alpha}$)보다 훨씬 빠르게 커지며 뭉침의 비용을 드러낸다.

| 적재율 $\alpha$ | 실패 (선형) | 실패 (균등) |
|---|---|---|
| 0.50 | 2.50 | 2.00 |
| 0.75 | 8.50 | 4.00 |
| 0.90 | 50.50 | 10.00 |
| 0.95 | 200.50 | 20.00 |

실용적인 결론은 이렇다. 선형 탐사가 효율을 지키려면 적재율을 $0.7$ 아래로 유지하라.

---

## 4. 연산

**삽입**: $h'(k)$에서 시작하여 빈 칸이나 묘비를 만날 때까지 탐사 열을 따라간 뒤 그 자리에 키를 놓는다.

**탐색**: $h'(k)$에서 시작하여 키를 찾거나 빈 칸(없다는 뜻)에 이를 때까지 탐사 열을 따라간다. 묘비는 탐색을 끝내지 않는다.

**삭제**: 개방 주소 테이블에서 키를 없앨 때에는 조심해야 한다. 두 가지 전략이 있다.

1. **묘비**: 칸을 삭제됨으로 표시한다. 탐색은 묘비를 지나가고 삽입은 다시 쓸 수 있다. 시간이 갈수록 묘비가 쌓여 성능이 나빠진다.
2. **뒤로 밀기**: 키를 없앤 뒤 뭉치의 뒤쪽 키를 앞으로 당겨 틈을 메운다. 묘비를 피할 수 있지만 삭제마다 일이 더 든다.

---

## 5. 캐시 성능

선형 탐사는 메모리를 차례대로 읽는다. 캐시 줄이 64바이트인 요즘 하드웨어에서는 캐시 실패 한 번에 (키가 8바이트라 할 때) 항목 8개를 불러오므로, 첫 탐사가 뒤이은 여러 탐사에 필요한 데이터를 함께 가져온다. 그래서 실무에서 선형 탐사는 체이닝이나 이중 해싱보다 훨씬 빠르며, 특히 키가 작을 때 그렇다.

---

## 6. 파이썬 구현

```python
"""
선형 탐사 해시 표 구현.

차례대로 탐사하고 묘비로 지우는 가장 단순한
열린 주소법 전략을 보인다.
"""

# === 선형 탐사 표 ===

class LinearProbingTable:
    """충돌 해결에 선형 탐사를 쓰는 해시 표."""

    _EMPTY = None
    _DELETED = "<DELETED>"

    def __init__(self, capacity=8):
        self.capacity = capacity
        self.size = 0
        self.table = [self._EMPTY] * capacity

    def _probe(self, key):
        """선형 탐사 순서를 만든다."""
        start = hash(key) % self.capacity
        for i in range(self.capacity):
            yield (start + i) % self.capacity

    def insert(self, key, value):
        """선형 탐사로 열쇠-값 쌍을 넣는다."""
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

    def display(self):
        """살펴볼 수 있도록 표의 내용을 보인다."""
        for i, entry in enumerate(self.table):
            status = "empty" if entry is self._EMPTY else (
                "deleted" if entry is self._DELETED else f"{entry}")
            print(f"  [{i}] {status}")

# === 시연 ===

if __name__ == "__main__":
    ht = LinearProbingTable(capacity=8)

    for k, v in [("cat", 1), ("dog", 2), ("rat", 3), ("bat", 4)]:
        ht.insert(k, v)

    print("Table after insertions:")
    ht.display()
    print(f"\nsearch('dog'): {ht.search('dog')}")
    print(f"search('fox'): {ht.search('fox')}")

    ht.delete("dog")
    print(f"\nAfter deleting 'dog':")
    print(f"search('dog'): {ht.search('dog')}")
    print(f"search('rat'): {ht.search('rat')}")
```

**출력:**
```
Table after insertions:
  [0] ('bat', 4)
  [1] empty
  [2] ('cat', 1)
  [3] ('dog', 2)
  [4] ('rat', 3)
  [5] empty
  [6] empty
  [7] empty

search('dog'): 2
search('fox'): None

After deleting 'dog':
search('dog'): None
search('rat'): 3
```

---

## 연습문제

**연습문제 1.**
선형 탐사에 대해, 적재율이 $\alpha = 0.75$일 때 삽입과 조회의 기대 시간과 최악의 경우 시간을 계산하라.

??? success "연습문제 1 풀이"
    기대 시간은 충돌 해결 전략에 달렸으며 균등 해싱을 가정한다. 체이닝에서는 기대 시간이 $O(1 + \alpha) = O(1.75)$이다. 개방 주소법에서는 탐색에 실패할 때 기대 탐사 횟수가 $\approx 1/(1-\alpha) = 4$이다. 최악의 경우는 모든 키가 같은 칸으로 해시될 때의 $O(n)$이다.

---

**연습문제 2.**
선형 탐사을(를) 써서 키 10, 22, 31, 4, 15, 28, 17을 크기가 7인 해시 테이블에 넣어라. 최종 테이블의 상태를 보여라.

??? success "연습문제 2 풀이"
    해시 함수 $h(k) = k \bmod 7$을 적용하고 이 쪽의 방법으로 충돌을 처리한다. 키마다 해시를 계산하고 충돌을 해결한 뒤 키를 놓는다. 최종 테이블의 내용을 보인다.

---

**연습문제 3.**
선형 탐사은(는) 딥러닝의 임베딩 테이블에서 어떻게 쓰이는가? 토큰 $V = 50{,}000$개의 어휘를 $m = 30{,}000$개의 버킷에 대응시킬 때 충돌의 양상을 분석하라.

??? success "연습문제 3 풀이"
    $V/m \approx 1.67$이므로 비둘기집 원리에 의해 충돌이 반드시 생긴다. 버킷마다 평균 1.67개의 토큰이 같은 임베딩을 나누어 쓴다. 충돌률과 그것이 모델의 품질에 미치는 영향은 해시 함수의 품질과 임베딩의 차원에 달렸다(차원이 높을수록 충돌을 더 잘 견딘다).

---

**연습문제 4.**
$\alpha > 0.75$일 때 해시 테이블의 크기를 다시 잡으면 삽입의 상각 비용이 $O(1)$으로 유지됨을 증명하라.

??? success "연습문제 4 풀이"
    크기를 다시 잡는 사이(용량 $m$에서 $2m$까지)에 삽입이 $m/4$번 일어난다(적재율이 $0.375$에서 $0.75$로 간다). 크기 조정에는 $O(m)$이 든다. 삽입 하나당 상각된 크기 조정 비용은 $O(m)/(m/4) = O(4) = O(1)$이다. 여기에 (균등 해싱 아래) 삽입마다의 기대 비용 $O(1)$을 더하면 전체 상각 비용은 $O(1)$이다. $\square$

## 정리하며

이 마당은 탐사 열、일차 뭉침、기대 탐사 횟수、연산을 차례로 짚었다.

**참고 문헌**

- [Introduction to Algorithms (CLRS), Chapter 11](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Knuth, D. E. *The Art of Computer Programming*, Vol. 3: Sorting and Searching.
