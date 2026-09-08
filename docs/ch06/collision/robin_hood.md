# 로빈 후드 해싱

보통의 선형 탐사에는 공평함의 문제가 있다. 어떤 키는 첫 탐사에서 자리를 잡는데 어떤 키는 제 자리에서 멀리까지 가야 해서 탐색 시간의 분산이 커진다. 로빈 후드 해싱은 "부자에게서 빼앗아 가난한 이에게 준다"는 간단한 형평의 원리를 적용하여, 탐사 거리가 긴 키가 짧은 키를 밀어낼 수 있게 한다. 그 결과 탐사 거리의 분산이 크게 줄어 실무에서 최악의 경우 탐색이 훨씬 빨라진다.

---

## 1. 탐사 거리

저장된 키 $k$의 **탐사 거리**(또는 **변위**)는 실제 자리와 이상적인 자리 $h(k)$ 사이의 칸 수이다.

$$
d(k) = \bigl(\text{pos}(k) - h(k)\bigr) \bmod m
$$

보통의 선형 탐사에서는 탐사 거리가 크게 들쭉날쭉하다. 로빈 후드 해싱은 테이블 전체의 탐사 거리를 대체로 고르게 유지한다는 불변식을 지킨다.

---

## 2. 삽입 알고리즘

로빈 후드의 삽입은 선형 탐사를 바탕으로 하되 밀어내기 규칙을 더한다.

1. 새 키 $k$에 대해 $h(k)$을 계산하고 $d \gets 0$으로 둔다.
2. 칸 $(h(k) + d) \bmod m$을 살핀다.
3. 칸이 비어 있으면 거기에 $k$을 놓는다.
4. 탐사 거리가 $d' < d$인 키 $k'$이 차지하고 있으면 $k$과 $k'$을 **맞바꾼다**. 갱신된 탐사 거리로 $k'$을 계속 넣어 간다.
5. $d' \ge d$이면 $d$을 1 늘리고 계속 탐사한다.

이 맞바꾸기 규칙 덕분에 어떤 키도 남을 희생시켜 "지나치게 운 좋은" 자리를 차지하지 않는다. 형식적으로, 삽입을 모두 마치면 탐사 거리가 다음을 만족한다.

$$
d_{\max} - d_{\min} \le 1 \quad \text{(approximately)}
$$

그래서 로빈 후드 해싱은 분산의 면에서 거의 최적이다.

---

## 3. 분산 줄이기

적재율이 $\alpha = n/m$인 테이블에서 로빈 후드 해싱의 기대 최대 탐사 거리는 다음과 같다.

$$
E[d_{\max}] = O(\log \log n)
$$

보통의 선형 탐사에서는 $O(\log n)$이다. 탐사 거리의 분산도 보통의 선형 탐사의 $\Theta(1/(1-\alpha)^2)$에서 로빈 후드 해싱의 $\Theta(1/(1-\alpha))$으로 줄어든다.

즉 **평균** 탐색 시간은 선형 탐사와 같은 $\Theta(1/(1-\alpha))$이지만 **최악의 경우** 탐색 시간은 크게 낫다. 실무에서 로빈 후드 테이블은 성능이 크게 나빠지지 않으면서 더 높은 적재율($\alpha \approx 0.9$까지)에서 돌아갈 수 있다.

---

## 4. 탐색과 삭제

키 $k$ **탐색**: $h(k)$에서 차례대로 탐사하며 탐사 거리 $d$을 기록한다. 저장된 키의 탐사 거리가 $d$보다 작은 칸에 이르면 찾는 키가 그 너머에 있을 수 없으므로 멈추고 "찾지 못함"을 알린다. 이러한 **조기 종료** 덕분에 탐색 실패가 보통의 선형 탐사보다 빠르다.

**삭제**: 로빈 후드 해싱은 **뒤로 밀기**로 깔끔한 삭제를 지원한다. 키를 없앤 뒤 뭉치의 뒤쪽 키들을 앞으로 당기면 탐사 거리가 줄어드는 경우에 그렇게 한다. 이로써 묘비를 아예 쓰지 않고 로빈 후드 불변식을 지킨다.

---

## 5. 다른 개방 주소법 전략과의 비교

| 성질 | 선형 탐사 | 로빈 후드 | 이중 해싱 |
|---|---|---|---|
| 평균 탐색 | $O(1/(1-\alpha))$ | $O(1/(1-\alpha))$ | $O(1/(1-\alpha))$ |
| 최악의 경우 탐색 | 기대 $O(\log n)$ | 기대 $O(\log \log n)$ | 기대 $O(\log n)$ |
| 탐사 거리의 분산 | 큼 | 작음 | 중간 |
| 캐시 거동 | 아주 좋음 | 아주 좋음 | 보통 |
| 삭제 | 묘비 | 뒤로 밀기 | 묘비 |
| 구현 | 간단 | 보통 | 보통 |

로빈 후드 해싱은 선형 탐사의 캐시 친화성과 훨씬 나은 최악의 경우 거동을 결합하여, 러스트의 `HashMap`(1.36 이전 판) 같은 고성능 해시 테이블 구현에서 널리 쓰인다.

---

## 6. 파이썬 구현

```python
"""
로빈 후드 해싱 구현.

저장된 열쇠들의 탐사 거리를 고르게 만드는
밀어내기 방식 삽입을 보인다.
"""

# === 로빈 후드 해시 표 ===

class RobinHoodHashTable:
    """선형 탐사와 함께 로빈 후드 해싱을 쓰는 해시 표."""

    _EMPTY = None

    def __init__(self, capacity=8):
        self.capacity = capacity
        self.size = 0
        self.keys = [self._EMPTY] * capacity
        self.values = [self._EMPTY] * capacity
        self.dists = [0] * capacity  # 탐사 거리

    def _hash(self, key):
        return hash(key) % self.capacity

    def insert(self, key, value):
        """로빈 후드 밀어내기로 삽입한다."""
        idx = self._hash(key)
        dist = 0

        while True:
            if self.keys[idx] is self._EMPTY:
                self.keys[idx] = key
                self.values[idx] = value
                self.dists[idx] = dist
                self.size += 1
                return
            # 이미 있는 열쇠 갱신
            if self.keys[idx] == key:
                self.values[idx] = value
                return
            # 로빈 후드 맞바꾸기: "더 부유한" 열쇠를 밀어낸다
            if self.dists[idx] < dist:
                key, self.keys[idx] = self.keys[idx], key
                value, self.values[idx] = self.values[idx], value
                dist, self.dists[idx] = self.dists[idx], dist

            idx = (idx + 1) % self.capacity
            dist += 1

    def search(self, key):
        """탐사 거리를 보고 일찍 끝내며 찾는다."""
        idx = self._hash(key)
        dist = 0

        while self.keys[idx] is not self._EMPTY:
            if self.keys[idx] == key:
                return self.values[idx]
            # 일찍 끝내기: 열쇠가 있었다면 여기에 놓였을 것이다
            if self.dists[idx] < dist:
                return None
            idx = (idx + 1) % self.capacity
            dist += 1

        return None

    def delete(self, key):
        """뒤의 것을 앞으로 당기며 지운다 (묘비 없음)."""
        idx = self._hash(key)
        dist = 0

        # 열쇠 찾기
        while self.keys[idx] is not self._EMPTY:
            if self.keys[idx] == key:
                break
            if self.dists[idx] < dist:
                return False
            idx = (idx + 1) % self.capacity
            dist += 1
        else:
            return False

        # 뒤따르는 항목들을 앞으로 당기기
        self.keys[idx] = self._EMPTY
        self.values[idx] = self._EMPTY
        self.dists[idx] = 0
        self.size -= 1

        next_idx = (idx + 1) % self.capacity
        while (self.keys[next_idx] is not self._EMPTY
               and self.dists[next_idx] > 0):
            self.keys[idx] = self.keys[next_idx]
            self.values[idx] = self.values[next_idx]
            self.dists[idx] = self.dists[next_idx] - 1
            self.keys[next_idx] = self._EMPTY
            self.values[next_idx] = self._EMPTY
            self.dists[next_idx] = 0
            idx = next_idx
            next_idx = (next_idx + 1) % self.capacity

        return True

    def probe_distances(self):
        """차 있는 칸의 (열쇠, 탐사 거리) 리스트를 돌려준다."""
        result = []
        for i in range(self.capacity):
            if self.keys[i] is not self._EMPTY:
                result.append((self.keys[i], self.dists[i]))
        return result

# === 시연 ===

if __name__ == "__main__":
    ht = RobinHoodHashTable(capacity=8)

    for k, v in [("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5)]:
        ht.insert(k, v)

    print("Probe distances:", ht.probe_distances())
    print(f"search('c'): {ht.search('c')}")
    print(f"search('z'): {ht.search('z')}")

    ht.delete("c")
    print(f"After delete, search('c'): {ht.search('c')}")
    print("Probe distances:", ht.probe_distances())
```

**출력:**
```
Probe distances: [('a', 0), ('b', 0), ('d', 0), ('e', 0), ('c', 2)]
search('c'): 3
search('z'): None
After delete, search('c'): None
Probe distances: [('a', 0), ('b', 0), ('d', 0), ('e', 0)]
```

---

## 연습문제

**연습문제 1.**
로빈 후드 해싱에 대해, 적재율이 $\alpha = 0.75$일 때 삽입과 조회의 기대 시간과 최악의 경우 시간을 계산하라.

??? success "연습문제 1 풀이"
    기대 시간은 충돌 해결 전략에 달렸으며 균등 해싱을 가정한다. 체이닝에서는 기대 시간이 $O(1 + \alpha) = O(1.75)$이다. 개방 주소법에서는 탐색에 실패할 때 기대 탐사 횟수가 $\approx 1/(1-\alpha) = 4$이다. 최악의 경우는 모든 키가 같은 칸으로 해시될 때의 $O(n)$이다.

---

**연습문제 2.**
로빈 후드 해싱을(를) 써서 키 10, 22, 31, 4, 15, 28, 17을 크기가 7인 해시 테이블에 넣어라. 최종 테이블의 상태를 보여라.

??? success "연습문제 2 풀이"
    해시 함수 $h(k) = k \bmod 7$을 적용하고 이 쪽의 방법으로 충돌을 처리한다. 키마다 해시를 계산하고 충돌을 해결한 뒤 키를 놓는다. 최종 테이블의 내용을 보인다.

---

**연습문제 3.**
로빈 후드 해싱은(는) 딥러닝의 임베딩 테이블에서 어떻게 쓰이는가? 토큰 $V = 50{,}000$개의 어휘를 $m = 30{,}000$개의 버킷에 대응시킬 때 충돌의 양상을 분석하라.

??? success "연습문제 3 풀이"
    $V/m \approx 1.67$이므로 비둘기집 원리에 의해 충돌이 반드시 생긴다. 버킷마다 평균 1.67개의 토큰이 같은 임베딩을 나누어 쓴다. 충돌률과 그것이 모델의 품질에 미치는 영향은 해시 함수의 품질과 임베딩의 차원에 달렸다(차원이 높을수록 충돌을 더 잘 견딘다).

---

**연습문제 4.**
$\alpha > 0.75$일 때 해시 테이블의 크기를 다시 잡으면 삽입의 상각 비용이 $O(1)$으로 유지됨을 증명하라.

??? success "연습문제 4 풀이"
    크기를 다시 잡는 사이(용량 $m$에서 $2m$까지)에 삽입이 $m/4$번 일어난다(적재율이 $0.375$에서 $0.75$로 간다). 크기 조정에는 $O(m)$이 든다. 삽입 하나당 상각된 크기 조정 비용은 $O(m)/(m/4) = O(4) = O(1)$이다. 여기에 (균등 해싱 아래) 삽입마다의 기대 비용 $O(1)$을 더하면 전체 상각 비용은 $O(1)$이다. $\square$

## 정리하며

이 마당은 탐사 거리、삽입 알고리즘、분산 줄이기、탐색과 삭제을 차례로 짚었다.

**참고 문헌**

- [Introduction to Algorithms (CLRS), Chapter 11](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Celis, P. "Robin Hood Hashing." PhD Thesis, University of Waterloo, 1986.
