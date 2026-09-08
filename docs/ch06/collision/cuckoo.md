# 뻐꾸기 해싱

대부분의 충돌 해결 전략은 최악의 경우 조회가 $O(n)$으로 떨어질 수 있음을 받아들인다. 뻐꾸기 해싱은 근본적으로 다른 길을 간다. 해시 함수 두 개를 쓰고 충돌이 일어나면 기존 키를 밀어내어 최악의 경우에도 $O(1)$ 조회와 삭제를 보장한다. 뻐꾸기가 둥지에서 다른 알을 밀어내는 모습과 비슷하다.

---

## 1. 두 테이블 방식

뻐꾸기 해싱은 크기가 각각 $m$인 테이블 $T_1$과 $T_2$을 두고 독립인 해시 함수 $h_1$과 $h_2$을 쓴다. 모든 키 $k$은 있을 수 있는 두 자리 가운데 정확히 하나에 놓인다.

$$
T_1[h_1(k)] \quad \text{or} \quad T_2[h_2(k)]
$$

키마다 후보 자리가 많아야 둘이고 자리마다 키가 많아야 하나이므로 조회와 삭제는 정확히 두 칸만 살핀다.

---

## 2. 조회와 삭제

키 $k$ **조회**: $T_1[h_1(k)]$과 $T_2[h_2(k)]$을 확인한다. 어느 한쪽에 $k$이 있으면 돌려주고, 없으면 "찾지 못함"을 알린다. 최악의 경우에도 $O(1)$이다.

키 $k$ **삭제**: 같은 두 칸 확인으로 $k$을 찾아 없앤다. 이 또한 최악의 경우에도 $O(1)$이다.

---

## 3. 삽입 알고리즘

뻐꾸기 해싱이 흥미로워지는 것은 삽입에서다. 키 $k$을 넣으려면 다음과 같이 한다.

1. $T_1[h_1(k)]$이 비어 있으면 거기에 $k$을 놓고 끝낸다.
2. 그렇지 않으면 $T_1[h_1(k)]$에 있던 $k'$을 밀어내고 그 자리에 $k$을 놓는다.
3. $k'$을 $T_2[h_2(k')]$에 놓으려 한다. 그 칸이 비어 있으면 $k'$을 놓고 끝낸다.
4. 그렇지 않으면 $T_2[h_2(k')]$의 주인을 밀어내고 밀어내기를 되풀이한다.
5. 밀어내기가 문턱값(보통 $O(\log n)$ 걸음)을 넘으면 **순환**으로 보고 **재해싱**한다. 새 해시 함수 $h_1, h_2$을 골라 모든 키를 다시 넣는다.

밀어내기의 사슬은 "차내기"의 열로 그려 볼 수 있다.

$$
k \xrightarrow{\text{evicts}} k_1 \xrightarrow{\text{evicts}} k_2 \xrightarrow{\text{evicts}} \cdots
$$

---

## 4. 순환 탐지와 재해싱

밀어내기의 사슬이 이미 밀려난 키로 되돌아오면 순환이 생긴 것이다. 무작위 해시 함수를 쓰고 적재율이 $50\%$ 아래이면 순환은 드물다. 순환이 탐지되면 새 해시 함수로 테이블 전체를 다시 세운다.

재해싱을 일으키기 전의 최대 밀어내기 사슬 길이는 보통 다음과 같이 정한다.

$$
\text{MaxKicks} = c \cdot \log n
$$

여기서 $c$은 작은 상수이며, 쓸데없는 재해싱과 지나치게 긴 삽입 사슬 사이의 균형을 맞춘다.

---

## 5. 분석

$h_1$과 $h_2$을 보편 해시 족에서 뽑았다고 가정하면 다음과 같다.

| 연산 | 시간 |
|---|---|
| 조회 | 최악의 경우 $O(1)$ |
| 삭제 | 최악의 경우 $O(1)$ |
| 삽입 | 기대 상각 $O(1)$ |

적재율이 다음을 만족할 때 삽입의 기대 상각 비용이 $O(1)$이다.

$$
\alpha = \frac{n}{2m} < \frac{1}{2}
$$

여기서 $n$은 저장된 키의 수이고 $2m$은 두 테이블을 합한 칸의 수이다. 이 문턱값을 넘으면 순환의 확률이 급격히 커진다.

기본 뻐꾸기 해싱의 공간 활용률은 많아야 $50\%$이어서 선형 탐사나 체이닝보다 효율이 낮다. 버킷마다 칸을 여럿 두는 **버킷화 뻐꾸기 해싱** 같은 확장은 적재율을 $90\%$ 넘게 올릴 수 있다.

---

## 6. 다른 전략과의 비교

| 성질 | 체이닝 | 선형 탐사 | 뻐꾸기 |
|---|---|---|---|
| 최악의 경우 조회 | $O(n)$ | $O(n)$ | $O(1)$ |
| 공간 부담 | 항목마다 포인터 | 없음 | 테이블 두 개 |
| 최대 적재율 | 제한 없음 | 실무에서 ${\sim}70\%$ | 기본형 ${\sim}50\%$ |
| 캐시 거동 | 나쁨 | 아주 좋음 | 보통 |

최악의 경우에도 $O(1)$이라는 보장 덕분에 뻐꾸기 해싱은 지연을 예측할 수 있어야 하는 실시간 시스템과 하드웨어 구현에서 매력적이다.

---

## 7. 파이썬 구현

```python
"""
표 두 개를 쓰는 뻐꾸기 해싱 구현.

찾기와 지우기의 최악 시간이 O(1)임을 보장하는
밀어내기 방식 삽입을 보인다.
"""

# === 뻐꾸기 해시 표 ===

class CuckooHashTable:
    """서로 독립인 표 두 개로 뻐꾸기 해싱을 하는 해시 표."""

    MAX_KICKS = 50  # 다시 해싱하기 전 밀어내기의 한계

    def __init__(self, capacity=8):
        self.capacity = capacity
        self.size = 0
        self.table1 = [None] * capacity
        self.table2 = [None] * capacity
        self._seed1 = 0
        self._seed2 = 1

    def _h1(self, key):
        return hash((key, self._seed1)) % self.capacity

    def _h2(self, key):
        return hash((key, self._seed2)) % self.capacity

    def lookup(self, key):
        """최악의 경우에도 O(1)인 찾기."""
        pos1 = self._h1(key)
        if self.table1[pos1] is not None and self.table1[pos1][0] == key:
            return self.table1[pos1][1]
        pos2 = self._h2(key)
        if self.table2[pos2] is not None and self.table2[pos2][0] == key:
            return self.table2[pos2][1]
        return None

    def delete(self, key):
        """최악의 경우에도 O(1)인 지우기."""
        pos1 = self._h1(key)
        if self.table1[pos1] is not None and self.table1[pos1][0] == key:
            self.table1[pos1] = None
            self.size -= 1
            return True
        pos2 = self._h2(key)
        if self.table2[pos2] is not None and self.table2[pos2][0] == key:
            self.table2[pos2] = None
            self.size -= 1
            return True
        return False

    def insert(self, key, value):
        """뻐꾸기 밀어내기로 삽입한다. 순환이 생기면 다시 해싱한다."""
        # 열쇠가 이미 있는지 확인
        if self.lookup(key) is not None:
            self.delete(key)

        item = (key, value)
        for _ in range(self.MAX_KICKS):
            # 1번 표에 시도
            pos1 = self._h1(item[0])
            if self.table1[pos1] is None:
                self.table1[pos1] = item
                self.size += 1
                return
            # 1번 표에서 내보내기
            item, self.table1[pos1] = self.table1[pos1], item

            # 2번 표에 시도
            pos2 = self._h2(item[0])
            if self.table2[pos2] is None:
                self.table2[pos2] = item
                self.size += 1
                return
            # 2번 표에서 내보내기
            item, self.table2[pos2] = self.table2[pos2], item

        # 순환 발견 — 새 씨앗으로 다시 해싱
        self._rehash(item)

    def _rehash(self, pending_item):
        """새 해시 함수로 두 표를 다시 만든다."""
        self._seed1 += 2
        self._seed2 += 2
        old_items = []
        for i in range(self.capacity):
            if self.table1[i] is not None:
                old_items.append(self.table1[i])
                self.table1[i] = None
            if self.table2[i] is not None:
                old_items.append(self.table2[i])
                self.table2[i] = None
        self.size = 0
        for k, v in old_items:
            self.insert(k, v)
        self.insert(pending_item[0], pending_item[1])

# === 시연 ===

if __name__ == "__main__":
    ct = CuckooHashTable(capacity=8)

    keys = ["alpha", "beta", "gamma", "delta", "epsilon"]
    for i, key in enumerate(keys):
        ct.insert(key, i + 1)

    for key in keys:
        print(f"lookup('{key}'): {ct.lookup(key)}")

    ct.delete("gamma")
    print(f"After delete, lookup('gamma'): {ct.lookup('gamma')}")
    print(f"Size: {ct.size}")
```

**출력:**
```
lookup('alpha'): 1
lookup('beta'): 2
lookup('gamma'): 3
lookup('delta'): 4
lookup('epsilon'): 5
After delete, lookup('gamma'): None
Size: 4
```

---

## 연습문제

**연습문제 1.**
뻐꾸기 해싱에 대해, 적재율이 $\alpha = 0.75$일 때 삽입과 조회의 기대 시간과 최악의 경우 시간을 계산하라.

??? success "연습문제 1 풀이"
    기대 시간은 충돌 해결 전략에 달렸으며 균등 해싱을 가정한다. 체이닝에서는 기대 시간이 $O(1 + \alpha) = O(1.75)$이다. 개방 주소법에서는 탐색에 실패할 때 기대 탐사 횟수가 $\approx 1/(1-\alpha) = 4$이다. 최악의 경우는 모든 키가 같은 칸으로 해시될 때의 $O(n)$이다.

---

**연습문제 2.**
뻐꾸기 해싱을(를) 써서 키 10, 22, 31, 4, 15, 28, 17을 크기가 7인 해시 테이블에 넣어라. 최종 테이블의 상태를 보여라.

??? success "연습문제 2 풀이"
    해시 함수 $h(k) = k \bmod 7$을 적용하고 이 쪽의 방법으로 충돌을 처리한다. 키마다 해시를 계산하고 충돌을 해결한 뒤 키를 놓는다. 최종 테이블의 내용을 보인다.

---

**연습문제 3.**
뻐꾸기 해싱은(는) 딥러닝의 임베딩 테이블에서 어떻게 쓰이는가? 토큰 $V = 50{,}000$개의 어휘를 $m = 30{,}000$개의 버킷에 대응시킬 때 충돌의 양상을 분석하라.

??? success "연습문제 3 풀이"
    $V/m \approx 1.67$이므로 비둘기집 원리에 의해 충돌이 반드시 생긴다. 버킷마다 평균 1.67개의 토큰이 같은 임베딩을 나누어 쓴다. 충돌률과 그것이 모델의 품질에 미치는 영향은 해시 함수의 품질과 임베딩의 차원에 달렸다(차원이 높을수록 충돌을 더 잘 견딘다).

---

**연습문제 4.**
$\alpha > 0.75$일 때 해시 테이블의 크기를 다시 잡으면 삽입의 상각 비용이 $O(1)$으로 유지됨을 증명하라.

??? success "연습문제 4 풀이"
    크기를 다시 잡는 사이(용량 $m$에서 $2m$까지)에 삽입이 $m/4$번 일어난다(적재율이 $0.375$에서 $0.75$로 간다). 크기 조정에는 $O(m)$이 든다. 삽입 하나당 상각된 크기 조정 비용은 $O(m)/(m/4) = O(4) = O(1)$이다. 여기에 (균등 해싱 아래) 삽입마다의 기대 비용 $O(1)$을 더하면 전체 상각 비용은 $O(1)$이다. $\square$

## 정리하며

이 마당은 두 테이블 방식、조회와 삭제、삽입 알고리즘、순환 탐지와 재해싱을 차례로 짚었다.

**참고 문헌**

- [Introduction to Algorithms (CLRS), Chapter 11](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Pagh, R. and Rodler, F. F. "Cuckoo Hashing." *Journal of Algorithms*, 51(2), 2004.
