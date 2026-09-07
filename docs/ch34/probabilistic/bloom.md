# 블룸 필터

해시 집합은 소속 물음에 딱 맞게 답하지만 갈무리한 원소 개수에 견주는 자리를 쓴다. 온 모임이 크고 **거짓 양성**의 작은 낌새를 받아들일 수 있다면 **블룸 필터**가 훨씬 적은 기억으로 같은 겉면을 준다. 거짓 음성이 없음을 보장한다. 곧 넣은 원소라면 필터가 늘 있다고 알린다.

## 얼개

블룸 필터는 다음으로 이루어진다.

- 처음에 모두 0인 비트 $m$개의 **비트 배열** $B$.
- 원소를 $\{0, 1, \ldots, m-1\}$으로 맞대는, 서로 매이지 않은 해시 함수 $k$개의 갈래 $h_1, h_2, \ldots, h_k$.

## 연산

**넣기($x$)**: 모든 $i \in \{1, \ldots, k\}$에 대해 $B[h_i(x)] = 1$으로 둔다.

**묻기($x$)**: 모든 $i$에 대해 $B[h_i(x)] = 1$이면 `True`을, 아니면 `False`을 돌려준다.

- $x$을 넣었다면 $k$개 자리가 모두 켜졌으므로 물음이 늘 `True`을 돌려준다(거짓 음성이 없다).
- $x$을 넣은 적이 없어도 다른 원소가 그 자리를 켜 놓았을 수 있어 거짓 양성이 생긴다.

**지우기**: 여느 블룸 필터는 받쳐 주지 않는다. 비트를 끄면 다른 원소의 자취까지 지울 수 있기 때문이다.

## 거짓 양성 낌새

비트가 $m$개이고 해시 함수가 $k$개인 필터에 원소 $n$개를 넣은 뒤, 비트 하나가 0으로 남을 낌새는 다음과 같다.

$$
\Pr[\text{비트 } j = 0] = \left(1 - \frac{1}{m}\right)^{kn} \approx e^{-kn/m}
$$

거짓 양성은 속하지 않은 것의 해시 자리 $k$개가 마침 모두 켜져 있을 때 생긴다.

$$
P_{\text{거짓양성}} \approx \left(1 - e^{-kn/m}\right)^k
$$

## 가장 좋은 해시 함수 개수

($m$과 $n$을 붙박인 것으로 여기고) $k$에 대해 $P_{\text{거짓양성}}$을 가장 작게 하면 다음을 얻는다.

$$
k^* = \frac{m}{n} \ln 2
$$

이 가장 좋은 자리에서 거짓 양성률은 다음으로 간추려진다.

$$
P_{\text{거짓양성}}^* = \left(\frac{1}{2}\right)^k = 2^{-k}
$$

겨눈 거짓 양성률 $\epsilon$에 대해 원소마다 있어야 하는 비트 수는 다음과 같다.

$$
\frac{m}{n} = -\frac{\ln \epsilon}{(\ln 2)^2} \approx 1.44 \log_2 \frac{1}{\epsilon}
$$

!!! example "실제 크기 잡기"
    $\epsilon = 1\%$(100에 1번 거짓 양성)이면 원소마다 약 $m/n \approx 9.6$비트와 해시 함수 $k = 7$개가 있어야 한다. 이는 놀랍도록 자리를 아낀다. 원소 $10^6$개를 갈무리하는 데 원소 크기와 상관없이 약 1.2 MB면 된다.

## 복잡도

| 연산 | 때 | 자리 |
|---|---|---|
| 넣기 | $O(k)$ | -- |
| 묻기 | $O(k)$ | -- |
| 온 얼개 | -- | 비트 $O(m)$개 |

$k$은 흔히 작은 상수(3~10)이므로 모든 연산이 $O(1)$ 때에 돈다.

## 구현

```python
"""
블룸 필터 -- 자리를 아끼는 확률 소속 시험.

m비트 배열 위에 서로 매이지 않은 해시 함수 k개를 쓴다. 거짓
음성이 없음을 보장하고 거짓 양성률은 대략 (1-e^{-kn/m})^k이다.
"""

import hashlib
import math


# === 블룸 필터 ================================================================

class BloomFilter:
    """더하기와 소속 묻기를 받쳐 주는 확률 집합."""

    def __init__(self, expected_items: int, fp_rate: float = 0.01):
        """주어진 거짓 양성률로 *expected_items*에 맞춰 필터 크기를 잡는다."""
        self.n_expected = expected_items
        self.fp_rate = fp_rate
        # 가장 좋은 비트 수
        self.m = max(1, int(-expected_items * math.log(fp_rate) / (math.log(2) ** 2)))
        # 가장 좋은 해시 함수 개수
        self.k = max(1, int((self.m / expected_items) * math.log(2)))
        self.bits = [False] * self.m
        self.count = 0

    def _hashes(self, item: str) -> list[int]:
        """*item*의 해시 자리 k개를 셈한다."""
        positions = []
        for i in range(self.k):
            digest = hashlib.md5(f"{item}:{i}".encode()).hexdigest()
            positions.append(int(digest, 16) % self.m)
        return positions

    def add(self, item: str) -> None:
        """*item*을 필터에 넣는다."""
        for pos in self._hashes(item):
            self.bits[pos] = True
        self.count += 1

    def query(self, item: str) -> bool:
        """*item*이 집합에 있을 수 있는지 시험한다."""
        return all(self.bits[pos] for pos in self._hashes(item))


# === 메인 =====================================================================

if __name__ == "__main__":
    bf = BloomFilter(expected_items=100, fp_rate=0.01)
    print(f"Filter size: {bf.m} bits, {bf.k} hash functions")

    # 항목 몇 개를 넣는다
    inserted = ["apple", "banana", "cherry", "date", "elderberry"]
    for word in inserted:
        bf.add(word)

    # 넣은 항목을 묻는다(모두 True여야 한다)
    print("\nInserted items:")
    for word in inserted:
        print(f"  {word}: {bf.query(word)}")

    # 넣지 않은 항목을 묻는다(거의 모두 False여야 한다)
    print("\nNon-inserted items:")
    for word in ["fig", "grape", "honeydew", "kiwi"]:
        print(f"  {word}: {bf.query(word)}")
```

**출력:**

```
Filter size: 958 bits, 6 hash functions

Inserted items:
  apple: True
  banana: True
  cherry: True
  date: True
  elderberry: True

Non-inserted items:
  fig: False
  grape: False
  honeydew: False
  kiwi: False
```

넣은 항목은 모두 (보장대로) `True`을 돌려주고, 이 작은 보기에서는 넣지 않은 항목이 `False`을 돌려준다. 원소가 늘어 필터 용량에 다가가면 정해 둔 1% 비율로 거짓 양성이 나타난다.

## 참고 문헌

- Bloom, B.H. "Space/Time Trade-offs in Hash Coding with Allowable Errors." *CACM*, 1970
- Mitzenmacher, M. and Upfal, E. *Probability and Computing*. Cambridge University Press, 2005

## 연습문제

**연습문제 1.**
어떤 블룸 필터가 원소 $n$개를 갈무리하려고 비트 $m$개와 해시 함수 $k$개를 쓴다. 거짓 양성 낌새를 $m$, $k$, $n$의 함수로 이끌어 내어라.

??? success "연습문제 1 풀이"
    해시 함수 $k$개로 원소 $n$개를 넣으면 해시 결과 $kn$개가 저마다 $m$개 비트 가운데 하나를 1로 켠다. 모두 넣은 뒤 어떤 비트 하나가 0으로 남을 낌새는 $(1 - 1/m)^{kn} \approx e^{-kn/m}$이다. 거짓 양성은 속하지 않은 것에 대해 살핀 비트 $k$개가 모두 1일 때 생긴다. 비트마다 서로 매이지 않고 낌새 $1 - e^{-kn/m}$으로 1이므로 거짓 양성 낌새는 다음과 같다.

    $$
    p = \left(1 - e^{-kn/m}\right)^k
    $$

    $k$에 대해 가장 작게 하면 가장 좋은 $k = (m/n) \ln 2$을 얻고 $p = (1/2)^k = (0.6185)^{m/n}$이 된다. $\square$

---

**연습문제 2.**
겨눈 거짓 양성률 $1\%$으로 원소 $n = 10^6$개를 갈무리하는 블룸 필터에 대해 가장 좋은 해시 함수 개수 $k$과 있어야 하는 비트 수 $m$을 셈하여라.

??? success "연습문제 2 풀이"
    가장 좋은 비트 수는 $p = 0.01$일 때 $m = -(n \ln p) / (\ln 2)^2$이다. 넣어 보면 $m = -(10^6 \times \ln 0.01) / (0.693)^2 = -(10^6 \times (-4.605)) / 0.480 = 9.59 \times 10^6 \approx 9.6 \times 10^6$ 비트, 곧 약 1.2 MB이다. 가장 좋은 해시 함수 개수는 $k = (m/n) \ln 2 = 9.6 \times \ln 2 \approx 6.6$이므로 $k = 7$으로 어림한다. $k = 7$이고 $m = 9.6 \times 10^6$이면 참 거짓 양성률은 $(1 - e^{-7 \times 10^6 / 9.6 \times 10^6})^7 \approx (0.518)^7 \approx 0.008 = 0.8\%$으로 겨눈 $1\%$보다 살짝 낮다. $\square$

---

**연습문제 3.**
블룸 필터에 거짓 음성이 있을 수 없음을(넣은 원소라면 필터가 늘 있다고 알림을) 증명하여라.

??? success "연습문제 3 풀이"
    원소 $x$을 넣으면 필터가 자리 $h_1(x), h_2(x), \ldots, h_k(x)$의 비트를 1로 켠다. 한 번 1로 켜진 비트는 결코 꺼지지 않는다(여느 블룸 필터에는 지우기 연산이 없다). $x$을 물으면 필터가 같은 자리 $h_1(x), \ldots, h_k(x)$을 살핀다. 이 비트들은 넣을 때 1로 켜졌고 결코 꺼지지 않으므로 물을 때에도 $k$개가 모두 1이다. 필터는 $k$개 비트가 모두 1일 때 "있음"이라 알리므로 넣은 원소에 대해 늘 "있음"이라 알린다. 따라서 거짓 음성은 있을 수 없다. $\square$

---

**연습문제 4.**
어떤 그물 기어 다니개가 이미 들른 주소를 다시 찾지 않으려고 블룸 필터를 쓴다. 거짓 양성률 $0.1\%$으로 주소 $10^8$개를 기어 다닌 뒤, 제대로 된 주소가 몇 개나 그릇되게 건너뛰어지는가? 실제로 미치는 바를 따져라.

??? success "연습문제 4 풀이"
    거짓 양성률이 $0.1\%$이면 필터에 없는 주소를 물을 때마다 $0.001$의 낌새로 거짓 양성이(들렀다고 그릇되게 알림이) 생긴다. 기어 다니개가 온통 서로 다른 주소 $10^9$개를 만난다면 $10^9 - 10^8 = 9 \times 10^8$개가 필터에 없다. 거짓 양성의 어림 개수는 $9 \times 10^8 \times 0.001 = 9 \times 10^5 = 900{,}000$개이므로 주소 90만 개가 그릇되게 건너뛰어진다. 실제로 미치는 바: 기어 다니개가 들러야 할 그물 쪽의 대략 0.1%를 놓친다. 찾기 엔진으로서는 받아들일 만하다. 거짓 양성이 아무렇게나 생기므로 놓친 쪽이 한쪽으로 치우칠 낌새가 낮다. 블룸 필터는 기억을 엄청나게 아낀다. 주소 $10^8$개를 글자열로 갈무리하면 약 10 GB가 들지만 블룸 필터는 약 120 MB를 쓴다. $\square$

---

**연습문제 5.**
소속 시험 문제에서 블룸 필터와 해시 집합을 견주어라. 어느 조건에서 어느 쪽이 나은가?

??? success "연습문제 5 풀이"
    **블룸 필터**: 자리가 비트 $O(n)$개이고 상수 인자는 겨눈 거짓 양성률에 달렸다(거짓 양성률 1%에 대략 원소마다 10비트). 거짓 음성이 없으나 거짓 양성이 있다. 원소 자체를 갈무리하지 않는다. **해시 집합**: 자리가 $O(n)$이고 상수 인자는 원소 크기에 달렸다(보기로 항목마다 손가락질 8바이트에 원소 크기를 더한 값). 딱 맞으며 거짓 양성도 거짓 음성도 없다. 원소를 갈무리하고 죽 늘어놓기를 받쳐 준다. 블룸 필터가 나은 때는 다음이다. (1) 기억이 종요롭고 원소가 클 때(주소, 파일 해시). (2) 거짓 양성을 받아들일 수 있을 때(캐시 살피기, 값비싼 찾기 앞의 미리 거르기). (3) 지우기가 필요 없을 때. 해시 집합이 나은 때는 다음이다. (1) 딱 맞는 소속이 있어야 할 때. (2) 원소를 되찾거나 죽 늘어놓아야 할 때. (3) 지우기를 받쳐 주어야 할 때. (4) (0.01%일지라도) 블룸 필터의 거짓 양성률을 받아들일 수 없을 때(보기로 안전이 종요로운 겹침 없애기). $\square$
