# 뻐꾸기 필터

블룸 필터는 자리를 아끼지만 (세는 갈래를 쓰지 않고서는) 지우기를 받쳐 주지 못하고 갈무리한 항목을 알려 주지도 못한다. **뻐꾸기 필터**는 뻐꾸기 해시 표에 야무진 손도장을 갈무리하여 이 두 매임을 함께 다룬다. 넣기, 지우기, 소속 묻기를 받쳐 주면서 세는 블룸 필터보다 자리를 잘 아끼고 거짓 양성률은 견줄 만하다.

## 얼개

뻐꾸기 필터는 다음으로 이루어진다.

- 두레박 $m$개의 해시 표. 두레박마다 항목을 $b$개까지 담는다(흔히 $b = 4$이다).
- 항목마다 (온전한 원소 $x$이 아니라) $f$비트짜리 **손도장** $f(x)$을 갈무리한다.
- 원소마다 후보 두레박 둘을 정하는 해시 함수 둘.

종요로운 눈길은 **조각 열쇠 뻐꾸기 해싱**이다. 두레박 번호 하나와 손도장이 주어지면 처음 원소를 몰라도 다른 쪽 두레박을 셈할 수 있다.

$$
h_1(x) = \text{hash}(x)
$$

$$
h_2(x) = h_1(x) \oplus \text{hash}(f(x))
$$

이 XOR 얽힘은 $h_1(x) = h_2(x) \oplus \text{hash}(f(x))$을 뜻하므로, 어느 두레박에서든 갈무리한 손도장만으로 다른 쪽 두레박을 찾을 수 있다.

## 연산

**넣기($x$)**: $f = f(x)$, $i_1 = h_1(x)$, $i_2 = h_2(x)$을 셈한다.

1. 두레박 $i_1$이나 $i_2$에 빈 칸이 있으면 거기에 $f$을 갈무리한다.
2. 아니면 두레박 하나를 골라(이를테면 $i_1$) 아무 항목 $f'$을 내쫓고 그 자리에 $f$을 갈무리한다.
3. $f'$을 그 다른 쪽 두레박으로 옮긴다. 그 두레박도 꽉 찼으면 내쫓기를 되풀이한다(발길질의 최대 횟수까지).
4. 최대 발길질을 넘기면 표가 꽉 찼다고 알리고 크기 바꾸기를 부른다.

**묻기($x$)**: $f = f(x)$, $i_1 = h_1(x)$, $i_2 = h_2(x)$을 셈한다. 두레박 $i_1$이나 $i_2$에 $f$이 있으면 `True`을 돌려준다.

**지우기($x$)**: $f$, $i_1$, $i_2$을 셈한다. 두레박 $i_1$이나 $i_2$에서 $f$을 찾으면 하나를 없앤다.

## 거짓 양성 살피기

거짓 양성은 속하지 않은 것의 손도장이 그 후보 두레박 둘 가운데 하나에 갈무리된 손도장과 들어맞을 때 생긴다. 손도장이 $f$비트이고 두레박 크기가 $b$이면 다음과 같다.

$$
P_{\text{거짓양성}} \le \frac{2b}{2^f}
$$

$b = 4$이고 $f = 12$비트이면 $P_{\text{거짓양성}} \le 8/4096 \approx 0.2\%$이다.

## 자리 아낌

원소마다 $f$비트의 곳간이 있어야 한다. 채움률 $\alpha$에서 원소 $n$개의 온 자리는 다음과 같다.

$$
\text{원소마다 비트} = \frac{f}{\alpha}
$$

$b = 4$인 두레박과 반쯤 매긴 두레박(엮기 다듬기)을 쓰면 뻐꾸기 필터는 대략 $\alpha = 0.95$을 이루고, 거짓 양성률이 $3\%$ 아래일 때 블룸 필터보다 원소마다 적은 비트를 쓴다.

!!! tip "뻐꾸기 필터를 고를 때"
    지우기 받침이 필요하거나, 낮은 거짓 양성률(3% 아래)에서 자리를 더 아끼고 싶거나, 더 빠른 찾기(블룸 필터의 $k$번이 아니라 기억에 두 번만 닿음)를 바랄 때 블룸 필터 대신 뻐꾸기 필터를 쓰라.

## 구현

```python
"""
뻐꾸기 필터 -- 지우기를 받쳐 주며 자리를 아끼는 확률 집합.

두레박 옮기기에 조각 열쇠 뻐꾸기 해싱을 쓰는 뻐꾸기 해시 표에
손도장을 갈무리한다.
"""

import hashlib
import random


# === 뻐꾸기 필터 ==============================================================

class CuckooFilter:
    """넣기, 지우기, 묻기를 받쳐 주는 확률 집합."""

    MAX_KICKS = 500

    def __init__(self, capacity: int, bucket_size: int = 4,
                 fingerprint_bits: int = 8):
        self.bucket_size = bucket_size
        self.fp_bits = fingerprint_bits
        self.fp_mask = (1 << fingerprint_bits) - 1
        self.num_buckets = max(1, capacity // bucket_size)
        self.buckets: list[list[int]] = [[] for _ in range(self.num_buckets)]
        self.count = 0

    def _fingerprint(self, item: str) -> int:
        """0이 아닌 손도장을 셈한다."""
        h = int(hashlib.sha256(item.encode()).hexdigest(), 16)
        fp = (h & self.fp_mask) or 1  # 0이 아니게 한다
        return fp

    def _hash(self, item: str) -> int:
        """으뜸 두레박 번호."""
        h = int(hashlib.md5(item.encode()).hexdigest(), 16)
        return h % self.num_buckets

    def _alt_index(self, index: int, fingerprint: int) -> int:
        """해시한 손도장과 XOR 하여 얻는 다른 쪽 두레박."""
        fp_hash = hash(fingerprint) % self.num_buckets
        return (index ^ fp_hash) % self.num_buckets

    def insert(self, item: str) -> bool:
        """*item*을 넣는다. 표가 꽉 찼으면 False을 돌려준다."""
        fp = self._fingerprint(item)
        i1 = self._hash(item)
        i2 = self._alt_index(i1, fp)

        if len(self.buckets[i1]) < self.bucket_size:
            self.buckets[i1].append(fp)
            self.count += 1
            return True
        if len(self.buckets[i2]) < self.bucket_size:
            self.buckets[i2].append(fp)
            self.count += 1
            return True

        # 내쫓기 되돌이
        idx = random.choice([i1, i2])
        for _ in range(self.MAX_KICKS):
            evict_pos = random.randrange(len(self.buckets[idx]))
            fp, self.buckets[idx][evict_pos] = (
                self.buckets[idx][evict_pos], fp
            )
            idx = self._alt_index(idx, fp)
            if len(self.buckets[idx]) < self.bucket_size:
                self.buckets[idx].append(fp)
                self.count += 1
                return True

        return False  # 표가 꽉 찼다

    def query(self, item: str) -> bool:
        """*item*이 집합에 있을 수 있는지 시험한다."""
        fp = self._fingerprint(item)
        i1 = self._hash(item)
        i2 = self._alt_index(i1, fp)
        return fp in self.buckets[i1] or fp in self.buckets[i2]

    def delete(self, item: str) -> bool:
        """*item*을 없앤다. 찾지 못하면 False을 돌려준다."""
        fp = self._fingerprint(item)
        i1 = self._hash(item)
        i2 = self._alt_index(i1, fp)
        if fp in self.buckets[i1]:
            self.buckets[i1].remove(fp)
            self.count -= 1
            return True
        if fp in self.buckets[i2]:
            self.buckets[i2].remove(fp)
            self.count -= 1
            return True
        return False


# === 메인 =====================================================================

if __name__ == "__main__":
    cf = CuckooFilter(capacity=100, fingerprint_bits=12)

    for word in ["apple", "banana", "cherry"]:
        cf.insert(word)

    print("After inserting apple, banana, cherry:")
    for word in ["apple", "banana", "cherry", "date"]:
        print(f"  {word}: {cf.query(word)}")

    cf.delete("banana")
    print("\nAfter deleting banana:")
    for word in ["apple", "banana", "cherry", "date"]:
        print(f"  {word}: {cf.query(word)}")
```

**출력:**

```
After inserting apple, banana, cherry:
  apple: True
  banana: True
  cherry: True
  date: False

After deleting banana:
  apple: True
  banana: False
  cherry: True
  date: False
```

지우기가 옳게 돈다. `banana`을 없애도 `apple`과 `cherry`은 흔들리지 않으며, 이는 비트 배열에 견준 손도장 곳간의 이로움을 보여 준다.

## 참고 문헌

- Fan, B., Andersen, D.G., Kaminsky, M., and Mitzenmacher, M. "Cuckoo Filter: Practically Better Than Bloom." *CoNEXT*, 2014
- Pagh, R. and Rodler, F.F. "Cuckoo Hashing." *ESA*, 2001

## 연습문제

**연습문제 1.**
뻐꾸기 필터의 얼개를 밝혀라. 원소는 어떻게 갈무리되고 소속 물음은 어떻게 도는가?

??? success "연습문제 1 풀이"
    뻐꾸기 필터는 두레박 $m$개(두레박마다 칸 $b$개, 흔히 $b = 4$)의 뻐꾸기 해시 표에 원소의 **손도장**(야무진 해시)을 갈무리한다. 원소 $x$에 대해 손도장 $f = \text{fingerprint}(x)$과 후보 두레박 번호 둘 $i_1 = h(x)$, $i_2 = i_1 \oplus h(f)$(XOR에 바탕을 둔 다른 쪽 자리)을 셈한다. $i_1$이나 $i_2$의 빈 칸에 $f$을 넣는다. 둘 다 꽉 찼으면 있던 손도장 하나를 아무렇게나 내쫓아 그 다른 쪽 두레박으로 옮기고, 발길질의 최대 횟수까지 되풀이한다. 소속 물음은 $f$을 셈하고 $i_1$과 $i_2$을 모두 살핀다. 어느 두레박에서든 $f$을 찾으면 "있음"(거짓 양성일 수 있다)을, 아니면 "없음"을 돌려준다. 거짓 양성은 다른 원소의 손도장이 후보 두레박 둘 가운데 하나에서 $f$과 부딪칠 때 생긴다. $\square$

---

**연습문제 2.**
처음 열쇠를 갈무리하지 않고 손도장만으로 다른 쪽 두레박 자리를 셈하게 하는 "조각 열쇠 뻐꾸기 해싱" 재주를 풀어라.

??? success "연습문제 2 풀이"
    여느 뻐꾸기 해싱에서는 두레박 자리 둘을 온전한 열쇠로 셈한다. 곧 $i_1 = h_1(x)$, $i_2 = h_2(x)$이다. 내쫓는 동안 손도장을 옮기려면 (다른 쪽 자리를 셈하려고) 처음 열쇠를 알아야 하는데, 필터는 그것을 갈무리하지 않는다. 조각 열쇠 뻐꾸기 해싱은 $i_2 = i_1 \oplus h(f)$으로 정의하여 이를 푼다. 여기서 $f$은 갈무리한 손도장이다. 이 얽힘은 맞바꿔도 되어 $i_1 = i_2 \oplus h(f)$이다. 지금 두레박 $i$에 있는 손도장 $f$의 다른 쪽 두레박은 $i \oplus h(f)$이며, 처음 열쇠 없이 $i$과 $f$만으로 셈할 수 있다. 이 재주가 필터에서 내쫓기를 이루게 한다. 매임은 $h(f)$이 내쫓긴 항목을 표에 고르게 흩뿌리도록 손도장이 넉넉히 아무래야 한다는 것이다. $\square$

---

**연습문제 3.**
8비트 손도장과 두레박 크기 4을 쓰는 뻐꾸기 필터에 원소 $n = 10^6$개를 담아 용량의 95%까지 채웠다. 거짓 양성률과 온 기억 쓰임을 셈하여라.

??? success "연습문제 3 풀이"
    8비트 손도장이면 두레박의 어떤 자리에서 두 원소가 부딪칠 낌새가 $1/2^8 = 1/256$이다. 물음은 크기 4인 두레박 둘을 살피므로 손도장을 $2b = 8$개까지 들여다본다. 손도장 크기를 $f = 8$이라 할 때 거짓 양성률은 대략 $2b / 2^f = 8 / 256 = 3.125\%$이다. 채움률 95%에서 두레박 개수는 $n / (b \times 0.95) = 10^6 / (4 \times 0.95) \approx 263{,}158$이다. 온 기억은 $263{,}158 \times 4 \times 8$비트 $= 8{,}421{,}056$비트로 약 1.03 MB이며 원소마다 약 8.4비트다. 견주어 보면, 거짓 양성률 3.125%의 여느 블룸 필터는 $-(n \ln 0.03125) / (\ln 2)^2 \approx 7.2 \times 10^6$비트 $= 0.88$ MB(원소마다 7.2비트)를 쓴다. 뻐꾸기 필터가 자리를 조금 더 쓰지만 지우기를 받쳐 준다. $\square$

---

**연습문제 4.**
뻐꾸기 필터의 지우기 연산을 밝혀라. 어떤 자리에서 지우기가 거짓 음성을 낳을 수 있는가?

??? success "연습문제 4 풀이"
    지우기: $f = \text{fingerprint}(x)$을 셈하고 두레박 $i_1$과 $i_2$을 살핀다. 어느 두레박에서든 $f$을 찾으면 하나를 없앤다. 찾지 못하면 그 원소는 없었던 것이다(또는 이미 지워졌다). 거짓 음성은 서로 다른 원소 $x$과 $y$이 같은 손도장 $f$을 지니고 두레박을 함께 쓸 때 생길 수 있다. $x$을 먼저 넣고 $y$을 나중에 넣었으며(둘 다 겹치는 두레박 $i$에 손도장 $f$을 둔다) 그 뒤 $x$을 지우면, 두레박 $i$에서 $f$ 하나가 없어진다. 마침 $y$이 기대던 것을 없애면(손도장과 두레박이 똑같으므로) 뒤이어 $y$을 물을 때 $f$을 찾지 못해 "없음"을 돌려주는 거짓 음성이 된다. 이는 두 원소의 손도장이 똑같고 후보 두레박이 겹칠 때에만 생긴다. 그 낌새는 ($1/2^f$에 견주어) 매우 낮지만 0은 아니다. $\square$

---

**연습문제 5.**
뻐꾸기 필터, 블룸 필터, 세는 블룸 필터를 원소마다 자리, 지우기 받침, 찾기 빠르기, 가장 나쁠 때 넣기 때라는 네 결에서 견주어라.

??? success "연습문제 5 풀이"
    | 결 | 블룸 | 세는 블룸 | 뻐꾸기 |
    |---|---|---|---|
    | 자리(거짓 양성률 1%) | 원소마다 9.6비트 | 원소마다 38.4비트 | 원소마다 12.6비트 |
    | 지우기 | 아니오 | 예 | 예 |
    | 찾기 | 아무 자리 $k$번 읽기 | 아무 자리 $k$번 읽기 | 잇단 자리 2번 읽기 |
    | 넣기(가장 나쁠 때) | $O(k)$ | $O(k)$ | $O(1/\epsilon)$ 나눠 갚음 |

    블룸 필터는 자리를 가장 잘 아끼고 넣기 비용을 어림할 수 있으나 지우기가 없다. 세는 블룸 필터는 4배 자리를 값으로 치르고 지우기를 받쳐 준다. 뻐꾸기 필터는 가장 좋은 어우름을 준다. 자리가 어중간하고 지우기가 본디부터 있으며 찾기가 캐시에 친화적이다(흩어진 자리 $k$개가 아니라 두레박 둘이다). 다만 채움률이 높으면 뻐꾸기 필터의 넣기가 어그러져 표 크기를 바꾸어야 하지만 블룸 필터는 넣기가 결코 어그러지지 않는다. 짐이 높고 적기가 많은 일감에서는 블룸 필터가 더 어림할 만하다. $\square$
