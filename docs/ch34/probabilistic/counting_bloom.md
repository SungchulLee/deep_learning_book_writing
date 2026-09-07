# 세는 블룸 필터

여느 블룸 필터는 넣기와 소속 묻기만 받쳐 준다. 비트를 끄면 다른 원소의 자취가 지워질 수 있으므로 지우기가 될 수 없다. **세는 블룸 필터**는 비트마다 정수 세개로 갈음하여, 자리를 아끼는 확률 소속 시험을 지키면서 지우기를 이루게 한다.

## 밑뜻

길잡이 판단을 갈무리하려고 블룸 필터를 쓰는 그물 길잡이를 여겨 보자. 길이 거두어지면 길잡이가 그 항목을 없애야 한다. 여느 블룸 필터로는 이를 안전하게 할 수 없지만, 세는 블룸 필터는 비트를 끄는 대신 세개를 내릴 수 있어 원소가 오가는 움직이는 모임을 받쳐 준다.

## 얼개

세는 블룸 필터는 다음으로 이루어진다.

- 처음에 모두 0인 세개 $m$개의 배열 $C$.
- 원소를 $\{0, 1, \ldots, m-1\}$으로 맞대는 해시 함수 $k$개의 갈래 $h_1, \ldots, h_k$.

## 연산

**넣기($x$)**: 모든 $i \in \{1, \ldots, k\}$에 대해 $C[h_i(x)]$을 올린다.

**지우기($x$)**: 모든 $i \in \{1, \ldots, k\}$에 대해 $C[h_i(x)]$을 내린다. 넣은 것이 틀림없는 원소만 지워야 한다. 속하지 않은 것을 지우면 **거짓 음성**이 들어온다.

**묻기($x$)**: 모든 $i$에 대해 $C[h_i(x)] > 0$이면 `True`을, 아니면 `False`을 돌려준다.

## 거짓 양성 살피기

거짓 양성 낌새는 여느 블룸 필터와 같다.

$$
P_{\text{거짓양성}} \approx \left(1 - e^{-kn/m}\right)^k
$$

여기서 $n$은 지금 들어 있는 원소의 개수다. 지우기가 $n$을 줄이므로 거짓 양성률이 낮아지는데, 이는 반가운 성질이다.

## 세개 넘침

세개마다 넘치지 않을 만큼 넉넉해야 한다. 해시 함수가 $k$개이고 원소가 $n$개이면 아무 자리의 어림 셈이 $kn/m$이다. 세개가 값 $c$에 이를 낌새는 푸아송 꼬리를 따른다.

$$
\Pr[C_j \ge c] \le \frac{(kn/m)^c}{c!}
$$

실제로는 4비트 세개(최댓값 15)면 거의 모든 쓰임새에 넉넉하고, $m/n$을 알맞게 잡으면 넘침이 매우 드물다.

!!! warning "세개가 밑으로 넘침"
    세개를 0 아래로 내려서는 안 된다. 넣지 않은 원소를 지우면 세개가 아래로 내려가 거짓 음성이 들어올 수 있다. 만들 때 이를 막아야 한다.

## 자리 견주기

| 얼개 | 칸마다 자리 | 지우기를 받쳐 주는가 |
|---|---|---|
| 여느 블룸 필터 | 1비트 | 아니오 |
| 세는 블룸 필터(4비트) | 4비트 | 예 |
| 세는 블룸 필터(8비트) | 8비트 | 예 |

4비트 세는 블룸 필터는 여느 블룸 필터의 4배 자리를 쓰지만, 그래도 해시 표보다는 놀랍도록 적다.

## 구현

```python
"""
세는 블룸 필터 -- 지우기를 받쳐 주는 확률 집합.

여느 블룸 필터의 비트 배열을 정수 세개로 갈음하여, 다른
원소를 건드리지 않고 원소를 없앨 수 있게 한다.
"""

import hashlib
import math


# === 세는 블룸 필터 ===========================================================

class CountingBloomFilter:
    """넣기, 지우기, 묻기를 받쳐 주는 세개를 둔 블룸 필터."""

    def __init__(self, expected_items: int, fp_rate: float = 0.01,
                 counter_bits: int = 4):
        self.n_expected = expected_items
        self.fp_rate = fp_rate
        self.max_count = (1 << counter_bits) - 1
        # 가장 좋은 크기 잡기
        self.m = max(1, int(-expected_items * math.log(fp_rate)
                            / (math.log(2) ** 2)))
        self.k = max(1, int((self.m / expected_items) * math.log(2)))
        self.counters = [0] * self.m

    def _hashes(self, item: str) -> list[int]:
        """*item*의 해시 자리 k개를 셈한다."""
        positions = []
        for i in range(self.k):
            digest = hashlib.md5(f"{item}:{i}".encode()).hexdigest()
            positions.append(int(digest, 16) % self.m)
        return positions

    def add(self, item: str) -> None:
        """세개를 올려 *item*을 넣는다."""
        for pos in self._hashes(item):
            if self.counters[pos] < self.max_count:
                self.counters[pos] += 1

    def remove(self, item: str) -> None:
        """세개를 내려 *item*을 없앤다(넣었던 것이어야 한다)."""
        for pos in self._hashes(item):
            if self.counters[pos] > 0:
                self.counters[pos] -= 1

    def query(self, item: str) -> bool:
        """*item*이 집합에 있을 수 있는지 시험한다."""
        return all(self.counters[pos] > 0 for pos in self._hashes(item))


# === 메인 =====================================================================

if __name__ == "__main__":
    cbf = CountingBloomFilter(expected_items=100, fp_rate=0.01)
    print(f"Filter: {cbf.m} counters, {cbf.k} hash functions")

    # 항목을 넣는다
    for word in ["apple", "banana", "cherry"]:
        cbf.add(word)

    print("\nAfter inserting apple, banana, cherry:")
    for word in ["apple", "banana", "cherry", "date"]:
        print(f"  {word}: {cbf.query(word)}")

    # banana를 지운다
    cbf.remove("banana")
    print("\nAfter deleting banana:")
    for word in ["apple", "banana", "cherry", "date"]:
        print(f"  {word}: {cbf.query(word)}")
```

**출력:**

```
Filter: 958 counters, 6 hash functions

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

지운 뒤 `banana`은 옳게 `False`을 돌려주고 `apple`과 `cherry`은 흔들리지 않는다. 이것이 여느 블룸 필터에 견준 종요로운 이로움이다. 다른 원소를 흐트러뜨리지 않고 원소를 없앨 수 있다.

## 참고 문헌

- Fan, L., Cao, P., Almeida, J., and Broder, A.Z. "Summary Cache: A Scalable Wide-Area Web Cache Sharing Protocol." *IEEE/ACM Trans. Networking*, 2000
- Mitzenmacher, M. and Upfal, E. *Probability and Computing*. Cambridge University Press, 2005

## 연습문제

**연습문제 1.**
여느 블룸 필터에서 지우기가 될 수 없는 까닭과 세는 블룸 필터가 이를 푸는 길을 풀어라. 자리 덧듦은 얼마인가?

??? success "연습문제 1 풀이"
    여느 블룸 필터에서 비트 자리마다 여러 원소가 켜 놓았을 수 있다. 지우면서 비트를 끄면 같은 자리로 해시되는 다른 원소의 자취까지 지워져 (블룸 필터의 보장을 깨는) 거짓 음성이 생길 수 있다. 세는 블룸 필터는 비트마다 정수 세개로 갈음한다. 넣기는 $k$개 세개를 모두 올리고 지우기는 $k$개 세개를 모두 내린다. 소속 물음은 $k$개 세개가 모두 $> 0$인지 살핀다. (엉뚱한 지우기가 없다고 여기면) 세개가 결코 아래로 내려가지 않으므로, 지우지 않은 원소는 제 모든 자리에서 0보다 큰 셈을 지닌다. 자리 덧듦: 세개마다 1비트가 아니라 $b$비트가 든다. 4비트 세개($b = 4$, 최대 셈 15)이면 자리가 $m$비트가 아니라 $4m$비트이므로 4배로 는다. 흔한 짐에서 세개가 15를 넘을 낌새가 셈에 넣지 않아도 될 만큼 작아 4비트가 여느 고름이다. $\square$

---

**연습문제 2.**
원소 $n$개, 세개 $m$개, 해시 함수 $k$개, 세개 너비 $b$비트가 주어질 때 세는 블룸 필터의 세개가 (최댓값을 넘어) 넘칠 낌새를 이끌어 내어라.

??? success "연습문제 2 풀이"
    넣을 때마다 $m$개 세개 가운데 $k$개를 올린다. 어떤 세개 하나가 올라가는 횟수는 이항 분포를 따른다. 곧 $X \sim \text{Binomial}(nk, 1/m)$이며, $nk$개의 해시 결과가 저마다 서로 매이지 않고 낌새 $1/m$으로 이 세개에 닿기 때문이다. 어림 셈은 $\mu = nk/m$이다. $X > 2^b - 1$이면 세개가 넘친다. $b = 4$이면 $X > 15$일 때 넘친다. $k = 7$이고 $m/n = 10$(거짓 양성률 1%에 가장 좋다)이면 $\mu = 7/10 = 0.7$이다. 푸아송 어림을 쓰면 $P(X > 15) \approx \sum_{i=16}^{\infty} e^{-0.7} (0.7)^i / i! < 10^{-15}$이다. 세개 $m = 10^7$개에 걸쳐 넘침의 어림 개수가 $< 10^{-8}$이므로 넘침은 사실상 있을 수 없다. $\square$

---

**연습문제 3.**
넣은 적 없는 원소를 세는 블룸 필터에서 지우면 무슨 일이 생기는가? 어그러지는 결을 밝히고 막을 길을 내놓아라.

??? success "연습문제 3 풀이"
    속하지 않은 것을 지우면 다른 원소가 올려 놓았을 수 있는 세개 $k$개를 내린다. 이 세개가 줄어 0에 이르면, 본디 그것을 올렸던 원소에 거짓 음성이 생긴다. 이는 옳음을 깨뜨리는 것으로, 거짓 음성이 없다는 보장이 무너진다. 보기: 원소 A와 B가 둘 다 세개 $i$으로 해시된다. 역시 세개 $i$으로 해시되는, 속하지 않은 C을 지운다. 세개 $i$이 줄고 0에 이르면 A와 B가 모두 거짓 음성을 낸다. 막을 길: 지우기 앞서 필터에 물어 그 원소가 (아마도) 있는지 살핀다. 물음이 "없음"을 돌려주면 지우기를 건너뛴다. 이로써 엉뚱한 지우기의 거의 모두를 막지만 빈틈이 없지는 않다(거짓 양성 때문에 속하지 않은 것을 그릇되게 지울 수 있다). 옳음을 보장하려면 세는 블룸 필터 곁에 딱 맞는 집합을 따로 지녀야 한다. $\square$

---

**연습문제 4.**
세는 블룸 필터와 뻐꾸기 필터를 자리 아낌, 지우기 받침, 거짓 양성률에서 견주어라. 어느 쪽이 언제 나은가?

??? success "연습문제 4 풀이"
    **세는 블룸 필터**: 여느 블룸 필터의 4배 자리(4비트 세개)를 쓴다. 지우기를 받쳐 준다. 거짓 양성률은 $m/n$과 $k$에 달렸다. 거짓 양성률 1%에 원소마다 약 40비트다. **뻐꾸기 필터**: 뻐꾸기 해시 표에 손도장을 갈무리한다. 8비트 손도장과 채움률 95%로 거짓 양성률 1%를 이루면 원소마다 약 8.5비트다. (손도장을 없애어) 지우기를 본디부터 받쳐 준다. 세는 블룸 필터가 나은 때는 다음이다. (1) 시스템이 이미 블룸 필터를 쓰고 있어 그대로 갈아 끼울 것이 필요할 때. (2) 해시 함수 개수를 마음대로 잡아야 할 때. 뻐꾸기 필터가 나은 때는 다음이다. (1) 자리 아낌이 대수로울 때(4~5배 더 야무지다). (2) 지우기가 필요할 때(설계에 본디부터 들어 있고 세개 넘침 걱정이 없다). (3) 찾기 성능이 대수로울 때(뻐꾸기 필터는 두레박 2개만 살피지만 블룸은 아무 자리 $k$개를 살핀다). $\square$

---

**연습문제 5.**
넣기, 지우기, 묻기 연산을 받쳐 주는 세는 블룸 필터를 의사코드로 만들어라. 세개 넘침 살피기를 넣어라.

??? success "연습문제 5 풀이"
    ```
    class CountingBloomFilter:
        init(m, k):
            counters = 모두 0인 정수 m개의 배열
            hash_functions = 서로 매이지 않은 해시 함수 k개

        insert(x):
            for i in 1..k:
                pos = hash_functions[i](x) % m
                if counters[pos] < MAX_COUNT:
                    counters[pos] += 1

        delete(x):
            if not query(x):
                return  # 엉뚱한 지우기를 막는다
            for i in 1..k:
                pos = hash_functions[i](x) % m
                if counters[pos] > 0:
                    counters[pos] -= 1

        query(x):
            for i in 1..k:
                pos = hash_functions[i](x) % m
                if counters[pos] == 0:
                    return False
            return True
    ```
    `insert`의 넘침 살피기가 세개를 `MAX_COUNT`(보기로 4비트 세개에서 15)으로 막는다. 지우기의 막음은 먼저 묻는 것이고 `> 0` 살피기가 밑으로 넘침을 막는다. $\square$
