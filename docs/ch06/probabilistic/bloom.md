# 블룸 필터

보통의 해시 집합은 모든 원소를 그대로 저장하여 원소 $n$개에 $O(n)$ 공간을 쓴다. 목적이 그저 "이 원소를 본 적이 있는가?"라는 소속 판정이라면 **블룸 필터**가 훨씬 적은 메모리로 답한다. 대신 거짓 양성이 작은 확률로 생긴다. 블룸 필터는 거짓 음성을 절대 내지 않는다. 없다고 답하면 정말로 없는 것이다.

## 구조

블룸 필터는 다음으로 이루어진다.

- 처음에 모두 0인 $m$비트짜리 **비트 배열** $B[0 \ldots m-1]$.
- 원소를 $\{0, 1, \ldots, m-1\}$으로 보내는 독립인 해시 함수 $h_1, h_2, \ldots, h_k$ 묶음.

원소 $x$ **넣기**: 각 $i \in \{1, \ldots, k\}$에 대해 $B[h_i(x)] = 1$으로 둔다.

원소 $x$ **질의**: 모든 $i$에 대해 $B[h_i(x)] = 1$이면 "있을 수 있음"을, 어느 하나라도 $B[h_i(x)] = 0$이면 "확실히 없음"을 돌려준다.

서로 다른 원소가 같은 비트를 세울 수 있으므로, 넣은 적 없는 원소에 대해서도 질의가 "있을 수 있음"을 돌려줄 수 있다. 이것이 **거짓 양성**이다. 그러나 $k$개 비트 가운데 하나라도 0이면 그 원소는 넣은 적이 없다. **거짓 음성은 없다.**

## 거짓 양성의 확률

비트가 $m$개이고 해시 함수가 $k$개인 블룸 필터에 원소 $n$개를 넣은 뒤, 특정 비트가 여전히 0일 확률은 다음과 같다.

$$
\left(1 - \frac{1}{m}\right)^{kn} \approx e^{-kn/m}
$$

구성원이 아닌 원소의 $k$개 비트가 모두 1일 때 거짓 양성이 일어난다. 거짓 양성의 확률은 다음과 같다.

$$
P(\text{FP}) \approx \left(1 - e^{-kn/m}\right)^k
$$

## 해시 함수의 최적 개수

주어진 $m$과 $n$에 대해 $P(\text{FP})$을 최소화하려면 $k$으로 미분하여 0으로 놓는다. 해시 함수의 최적 개수는 다음과 같다.

$$
k^* = \frac{m}{n} \ln 2 \approx 0.693 \cdot \frac{m}{n}
$$

다시 대입하면 최적 $k$에서의 최소 거짓 양성률은 다음과 같다.

$$
P(\text{FP})_{\min} = \left(\frac{1}{2}\right)^{k^*} = 2^{-(m/n) \ln 2}
$$

## 원소당 비트 수

목표 거짓 양성률이 $\epsilon$일 때 필요한 원소당 비트 수는 다음과 같다.

$$
\frac{m}{n} = -\frac{\ln \epsilon}{(\ln 2)^2} \approx -1.44 \log_2 \epsilon
$$

| 목표 거짓 양성률 $\epsilon$ | 원소당 비트 수 $m/n$ | 해시 함수 수 $k^*$ |
|---|---|---|
| 1% | 9.6 | 7 |
| 0.1% | 14.4 | 10 |
| 0.01% | 19.2 | 13 |

거짓 양성률 1%에는 원소당 10비트도 들지 않는다. 원소 자체(수백 바이트짜리 문자열일 수도 있다)를 저장하는 것보다 훨씬 적다.

## 한계

- **삭제 불가**: 비트를 0으로 되돌리면 다른 원소에 영향을 준다. 계수 블룸 필터는 비트 하나를 세개로 바꾸어 삭제를 지원한다.
- **열거 불가**: 필터에 담긴 원소를 나열할 수 없다.
- **고정된 용량**: 설계한 용량을 넘겨 원소를 더 넣을수록 거짓 양성률이 올라간다.

## 응용

블룸 필터는 다음에 쓰인다.

- **웹 캐싱**: 크롬의 안전 브라우징은 서버에 묻기 전에 블룸 필터로 URL을 확인한다.
- **데이터베이스 질의 최적화**: 없는 키를 위해 값비싼 디스크 읽기를 하지 않는다.
- **네트워크 라우팅**: 고속 스위치에서 중복 패킷을 찾아낸다.
- **맞춤법 검사**: 사전에 확실히 없는 낱말을 빠르게 걸러 낸다.

## 파이썬 구현

```python
"""
블룸 필터 구현.

거짓 음성이 없는 근사 소속 판정을 위한
공간 효율적인 확률적 자료 구조를 보인다.
"""

import hashlib
import math


# === 블룸 필터 ===

class BloomFilter:
    """크기와 해시 개수를 정할 수 있는 블룸 필터."""

    def __init__(self, expected_items, fp_rate=0.01):
        # 최적의 m과 k 계산
        self.size = self._optimal_size(expected_items, fp_rate)
        self.num_hashes = self._optimal_hashes(self.size, expected_items)
        self.bits = [False] * self.size
        self.count = 0

    @staticmethod
    def _optimal_size(n, p):
        """원소 n개와 거짓 양성률 p에 대한 최적의 비트 배열 크기를 계산한다."""
        m = -n * math.log(p) / (math.log(2) ** 2)
        return int(math.ceil(m))

    @staticmethod
    def _optimal_hashes(m, n):
        """최적의 해시 함수 개수를 계산한다."""
        k = (m / n) * math.log(2)
        return max(1, int(round(k)))

    def _hashes(self, item):
        """주어진 원소에 대해 해시값 k개를 만든다."""
        for i in range(self.num_hashes):
            h = int(hashlib.md5(
                f"{item}:{i}".encode()
            ).hexdigest(), 16)
            yield h % self.size

    def add(self, item):
        """블룸 필터에 원소를 넣는다."""
        for h in self._hashes(item):
            self.bits[h] = True
        self.count += 1

    def query(self, item):
        """원소가 필터에 있을 수 있는지 확인한다.

        있을 수 있으면 True를, 확실히 없으면 False를 돌려준다.
        """
        return all(self.bits[h] for h in self._hashes(item))


# === 시연 ===

if __name__ == "__main__":
    bf = BloomFilter(expected_items=100, fp_rate=0.01)
    print(f"Bit array size: {bf.size}")
    print(f"Hash functions: {bf.num_hashes}")

    # 원소 몇 개 넣기
    present = ["apple", "banana", "cherry", "date", "elderberry"]
    for item in present:
        bf.add(item)

    # 있는 원소와 없는 원소 물어보기
    test = ["apple", "banana", "grape", "melon", "cherry", "kiwi"]
    for item in test:
        status = "possibly present" if bf.query(item) else "definitely absent"
        print(f"  {item:12s} -> {status}")
```

**출력:**
```
Bit array size: 959
Hash functions: 7
  apple        -> possibly present
  banana       -> possibly present
  grape        -> definitely absent
  melon        -> definitely absent
  cherry       -> possibly present
  kiwi         -> definitely absent
```

## 참고 문헌

- [Introduction to Algorithms (CLRS), Chapter 11](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Bloom, B. H. "Space/Time Trade-offs in Hash Coding with Allowable Errors." *Communications of the ACM*, 13(7), 1970.


## 연습문제

**연습문제 1.**
블룸 필터에 대해, 적재율이 $\alpha = 0.75$일 때 삽입과 조회의 기대 시간과 최악의 경우 시간을 계산하라.

??? success "연습문제 1 풀이"
    기대 시간은 충돌 해결 전략에 달렸으며 균등 해싱을 가정한다. 체이닝에서는 기대 시간이 $O(1 + \alpha) = O(1.75)$이다. 개방 주소법에서는 탐색에 실패할 때 기대 탐사 횟수가 $\approx 1/(1-\alpha) = 4$이다. 최악의 경우는 모든 키가 같은 칸으로 해시될 때의 $O(n)$이다.

---

**연습문제 2.**
블룸 필터을(를) 써서 키 10, 22, 31, 4, 15, 28, 17을 크기가 7인 해시 테이블에 넣어라. 최종 테이블의 상태를 보여라.

??? success "연습문제 2 풀이"
    해시 함수 $h(k) = k \bmod 7$을 적용하고 이 쪽의 방법으로 충돌을 처리한다. 키마다 해시를 계산하고 충돌을 해결한 뒤 키를 놓는다. 최종 테이블의 내용을 보인다.

---

**연습문제 3.**
블룸 필터은(는) 딥러닝의 임베딩 테이블에서 어떻게 쓰이는가? 토큰 $V = 50{,}000$개의 어휘를 $m = 30{,}000$개의 버킷에 대응시킬 때 충돌의 양상을 분석하라.

??? success "연습문제 3 풀이"
    $V/m \approx 1.67$이므로 비둘기집 원리에 의해 충돌이 반드시 생긴다. 버킷마다 평균 1.67개의 토큰이 같은 임베딩을 나누어 쓴다. 충돌률과 그것이 모델의 품질에 미치는 영향은 해시 함수의 품질과 임베딩의 차원에 달렸다(차원이 높을수록 충돌을 더 잘 견딘다).

---

**연습문제 4.**
$\alpha > 0.75$일 때 해시 테이블의 크기를 다시 잡으면 삽입의 상각 비용이 $O(1)$으로 유지됨을 증명하라.

??? success "연습문제 4 풀이"
    크기를 다시 잡는 사이(용량 $m$에서 $2m$까지)에 삽입이 $m/4$번 일어난다(적재율이 $0.375$에서 $0.75$로 간다). 크기 조정에는 $O(m)$이 든다. 삽입 하나당 상각된 크기 조정 비용은 $O(m)/(m/4) = O(4) = O(1)$이다. 여기에 (균등 해싱 아래) 삽입마다의 기대 비용 $O(1)$을 더하면 전체 상각 비용은 $O(1)$이다. $\square$