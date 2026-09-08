# 미스라-그리스 알고리즘

원소 $n$개의 흐름이 하나씩 들어올 때 $n/k$번보다 자주 나타나는 것,
곧 **큰손**을 찾고자 한다고 하자. 모든 셈을 담으려면
서로 다른 원소의 개수에 비례하는 기억이 드는데
이는 엄청날 수 있다. 미스라-그리스 알고리즘은 셈틀 $k - 1$개만으로,
흐름 길이와 상관없이 공간 $O(k)$으로 이를 푼다.

---

## 1. 문제 서술

온 모임 $\mathcal{U}$ 위의 흐름 $a_1, a_2, \dots, a_n$과 잡 $k \ge 2$이 주어질 때
잦음이 $n/k$을 넘는 원소를 모두 찾아라.

그런 원소는 많아야 $k - 1$개다(저마다 $n/k$을 넘는 원소가 $k$개면
온 나타남이 $n$을 넘어야 하기 때문이다).

---

## 2. 알고리즘

많아야 $k - 1$개의 후보-셈틀 짝 모임
$D = \{(e_i, c_i)\}$을 지닌다.

들어오는 원소 $a$마다:

1. **$a \in D$이면:** 그 셈틀을 올린다. $c_a \leftarrow c_a + 1$.
2. **아니고 $|D| < k - 1$이면:** $(a, 1)$을 $D$에 넣는다.
3. **그 밖에는:** 모든 셈틀을 $1$씩 내리고 셈이 $0$인 것을 뺀다.

흐름이 끝나면 $D$의 후보가 큰손이 될 만한 것들이다.
두 번째 지나기(또는 어림 받아들이기)로 어느 것이 참으로 문턱을
넘는지 확인한다.

!!! note "내리기의 느낌"
    내리기 연산마다 서로 다른 원소 $k$개를 뜻으로 "지운다"
    (새로 온 것에 후보 $k - 1$개). 잦음이 $> n/k$인 참 큰손은
    그 나타남을 모두 지울 만큼 다른 원소가 많지 않으므로
    살아남는다.

---

## 3. 올바름 보장

**정리.** 잦음이 $f_e > n/k$인 원소는 모두 마지막 모임 $D$에
나타난다.

**밝힘 밑그림.** 내리기 연산마다 모든 셈틀이 $1$씩 줄고
새 원소가 빠진다. 모든 연산을 통틀어 내리기의 온 횟수는
많아야 $n/k$이다(내리기마다 흐름에서 원소 $k$개를 쓰기 때문이다).
나타남이 $f_e > n/k$인 원소는 셈틀이 $f_e$번보다 적게 내려가므로
셈틀이 양수로 남는다.
$\square$

알고리즘은 $f_e \le n/k$인 원소도 남길 수 있지만(거짓 양성)
참 큰손을 놓치는 일은 결코 없다(거짓 음성 없음).

---

## 4. 어긋남 가둠

흐름을 다 다룬 뒤 $D$에 있는 원소 $e$의 어림 셈 $\hat{f}_e$은
다음을 만족한다:

$$
f_e - \frac{n}{k} \le \hat{f}_e \le f_e
$$

참 셈을 넘치게 세는 일은 없고 모자라는 정도는 많아야
$n/k$이다.

---

## 5. 구현

```python
"""
흐름 속 잦은 것을 찾는 미스라-그리스 알고리즘.

Space: O(k)
때 : 흩기 표로 짜면 고루 나누어 O(n)
"""

# === 미스라-그리스 ===
class MisraGries:
    """세개 k-1개로 잦기가 n/k를 넘는 원소를 찾는다."""

    def __init__(self, k: int):
        self.k = k
        self.counters: dict[str, int] = {}

    def process(self, item: str) -> None:
        """흐름 원소 하나를 다룬다."""
        if item in self.counters:
            self.counters[item] += 1
        elif len(self.counters) < self.k - 1:
            self.counters[item] = 1
        else:
            # 모든 셈틀을 내리고 0인 것을 뺀다
            to_remove = []
            for key in self.counters:
                self.counters[key] -= 1
                if self.counters[key] == 0:
                    to_remove.append(key)
            for key in to_remove:
                del self.counters[key]

    def get_candidates(self) -> dict[str, int]:
        """큰손 후보와 그 어림 셈을 돌려준다."""
        return dict(self.counters)

# === 확인 지나기 ===
def verify_heavy_hitters(
    stream: list[str], candidates: dict[str, int], k: int
) -> dict[str, int]:
    """후보의 정확한 셈을 얻는 두 번째 지나기."""
    counts: dict[str, int] = {c: 0 for c in candidates}
    n = len(stream)
    for item in stream:
        if item in counts:
            counts[item] += 1
    return {item: count for item, count in counts.items() if count > n // k}

# === 보기 ===
if __name__ == "__main__":
    stream = ["a", "b", "a", "c", "a", "b", "a", "d", "a", "b"]
    k = 3  # 잦음이 > 10/3 ~ 3.33인 원소 찾기

    mg = MisraGries(k)
    for item in stream:
        mg.process(item)

    candidates = mg.get_candidates()
    print(f"Candidates: {candidates}")

    confirmed = verify_heavy_hitters(stream, candidates, k)
    print(f"Confirmed heavy hitters (freq > {len(stream)//k}): {confirmed}")
```

---

## 6. 복잡도

| 측면 | 한계 |
|---|---|
| 공간 | 셈틀 $O(k)$개 |
| 원소마다 때 | 고루 나누어 $O(1)$ |
| 온 때 | $O(n)$ |
| 거짓 음성 | 0(보장됨) |
| 거짓 양성 | 많아야 $k - 1$ |

---

## 7. 딸린 알고리즘과 견주기

| 알고리즘 | 공간 | 셈 | 정해짐 |
|---|---|---|---|
| 미스라-그리스 | $O(k)$ | 어림 | 예 |
| Count-Min 스케치 | $O(w \times d)$ | 어림(넘치게 셈) | 아니오(흩기 바탕) |
| 공간 아끼기 | $O(k)$ | 어림 | 예 |

!!! tip "공간 아끼기로 넓히기"
    공간 아끼기 알고리즘(Metwally et al.)은 미스라-그리스와 같지만
    모든 셈틀을 내리는 대신 셈이 가장 작은 후보를 바꾸어
    실제로 더 빡빡한 어긋남 가둠을 준다.

---

## 연습문제

**연습문제 1.**
잦은 것을 찾는, 잡이 $k$인 미스라-그리스 알고리즘을 밝혀라.

??? success "연습문제 1 풀이"
    빈 사전 $D$으로 시작한다. 흐름 원소 $e$마다, $e \in D$이면 $D[e]$을 올린다. 아니고 $|D| < k$이면 $D[e] = 1$으로 둔다. 그 밖에는 $D$의 모든 셈을 1씩 내리고 0인 칸을 뺀다. 원소 $N$개를 다 다루면 $D$에 많아야 $k$개가 남는다. 참 잦음이 $> N/(k+1)$인 것은 $D$에 있음이 보장된다. 어림 셈 $\hat{f}_e = D[e]$(없으면 0)은 $f_e - N/(k+1) \leq \hat{f}_e \leq f_e$을 만족한다.

---

**연습문제 2.**
잦음이 $> N/(k+1)$인 것은 미스라-그리스 간추림에 살아남음을 밝혀라.

??? success "연습문제 2 풀이"
    내리기 연산마다 $k$개에서 한꺼번에 1씩 빠지며 흐름 원소를 모두 $k + 1$개 쓴다(내려간 $k$개에 내리기를 일으킨 새 것). 온 내리기 $\leq N/(k+1)$이다. 잦음이 $> N/(k+1)$인 것은 0까지 다 내려갈 수 없다. 그러려면 그것에만 $> N/(k+1)$번 내려야 하는데 온 내리기가 $\leq N/(k+1)$이기 때문이다. 그러므로 양수 셈으로 살아남는다.

---

**연습문제 3.**
미스라-그리스와 보이어-무어 과반 투표 알고리즘을 견주어라.

??? success "연습문제 3 풀이"
    보이어-무어는 미스라-그리스의 $k = 1$인 특별한 경우이다. 후보 하나와 셈틀 하나를 지닌다. 공간 $O(1)$으로 잦음이 $> N/2$인 원소(과반)를 찾는다. 미스라-그리스는 이를 넓혀 공간 $O(k)$으로 잦음이 $> N/(k+1)$인 원소를 모두 찾는다. 둘 다 정해져 있고 문턱 위의 것에 대해 정확하다(거짓 음성 없음). 다만 거짓 양성이 있을 수 있다(문턱 아래의 것이 간추림에 들 수 있다). 두 번째 지나기로 어느 후보가 참으로 잦은지 확인할 수 있다.

---

**연습문제 4.**
미스라-그리스 틀은 자연어 다루기의 낱말 다루기에 어떻게 쓰이는가?

??? success "연습문제 4 풀이"
    자연어 다루기에서 익히기 말뭉치에는 서로 다른 토막이 수백만 개 있다. 미스라-그리스(또는 공간 아끼기)는 말뭉치를 흐름으로 한 번 지나며 위 $k$개 낱말에 기억 $O(k)$을 써서 가장 잦은 토막을 가려낸다. 쓰임새: (1) 낱말 박아 넣기용 낱말 짓기($> T$번 나타나는 토막만 남김), (2) 하위 낱말 토막 내기(BPE은 글자 잦음에서 시작한다), (3) 흔한 말 찾기, (4) 익히기 전 드문 토막 거르기. 흐름 방식은 기억에 다 담기지 않는 말뭉치도 다룬다.

## 정리하며

이 마당은 문제 서술、알고리즘、올바름 보장、어긋남 가둠을 차례로 짚었다.

**참고 문헌**

- Misra, J. & Gries, D. "Finding repeated elements." *Science of Computer
  Programming*, 2(2), 1982.
- Muthukrishnan, S. "Data Streams: Algorithms and Applications." Foundations
  and Trends in Theoretical Computer Science, 2005.
