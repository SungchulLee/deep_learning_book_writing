# 버퍼 나무

여느 B 나무는 넣기와 지우기를 하나씩 다루어 고침마다 들고남 $O(\log_B N)$번이 든다. 알고리즘이 고침을 차례로 $N$번 하면 온 비용이 들고남 $O(N \log_B N)$번이다. **버퍼 나무**는 안쪽 마디마다 버퍼를 달아 고침을 묶어 모으고 버퍼가 찰 때만 나무 아래로 흘려보내 이를 좋게 한다. 이 고루 나누기가 고침마다 들고남 비용을 $O((1/B) \log_{M/B}(N/B))$으로 줄이며 이는 줄 세우기 가둠을 $N$으로 나눈 값과 같다.

---

## 1. 핵심 생각

버퍼 나무는 갈래 수가 $\Theta(M/B)$인 B 나무이며 안쪽 마디마다 원소를 많아야 $M$개(덩이 $M/B$개) 담는 **버퍼**가 딸려 있다. 고침이 오면 뿌리의 버퍼에 그냥 덧붙인다. 어느 버퍼든 가득 차면 그 속을 줄 세워 알맞은 자식 버퍼로 나누고 어버이 버퍼를 비운다. 이 묶음 흘려보내기가 고침을 나무 아래로 밀어 내리는 들고남 비용을 고루 나눈다.

---

## 2. 구조

원소 $N$개의 버퍼 나무는 다음 성질을 가진다:

| 성질 | 값 |
|---|---|
| 갈래 수 | $\Theta(M/B)$ |
| 마디마다 버퍼 크기 | 원소 $M$개(덩이 $M/B$개) |
| 나무 높이 | $O\!\left(\log_{M/B} \frac{N}{B}\right)$ |
| 잎 | 크기 $B$의 덩이에 줄 세운 자료를 담는다 |

주어진 켜의 모든 마디를 통틀은 온 버퍼 공간은 많아야 원소 $O(N)$개다. 어느 때든 원소마다 많아야 버퍼 하나에만 있기 때문이다.

---

## 3. 고루 나눈 들고남 살피기

크기 $M$의 버퍼를 흘려보낼 때 원소 $M$개를 속에서 줄 세우고(들고남 비용 없음) 자식 $\Theta(M/B)$개에 나눈다. 흘려보내기는 덩이 $O(M/B)$개를 읽고 적는다. 흘려보내기 전에 원소 $M$개를 묶었으므로 원소마다 고루 나눈 비용은 다음과 같다:

$$
\frac{O(M/B)}{M} = O\!\left(\frac{1}{B}\right) \text{ per element per level}
$$

원소마다 뿌리에서 잎까지 켜 $O(\log_{M/B}(N/B))$개를 지나므로 고침마다 온 고루 나눈 들고남은 다음과 같다:

$$
O\!\left(\frac{1}{B} \log_{M/B} \frac{N}{B}\right)
$$

고침 $N$번의 온 들고남은 다음과 같다:

$$
O\!\left(\frac{N}{B} \log_{M/B} \frac{N}{B}\right) = O(\text{sort}(N))
$$

이는 견줌 바탕 알고리즘에서 가장 좋은 값인 바깥 기억 줄 세우기 가둠과 맞는다.

---

## 4. 연산

### 삽입

넣기 연산은 뿌리 버퍼에 그냥 덧붙인다. 뿌리 버퍼가 원소 $M$개에 이르면 흘려보내기가 그것을 자식 버퍼로 민다. 자식 버퍼도 차면 흘려보내기가 되돌이로 줄줄이 번진다.

### 지우기

지우기는 **반원소**로 다룬다. 열쇠 $k$의 지움 표시를 뿌리 버퍼에 넣는다. 흘려보내는 동안 지움 표시가 해당 원소를 만나면 둘 다 없앤다. 이 느긋한 지우기가 같은 고루 나눈 가둠을 지킨다.

### 찾기

점 묻기는 찾기 길의 마디마다 버퍼를 살펴야 한다. 기다리는 고침이 아직 잎에 닿지 않았을 수 있기 때문이다. 드는 값은 다음과 같다:

$$
O\!\left(\frac{M}{B} \cdot \log_{M/B} \frac{N}{B}\right)
$$

이는 여느 B 나무 찾기보다 비싸다. 그래서 버퍼 나무는 **고침이 많고 묻기가 적은** 일감이나 모든 고침 뒤에 모든 묻기를 하는 묶음 연산에 가장 알맞다.

---

## 5. B 나무와 견주기

| 성질 | B 나무 | 버퍼 나무 |
|---|---|---|
| 갈래 수 | $\Theta(B)$ | $\Theta(M/B)$ |
| 찾기 들고남 | $O(\log_B N)$ | $O((M/B) \log_{M/B}(N/B))$ |
| 넣기 들고남(가장 나쁜 경우) | $O(\log_B N)$ | $O(\log_{M/B}(N/B))$ |
| 넣기 들고남(고루 나눔) | $O(\log_B N)$ | $O((1/B) \log_{M/B}(N/B))$ |
| 알맞은 곳 | 묻기가 많음 | 고침이 많음 / 묶음 |

---

## 6. 보기: 버퍼 나무 흘려보내기 흉내내기

```python
"""
버퍼 나무 흘려보내기 흉내내기.

버퍼 나무가 고침을 묶고 버퍼가 찰 때만 흘려보내
들고남 비용을 고루 나누는 모습을 보여 준다.
"""

import math

# ===================================================================
# 버퍼 나무 흉내내기
# ===================================================================

class BufferTreeNode:
    """담는 양이 붙박인 버퍼를 가진 버퍼 나무의 마디."""

    def __init__(self, branching_factor: int, buffer_capacity: int,
                 leaf: bool = False):
        self.branching_factor = branching_factor
        self.buffer_capacity = buffer_capacity
        self.buffer: list[int] = []
        self.leaf = leaf
        self.children: list[BufferTreeNode] = []
        self.io_count = 0

    def add_to_buffer(self, key: int) -> int:
        """버퍼에 열쇠를 더한다. 쓴 들고남 횟수를 돌려준다."""
        self.buffer.append(key)
        ios = 0
        if len(self.buffer) >= self.buffer_capacity:
            ios = self._flush()
        return ios

    def _flush(self) -> int:
        """버퍼 속을 자식에게 흘려보낸다. 들고남 횟수를 돌려준다."""
        if self.leaf:
            # 잎에서는 버퍼만 비운다(줄 세운 자료를 적는다)
            ios = math.ceil(len(self.buffer) / 100)  # 덩이 적기를 흉내 낸다
            self.buffer.clear()
            return ios

        # 버퍼를 줄 세워 자식에게 나눈다
        self.buffer.sort()
        blocks_read_write = math.ceil(
            len(self.buffer) * 2 / 100  # 읽기 + 적기
        )

        # 자식에게 나눈다
        child_ios = 0
        per_child = len(self.buffer) // max(1, len(self.children))
        for i, child in enumerate(self.children):
            start = i * per_child
            end = start + per_child if i < len(self.children) - 1 \
                else len(self.buffer)
            for key in self.buffer[start:end]:
                child_ios += child.add_to_buffer(key)

        self.buffer.clear()
        return blocks_read_write + child_ios

def simulate_buffer_tree(n: int, m: int, b: int) -> dict:
    """
    버퍼 나무에 넣기 N번을 흉내 낸다.

    매개변수
    ----------
    n : 넣을 원소의 개수.
    m : 기억이 담는 양(마디마다 버퍼 크기).
    b : 덩이 크기.

    반환값
    -------
    들고남 통계를 담은 사전.
    """
    fan_out = max(2, m // b)
    height = max(1, math.ceil(math.log(max(1, n / b)) / math.log(fan_out)))

    # 고루 나눈 가둠을 셈한다
    amortized_per_element = (1 / b) * height
    total_amortized = n * amortized_per_element
    btree_total = n * math.ceil(math.log(max(2, n)) / math.log(max(2, b)))

    return {
        "fan_out": fan_out,
        "height": height,
        "amortized_per_element": amortized_per_element,
        "total_buffer_tree": total_amortized,
        "total_btree": btree_total,
        "speedup": btree_total / max(1, total_amortized),
    }

# ===================================================================
# 메인
# ===================================================================

if __name__ == "__main__":
    N = 10**7
    M = 10**5
    B = 1000

    stats = simulate_buffer_tree(N, M, B)
    print(f"Buffer Tree vs B-Tree for N={N:,} insertions")
    print(f"  M = {M:,}, B = {B:,}")
    print(f"  Fan-out (M/B):        {stats['fan_out']}")
    print(f"  Height:               {stats['height']}")
    print(f"  Amortized I/O/insert: {stats['amortized_per_element']:.4f}")
    print(f"  Total buffer tree:    {stats['total_buffer_tree']:,.0f} I/Os")
    print(f"  Total B-tree:         {stats['total_btree']:,.0f} I/Os")
    print(f"  Speedup:              {stats['speedup']:.1f}x")
```

??? example "보기 내놓기"

    ```
    Buffer Tree vs B-Tree for N=10,000,000 insertions
      M = 100,000, B = 1,000
      퍼짐(M/B):        100
      높이:               2
      넣기마다 고루 나눈 들고남: 0.0020
      버퍼 나무 온 들고남:    20,000번
      B 나무 온 들고남:         30,000,000번
      빨라짐:              1500.0배
    ```

    버퍼 나무는 고침을 묶어 온 들고남을 1500배 줄이며 고루 나눈 들고남 살피기의 힘을 보여 준다.

---

## 7. 응용

버퍼 나무는 묻기 결과가 필요하기 전에 고침을 많이 하는 알고리즘에 쓰인다:

- **바깥 기억 앞섬 줄:** 넣기와 최소 지우기를 버퍼에 담아 연산마다 고루 나눈 들고남 $O((1/B) \log_{M/B}(N/B))$번을 이룬다.
- **바깥 기억 그래프 알고리즘:** 바깥 기억 너비 우선 찾기와 최소 뻗은 나무 같은 알고리즘이 변 느슨히 하기를 묶으려 버퍼 나무를 쓴다.
- **색인 한꺼번에 올리기:** B 나무 색인을 맨바닥부터 세울 때 버퍼 나무가 하나씩 넣는 덧짐을 피한다.

---

## 연습문제

**연습문제 1.**
버퍼 나무 자료 얼개와 그 고루 나눈 들고남 복잡도를 밝혀라.

??? success "연습문제 1 풀이"
    버퍼 나무는 안쪽 마디마다 덩이 $O(M/B)$개 크기의 버퍼를 가진 B 나무다. 연산(넣기, 지우기, 묻기)을 먼저 뿌리의 버퍼에 놓는다. 버퍼가 넘치면(덩이 $> M/B$개) '흘려보낸다'. 버퍼를 줄 세워 원소를 자식 버퍼로 나눈다. 이는 들고남을 묶는다. 흘려보내기 한 번이 연산 $O(M/B)$개를 다룬다. 연산마다 고루 나눈 비용: 들고남 $O(1/B \cdot \log_{M/B}(N/B))$번이며 이는 줄 세우기 아래 가둠을 $N$으로 나눈 값과 맞는다.

---

**연습문제 2.**
버퍼 나무가 한꺼번에 하는 연산에서 여느 B 나무보다 고루 나눈 들고남이 왜 나은지 밝혀라.

??? success "연습문제 2 풀이"
    여느 B 나무: 넣기마다 들고남 $O(\log_B N)$번. 넣기 $N$번이면 모두 $O(N \log_B N)$. 버퍼 나무: 넣기 $N$번에 모두 $O(N/B \cdot \log_{M/B}(N/B))$, 곧 줄 세우기 가둠이다. 나아짐의 배수는 $B \log_B N / \log_{M/B}(N/B) \approx B$이며 $B$이 크면 크다. 핵심 통찰: 버퍼에 담으면 들고남마다 원소 1개가 아니라 $B$개를 다루어 원소마다 들고남 비용이 고루 나뉜다.

---

**연습문제 3.**
버퍼 나무는 어떤 연산을 효율 좋게 받쳐 주는가?

??? success "연습문제 3 풀이"
    버퍼 나무는 다음을 받쳐 준다. (1) 넣기: 고루 나눈 들고남 $O(1/B \cdot \log_{M/B}(N/B))$번, (2) 지우기: 넣기와 같다(반원소로 하는 느긋한 지우기), (3) 묶음 묻기: 고루 나누어 $O(1/B \cdot \log_{M/B}(N/B))$번, (4) 구간 묻기: $K$이 내놓기 크기일 때 $O(1/B \cdot \log_{M/B}(N/B) + K/B)$번. 결과가 기다리는 버퍼 흘려보내기에 매이므로 묻기가 조금 늦어도 되는, 적기가 많은 일감에 안성맞춤이다.

---

**연습문제 4.**
버퍼 나무는 요즘 자료 바탕이 쓰는 LSM 나무와 어떻게 이어지는가?

??? success "연습문제 4 풀이"
    LSM 나무(기록 얼개 합침 나무)는 버퍼 원칙을 함께 쓴다. 적기가 기억 속 버퍼(멤테이블)로 가고 가득 차면 줄 세운 줄기로 원반에 흘려보낸다. 때때로 하는 다지기가 줄 세운 줄기를 합치며 이는 버퍼 나무의 흘려보내기와 비슷하다. 둘 다 적기에 맞춘 들고남을 이룬다. 적기마다 고루 나누어 $O(1/B \cdot \log_{M/B}(N/B))$번이다. LSM 나무는 짜기가 더 단순하고 널리 쓰인다(LevelDB, RocksDB, Cassandra). 버퍼 나무는 묻기의 가장 나쁜 경우를 더 잘 보장한다. 설계 철학은 같다. 연산을 묶고 줄 세워 아무 들고남을 차례 들고남으로 바꾼다.

## 정리하며

이 마당은 핵심 생각、구조、고루 나눈 들고남 살피기、연산을 차례로 짚었다.

**참고 문헌**

- Arge, L. "The Buffer Tree: A Technique for Designing Batched External Data Structures," *Algorithmica*, 37(1), 2003.
- Vitter, J. S. *Algorithms and Data Structures for External Memory*, 2008.
