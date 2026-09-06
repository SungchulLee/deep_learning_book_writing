# 일관 해싱

분산 시스템이 단순한 나머지 해싱 $h(k) = k \bmod m$으로 서버 $m$대에 데이터를 나누어 담을 때, 서버를 하나만 더하거나 빼도 $m$이 바뀌어 거의 모든 키가 다른 서버로 옮겨 가야 한다. 수백만 요청을 처리하는 캐시 무리에서 이러한 대규모 재배치는 **캐시 사태**를 일으킨다. 거의 모든 요청이 동시에 캐시 실패가 되는 것이다. 일관 해싱은 마디를 더하거나 뺄 때 평균적으로 $O(n/m)$개의 키만 옮기게 하여 이 문제를 푼다. 여기서 $n$은 전체 키의 수, $m$은 마디의 수이다.

## 나머지 해싱의 문제

$h(k) = k \bmod m$으로 서버 $m$대에 흩어 놓은 키 $n$개를 생각해 보자. 서버의 수가 $m$에서 $m+1$으로 바뀔 때 키 $k$이 같은 서버에 남으려면 다음이 성립해야 한다.

$$
k \bmod m = k \bmod (m+1)
$$

대부분의 키에서는 이 등식이 성립하지 않는다. 옮겨야 하는 키의 비율은 다음에 다가간다.

$$
\frac{m}{m+1} \approx 1 - \frac{1}{m}
$$

서버가 $m = 100$대이면 서버 한 대를 더할 때 키의 약 99%를 다시 배치해야 한다. 대규모 시스템에서는 받아들일 수 없다.

## 해시 고리

일관 해싱은 키와 마디를 모두 크기가 $2^{b}$인 원형 해시 공간("고리")에 올린다. 여기서 $b$은 해시 함수의 출력 비트 수이다.

**준비.** 해시 함수 $h: U \to [0, 2^b)$을 고른다(예: $b = 160$인 SHA-1). 각 마디 $s_i$은 고리 위의 점 $h(s_i)$으로 해시된다. 각 키 $k$도 점 $h(k)$으로 해시된다.

**배정 규칙.** 키 $k$은 고리에서 $h(k)$부터 시계 방향으로 걸어가며 처음 만나는 마디에 배정된다. 형식적으로 키 $k$은 다음 마디에 저장된다.

$$
\text{node}(k) = \arg\min_{s_i} \bigl( h(s_i) - h(k) \bigr) \bmod 2^b
$$

여기서 최솟값은 살아 있는 모든 마디에 대해 취한다.

??? example "고리 걸어가기"
    크기가 256인 고리에서 마디 $A$, $B$, $C$이 각각 위치 10, 80, 200으로 해시되었다고 하자. $h(k) = 50$인 키 $k$은 $A$(10)와 $B$(80) 사이에 있으므로 $B$에 배정된다. $h(k) = 210$인 키는 $C$(200)와 (돌아 감아서) $A$(10) 사이에 있으므로 $A$에 배정된다.

## 마디 더하기와 빼기

일관 해싱의 핵심 이점은 마디 집합이 바뀔 때 흔들림이 가장 적다는 것이다.

**마디 더하기.** 새 마디 $s_{\text{new}}$이 들어오면 고리에서 자기 앞 마디와 자기 사이의 호에 있는 키만 넘겨받는다. 나머지 키는 지금 마디에 그대로 남는다.

**마디 빼기.** 마디 $s_i$이 빠지면 그 키들이 시계 방향의 다음 마디로 옮겨 간다. 다른 키는 움직이지 않는다.

**정리.** 키 $n$개를 담은 마디 $m$개의 시스템에서 마디를 하나 더하거나 뺄 때 다시 배치해야 하는 키의 기대 개수는 다음과 같다.

$$
\frac{n}{m+1} \quad \text{(addition)} \qquad \text{or} \qquad \frac{n}{m} \quad \text{(removal)}
$$

여기서 키가 고리 위에 고르게 흩어져 있다고 가정한다.

이는 최적이다. 마디 $m$개에 부하를 고르게 나누는 어떤 방식이든 새 마디가 들어오면 적어도 $n/(m+1)$개의 키를 다시 배정해야 한다. 새 마디도 제 몫의 키를 받아야 하기 때문이다.

## 가상 마디

고리에 물리적인 마디가 $m$개뿐이면 이웃한 마디 사이의 호 길이가 크게 들쭉날쭉해져 부하가 고르지 않게 된다. 마디가 $m$개일 때 마디마다의 부하의 표준편차는 $O(n / \sqrt{m})$으로 크다.

**가상 마디**는 물리적인 마디마다 고리 위의 자리를 $v$개씩 두어 이를 푼다. 물리 마디 $s_i$은 저마다 따로 해시되는 가상 마디 $s_i^{(1)}, s_i^{(2)}, \ldots, s_i^{(v)}$으로 나타난다.

$$
h(s_i^{(j)}) = h(\text{concat}(s_i, j)) \quad \text{for } j = 1, 2, \ldots, v
$$

물리 마디마다 가상 마디가 $v$개이면 고리에 점이 $vm$개 있게 되어 호의 길이가 더 고르게 된다. $v$이 커질수록 부하의 불균형이 줄어든다.

$$
\text{max load} \leq \frac{n}{m} \left(1 + O\!\left(\sqrt{\frac{\log(vm)}{v}}\right)\right)
$$

실무에서는 물리 마디마다 가상 마디 $v = 100$~$200$개면 균형이 좋다.

!!! tip "가상 마디의 수 고르기"
    가상 마디가 많을수록 균형이 좋아지지만 고리 자료 구조(항목 $vm$개의 정렬된 배열이나 균형 이진 탐색 나무)가 커진다. 조회 시간이 $O(\log m)$에서 $O(\log(vm))$으로 늘지만 그리 크지 않다. Amazon DynamoDB나 Apache Cassandra 같은 시스템은 보통 물리 마디마다 가상 마디를 256개 쓴다.

## 구현

고리는 (해시값, 마디 ID) 쌍의 정렬된 배열로 저장한다. 키 조회는 이진 탐색으로 키의 해시에서 시계 방향으로 처음 만나는 마디를 찾는다.

```python
"""가상 노드를 쓰는 일관 해싱."""

import hashlib
from bisect import bisect_right


# === 일관 해시 고리 ===

class ConsistentHashRing:
    """가상 노드 수를 정할 수 있는 일관 해시 고리."""

    def __init__(self, num_virtual: int = 150):
        self.num_virtual = num_virtual
        self.ring: list[tuple[int, str]] = []
        self.nodes: set[str] = set()

    def _hash(self, key: str) -> int:
        """열쇠를 고리 위의 자리로 해싱한다."""
        digest = hashlib.sha256(key.encode()).hexdigest()
        return int(digest, 16) % (2**32)

    def add_node(self, node: str) -> None:
        """물리 노드를 그 가상 노드와 함께 더한다."""
        self.nodes.add(node)
        for i in range(self.num_virtual):
            h = self._hash(f"{node}:{i}")
            self.ring.append((h, node))
        self.ring.sort()

    def remove_node(self, node: str) -> None:
        """물리 노드와 그 가상 노드를 모두 없앤다."""
        self.nodes.discard(node)
        self.ring = [(h, n) for h, n in self.ring if n != node]

    def get_node(self, key: str) -> str:
        """주어진 열쇠를 맡은 노드를 찾는다."""
        if not self.ring:
            raise ValueError("Empty ring")
        h = self._hash(key)
        idx = bisect_right(self.ring, (h,))
        if idx == len(self.ring):
            idx = 0  # 한 바퀴 돌아 감기
        return self.ring[idx][1]


# === 시연 ===

if __name__ == "__main__":
    ring = ConsistentHashRing(num_virtual=150)
    for node in ["server-A", "server-B", "server-C"]:
        ring.add_node(node)

    # 열쇠 1000개를 배정하고 분포 세기
    from collections import Counter
    counts = Counter(ring.get_node(f"key-{i}") for i in range(1000))
    print("Distribution with 3 nodes:")
    for node, count in sorted(counts.items()):
        print(f"  {node}: {count} keys")

    # 노드를 하나 더하고 옮겨진 열쇠 수 재기
    old_assignments = {f"key-{i}": ring.get_node(f"key-{i}") for i in range(1000)}
    ring.add_node("server-D")
    moved = sum(
        1 for k in old_assignments if ring.get_node(k) != old_assignments[k]
    )
    print(f"\nAfter adding server-D: {moved}/1000 keys moved")
    print(f"Expected: ~{1000 // 4} keys (n/(m+1) = 1000/4)")
```

## 복잡도

| 연산 | 시간 | 공간 |
|---|---|---|
| 마디 더하기 | $O(v \log(vm))$ | 전체 $O(vm)$ |
| 마디 빼기 | $O(vm)$ | 전체 $O(vm)$ |
| 키 조회 | $O(\log(vm))$ | -- |

여기서 $v$은 물리 마디마다의 가상 마디 수이고 $m$은 물리 마디의 수이다.

## 부하가 제한된 일관 해싱

보통의 일관 해싱은 부하에 대한 확고한 보장을 주지 않는다. **부하가 제한된 일관 해싱**(Mirrokni 등, 2018)은 용량 제약을 더한다. 조율 가능한 매개변수 $\varepsilon > 0$에 대해 각 마디가 많아야 $(1 + \varepsilon) \cdot n/m$개의 키를 갖는다.

키의 시계 방향 다음 마디가 가득 찼으면 키는 여유가 있는 마디를 만날 때까지 시계 방향으로 계속 간다. 이로써 최악의 경우 부하에 한계가 생긴다.

$$
\text{max load per node} \leq \left\lceil (1 + \varepsilon) \cdot \frac{n}{m} \right\rceil
$$

구글의 부하 분산기는 이 변형으로 뒷단 서버에 트래픽을 나눈다.

## 응용

일관 해싱은 분산 시스템의 바탕이 된다.

- **분산 캐시** (Memcached, Redis Cluster): 규모를 바꿀 때 재배치를 최소로 하며 캐시 키를 서버에 나눈다.
- **분산 데이터베이스** (Amazon DynamoDB, Apache Cassandra): 자동으로 균형을 맞추며 데이터를 저장 마디에 나눈다.
- **콘텐츠 전송망** (Akamai): 요청한 내용을 가진 가장 가까운 캐시로 요청을 보낸다.
- **부하 분산기**: 세션의 연속성을 지키면서 요청을 뒷단 서버에 나눈다.

## 참고 문헌

- Karger, D., Lehman, E., Leighton, T., Panigrahy, R., Levine, M., & Lewin, D. (1997). Consistent hashing and random trees. *Proceedings of the 29th ACM Symposium on Theory of Computing (STOC)*, 654--663.
- Mirrokni, V., Thorup, M., & Zadimoghaddam, M. (2018). Consistent hashing with bounded loads. *Proceedings of the 29th ACM-SIAM Symposium on Discrete Algorithms (SODA)*.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 11. MIT Press.


## 연습문제

**연습문제 1.**
일관 해싱에 대해, 적재율이 $\alpha = 0.75$일 때 삽입과 조회의 기대 시간과 최악의 경우 시간을 계산하라.

??? success "연습문제 1 풀이"
    기대 시간은 충돌 해결 전략에 달렸으며 균등 해싱을 가정한다. 체이닝에서는 기대 시간이 $O(1 + \alpha) = O(1.75)$이다. 개방 주소법에서는 탐색에 실패할 때 기대 탐사 횟수가 $\approx 1/(1-\alpha) = 4$이다. 최악의 경우는 모든 키가 같은 칸으로 해시될 때의 $O(n)$이다.

---

**연습문제 2.**
일관 해싱을(를) 써서 키 10, 22, 31, 4, 15, 28, 17을 크기가 7인 해시 테이블에 넣어라. 최종 테이블의 상태를 보여라.

??? success "연습문제 2 풀이"
    해시 함수 $h(k) = k \bmod 7$을 적용하고 이 쪽의 방법으로 충돌을 처리한다. 키마다 해시를 계산하고 충돌을 해결한 뒤 키를 놓는다. 최종 테이블의 내용을 보인다.

---

**연습문제 3.**
일관 해싱은(는) 딥러닝의 임베딩 테이블에서 어떻게 쓰이는가? 토큰 $V = 50{,}000$개의 어휘를 $m = 30{,}000$개의 버킷에 대응시킬 때 충돌의 양상을 분석하라.

??? success "연습문제 3 풀이"
    $V/m \approx 1.67$이므로 비둘기집 원리에 의해 충돌이 반드시 생긴다. 버킷마다 평균 1.67개의 토큰이 같은 임베딩을 나누어 쓴다. 충돌률과 그것이 모델의 품질에 미치는 영향은 해시 함수의 품질과 임베딩의 차원에 달렸다(차원이 높을수록 충돌을 더 잘 견딘다).

---

**연습문제 4.**
$\alpha > 0.75$일 때 해시 테이블의 크기를 다시 잡으면 삽입의 상각 비용이 $O(1)$으로 유지됨을 증명하라.

??? success "연습문제 4 풀이"
    크기를 다시 잡는 사이(용량 $m$에서 $2m$까지)에 삽입이 $m/4$번 일어난다(적재율이 $0.375$에서 $0.75$로 간다). 크기 조정에는 $O(m)$이 든다. 삽입 하나당 상각된 크기 조정 비용은 $O(m)/(m/4) = O(4) = O(1)$이다. 여기에 (균등 해싱 아래) 삽입마다의 기대 비용 $O(1)$을 더하면 전체 상각 비용은 $O(1)$이다. $\square$