# 나눠진 흩기 표

가운데 모은 열쇠-값 곳간은 함께하는 이가 늘수록 병목이 된다.
나눠진 흩기 표는 담기와 찾기의 몫을 마디 $n$개에 흩어, 어떤 열쇠든
$O(\log n)$번 건너 찾을 수 있게 하고 한 곳이 무너져도 멈추지 않게 한다.
나눠진 흩기 표는 동무끼리 시스템, 내용으로 찾는 그물,
나눠진 자료 바탕을
떠받친다.

---

## 1. 핵심 생각

마디마다 그리고 열쇠마다 흩기 함수(보통 SHA-1)로 둥근 번호 자리
$\{0, 1, \dots, 2^m - 1\}$에서 번호를 받는다.
열쇠 $k$은 둥근 자리에서 $k$ 바로 다음에 오는 번호를 가진 마디,
곧 $k$의 **다음 마디**에 담긴다.

---

## 2. 한결같은 흩기

여느 흩기($h(k) = k \bmod n$)는 $n$이 바뀌면 거의 모든 열쇠를 다시
옮겨야 한다. 한결같은 흩기는 열쇠와 마디를 모두 크기 $2^m$의 고리에
옮긴다:

$$
\text{node\_id} = \text{hash}(\text{IP address}), \quad \text{key\_id} = \text{hash}(\text{key})
$$

열쇠는 고리 위 제 자리에서 시계 방향으로 처음 만나는 마디에 맡겨진다.
마디가 들어오거나 나가면 ($K$이 온 열쇠 수일 때) 열쇠 $O(K/n)$개만
다시 맡기면 된다. 법 흩기의 $O(K)$과 견주어 훨씬
적다.

---

## 3. Chord 규약

Chord(Stoica et al., 2001)은 본보기가 되는 나눠진 흩기 표로 마디마다
상태 $O(\log n)$으로 $O(\log n)$ 찾기를 준다.

### 손가락 표

마디 $p$마다 칸 $m$개의 **손가락 표**를 지닌다:

$$
\text{finger}[i] = \text{successor}(p + 2^{i-1}) \quad \text{for } i = 1, 2, \dots, m
$$

$i$번째 손가락은 번호 $p + 2^{i-1} \pmod{2^m}$을 맡은 마디를 가리킨다.
손가락은 고리를 따라 지수로 늘어나는 거리를 뻗어, 건널 때마다 남은 거리를
반으로 줄인다.

### 찾기

열쇠 $k$을 맡은 마디를 찾으려면:

1. $k$이 지금 마디 $p$과 그 다음 마디 사이에 있으면 다음 마디를
   돌려준다.
2. 그렇지 않으면 바로 앞서는 가장 가까운 손가락에 묻기를 넘긴다.

건널 때마다 과녁까지의 거리가 적어도 반으로 줄어 기댓값으로 $O(\log n)$번
건넌다.

### 마디 들어오기

새 마디 $p$이 들어오면:

1. 이미 있는 마디의 찾기로 $p$의 다음 마디를 찾는다.
2. 다음 마디에서 알맞은 열쇠를 $p$으로 옮긴다.
3. 다른 마디의 손가락 표를 고친다(안정 규약으로 느긋하게
   한다).

### 마디 나가기

마디 $p$이 나가면:

1. $p$의 열쇠를 다음 마디로 옮긴다.
2. 다른 마디의 손가락 표는 때때로 하는 안정 과정에서 고쳐진다.

---

## 4. 복잡도

| 연산 | 건너기 | 쪽지 |
|---|---|---|
| 찾기 | $O(\log n)$ | $O(\log n)$ |
| 열쇠 넣기 | $O(\log n)$ | $O(\log n)$ |
| 들어오기 | $O(\log^2 n)$ | $O(\log^2 n)$ |
| 마디마다 상태 | 칸 $O(\log n)$개 | --- |

---

## 5. 구현

```python
"""
간단히 한 Chord 방식 나눠진 흩기 표 흉내내기.

한결같은 흩기와 손가락 표 바탕 찾기를 보여 준다.
"""

import hashlib

# === 한결같은 흩기 ===
def hash_id(key: str, m: int = 8) -> int:
    """열쇠를 2^m 고리 위 자리로 흩는다."""
    digest = hashlib.sha1(key.encode()).hexdigest()
    return int(digest, 16) % (2**m)

# === Chord 마디 ===
class ChordNode:
    """간단히 한 Chord 나눠진 흩기 표의 마디."""

    def __init__(self, node_id: int, m: int = 8):
        self.node_id = node_id
        self.m = m
        self.ring_size = 2**m
        self.finger_table: list[int] = []
        self.data: dict[int, str] = {}
        self.successor: int = node_id

    def build_finger_table(self, sorted_nodes: list[int]) -> None:
        """모든 마디 번호를 줄 세운 목록에서 손가락 표를 짓는다."""
        self.finger_table = []
        for i in range(self.m):
            target = (self.node_id + 2**i) % self.ring_size
            # 과녁의 다음 마디를 찾는다
            succ = sorted_nodes[0]  # 한 바퀴 돌아 감기
            for n in sorted_nodes:
                if n >= target:
                    succ = n
                    break
            self.finger_table.append(succ)
        self.successor = self.finger_table[0]

# === 나눠진 흩기 표 ===
class ChordDHT:
    """손가락 표 찾기를 쓰는 간단히 한 Chord 나눠진 흩기 표."""

    def __init__(self, m: int = 8):
        self.m = m
        self.nodes: dict[int, ChordNode] = {}
        self.sorted_ids: list[int] = []

    def add_node(self, node_id: int) -> None:
        """나눠진 흩기 표에 마디를 더한다."""
        node = ChordNode(node_id, self.m)
        self.nodes[node_id] = node
        self.sorted_ids = sorted(self.nodes.keys())
        # 모든 손가락 표를 다시 짓는다
        for n in self.nodes.values():
            n.build_finger_table(self.sorted_ids)

    def lookup(self, key: str) -> tuple[int, int]:
        """열쇠를 맡은 마디를 찾는다. (key_id, node_id)을 돌려준다."""
        key_id = hash_id(key, self.m)
        # key_id의 다음 마디를 찾는다
        responsible = self.sorted_ids[0]
        for nid in self.sorted_ids:
            if nid >= key_id:
                responsible = nid
                break
        return key_id, responsible

    def put(self, key: str, value: str) -> int:
        """열쇠-값 짝을 담는다. 맡은 마디 번호를 돌려준다."""
        key_id, node_id = self.lookup(key)
        self.nodes[node_id].data[key_id] = value
        return node_id

    def get(self, key: str) -> str | None:
        """열쇠로 값을 가져온다."""
        key_id, node_id = self.lookup(key)
        return self.nodes[node_id].data.get(key_id)

# === 보기 ===
if __name__ == "__main__":
    dht = ChordDHT(m=8)
    for nid in [0, 32, 64, 128, 192]:
        dht.add_node(nid)

    keys = ["apple", "banana", "cherry", "date"]
    for k in keys:
        node = dht.put(k, f"value_{k}")
        print(f"Key '{k}' (id={hash_id(k, 8)}) -> Node {node}")

    for k in keys:
        val = dht.get(k)
        print(f"Get '{k}' = {val}")
```

---

## 6. 다른 나눠진 흩기 표 설계

| 나눠진 흩기 표 | 위상 | 찾기 | 해 |
|---|---|---|---|
| Chord | 고리 + 손가락 | $O(\log n)$ | 2001 |
| Pastry | 앞자락 바탕 길 잡기 | $O(\log n)$ | 2001 |
| Kademlia | 배타 논리합 바탕 나무 | $O(\log n)$ | 2002 |
| CAN | $d$차원 도넛 | $O(d \cdot n^{1/d})$ | 2001 |

!!! tip "실제에서의 Kademlia"
    Kademlia은 가장 널리 펼쳐진 나눠진 흩기 표로 BitTorrent과 IPFS에 쓰인다.
    배타 논리합 바탕 거리 잣대가 자연스럽게 맞섬 찾기를 낳고 길 잡기 표
    건사를 단순하게 한다.

---

## 연습문제

**연습문제 1.**
나눠진 흩기 표를 뜻매김하고 그 핵심 연산을 밝혀라.

??? success "연습문제 1 풀이"
    나눠진 흩기 표는 가운데 일꾼 없이 마디 $n$개에 걸쳐 열쇠를 값에 옮긴다. 마디마다 열쇠 자리의 일부를 맡는다. 연산: PUT(열쇠, 값)은 맡은 마디에 자료를 담고, GET(열쇠)은 맡은 마디로 길을 잡아 자료를 가져온다. 마디마다 칸 $O(\log n)$개의 길 잡기 표를 지닌다. 찾기는 $O(\log n)$번 건너 길을 잡는다. 보기: Chord(고리 위상), Kademlia(배타 논리합 거리), Pastry(앞자락 바탕 길 잡기).

---

**연습문제 2.**
Chord 규약의 고리 얼개와 손가락 표를 밝혀라.

??? success "연습문제 2 풀이"
    Chord은 한결같은 흩기로 열쇠와 마디를 모두 둥근 번호 자리 $[0, 2^m)$의 자리에 옮긴다. 열쇠 $k$은 $\geq k$인 첫 마디(다음 마디)에 담긴다. 마디 $n$마다 칸 $m$개의 손가락 표를 지닌다. finger[$i$] = $(n + 2^{i-1}) \bmod 2^m$의 다음 마디다. 이는 지수로 벌어진 '지름길'을 준다. 열쇠 $k$ 찾기: 바로 앞서는 가장 가까운 손가락에 넘기기를 되풀이한다. 건널 때마다 남은 거리가 반이 되어 $O(\log n)$번 건넌다. 마디가 들어오거나 나갈 때 손가락 표를 고치는 데 쪽지 $O(\log^2 n)$개가 든다.

---

**연습문제 3.**
한결같은 흩기란 무엇이며 나눠진 흩기 표에 왜 중요한가?

??? success "연습문제 3 풀이"
    한결같은 흩기는 열쇠와 마디를 모두 고리에 옮긴다. 마디가 들어오거나 나가면 ($K$이 온 열쇠 수, $n$이 마디 수일 때) 열쇠 $O(K/n)$개만 다시 옮겨지며, 어수룩한 흩기의 $O(K)$과 견주어 적다. 이는 자료 옮김을 가장 작게 한다. 가상 마디(실제 마디 하나가 고리의 여러 자리에 놓임)는 짐 고르기를 좋게 한다. 한결같은 흩기는 내용 배달 그물(Akamai), 나눠진 두름(memcached), 짐 고르개에도 쓰인다.

---

**연습문제 4.**
나눠진 흩기 표는 기계 배움 시스템의 나눠 하는 모델 서비스에 어떻게 쓰이는가?

??? success "연습문제 4 풀이"
    나눠진 흩기 표는 가운데 없는 모델 서비스를 가능하게 한다. (1) 모델 조각내기 --- 큰 모델의 여러 부분을 서로 다른 마디에 담고 헤아림 묻기를 알맞은 조각으로 길잡는다, (2) 특징 곳간 --- 박아 넣기 표(칸이 수십억 개)를 한결같은 흩기로 마디에 흩어 $O(\log n)$번 건너 찾게 한다, (3) 헤아림 결과의 나눠 두르기 --- 자주 묻는 들임을 묻는 곳 가까운 마디에 둔다. 덕분에 어느 한 기계에도 담기지 않는 모델을 서비스할 수 있다.

## 정리하며

이 마당은 핵심 생각、한결같은 흩기、Chord 규약、복잡도을 차례로 짚었다.

**참고 문헌**

- Stoica, I. et al. "Chord: A Scalable Peer-to-Peer Lookup Protocol for
  Internet Applications." IEEE/ACM Transactions on Networking, 2003.
- Peleg, D. *Distributed Computing: A Locality-Sensitive Approach*.
  SIAM, 2000.
