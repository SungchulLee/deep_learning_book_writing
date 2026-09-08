# 나눠 하는 너비 우선 찾기

나눠진 그물에서 마디마다 하나의 셈틀이며 쪽지 건네기로 바로 이웃하고만
말을 주고받는다. 어느 셈틀도 그래프를 온 자리에서 보지 못한다.
나눠 하는 너비 우선 찾기는 정해진 샘에서 너비 우선 뻗은 나무를 지어
최단 길 잡기를 할 수 있게 하고 다른 많은 나눠 하는 알고리즘의
등뼈가 된다.

---

## 1. 모형

**맞춘 쪽지 건네기 모형**을 여긴다:

- 그물은 마디 $n = |V|$개와 변 $m = |E|$개를 가진 방향 없는 이어진 그래프
  $G = (V, E)$이다.
- 마디마다 하나뿐인 번호를 가지며 자기 이웃만 안다.
- 셈은 맞춘 **바퀴**로 나아간다. 바퀴마다 마디마다 이웃에 쪽지를 보내고
  쪽지를 받고 그 자리에서 셈을
  할 수 있다.
- 정해진 **샘** 마디 $s$이 너비 우선 찾기를 시작한다.

---

## 2. 알고리즘(맞춘 판)

이 알고리즘은 차례대로 하는 너비 우선 찾기를 그대로 따르되 바퀴마다 한 켜씩 나아간다.

**첫 값 두기.** 샘 $s$은 거리를 $d_s = 0$으로, 어버이를
$\text{parent}_s = \text{nil}$으로 둔다.

**바퀴 $i$($i = 0, 1, 2, \dots$에 대해$)$:**

1. 앞 바퀴에 $d_v = i$으로 둔 마디 $v$마다 모든 이웃에
   `SEARCH` 쪽지를 보낸다.
2. 아직 들르지 않았는데 `SEARCH` 쪽지를 받은 마디 $u$마다 (처음 받은
   `SEARCH`에 대해) $d_u = i + 1$과 $\text{parent}_u = v$으로 둔다.
   그러고서 자신을 들렀다고 표시한다.

새로 들르는 마디가 없는 바퀴가 나오면 알고리즘이 멈춘다.

---

## 3. 복잡도

| 지표 | 한계 |
|---|---|
| 바퀴(때) | $D$이 $G$의 지름일 때 $O(D)$ |
| 쪽지 | 모든 바퀴를 통틀어 $O(m)$ |
| 마디마다 공간 | $O(\deg(v))$ |

변마다 많아야 쪽지 둘을 나르므로(양 끝에서 하나씩)
쪽지 가둠이 $O(m)$이 된다.

---

## 4. 맞추지 않은 판

맞추지 않은 그물에서는 쪽지가 (유한하지만) 아무만큼 늦어질 수 있다.
마디가 바퀴가 언제 끝나는지 알 수 없으므로 맞춘 알고리즘을 그대로
쓸 수 없다.

Awerbuch의 **맞추지 않은 너비 우선 찾기** 알고리즘은 받음 알림에 바탕한
길을 쓴다:

1. 샘이 `SEARCH` 쪽지를 보낸다.
2. 첫 `SEARCH`을 받은 마디 $u$은 어버이를 정하고 모든 이웃에 `SEARCH`을
   넘긴 뒤 받음 알림을 기다린다.
3. 받음 알림이 샘으로 되돌아가고 샘이 다음 켜를
   시작한다.

이는 (가장 큰 쪽지 늦음을 낱덩이로 잰) 때 복잡도 $O(D)$과 가장 나쁜 경우
쪽지 $O(m + n D)$개를 이룬다.

!!! note "켜 짓기와 물 대기"
    단순한 물 대기 방식은 샘에서 널리 알린다. 마디마다 처음 받은 쪽지를
    넘긴다. 이는 $O(D)$ 때에 쪽지 $O(m)$개로 너비 우선 나무를 찾지만
    맞춘 모형에서만 그렇다.
    맞추지 않은 물 대기는 올바른 너비 우선 나무를 내지 못할 수 있다(모든 이음의
    늦음이 같을 때만 최단 길 나무가 된다).

---

## 5. 흉내 내기

```python
"""
맞춘 나눠 하는 너비 우선 찾기 흉내내기.

마디마다 그 자리 상태를 가진 셈틀로 나타낸다.
쪽지는 맞춘 바퀴마다 오간다.
"""

# === 마디 셈틀 ===
class Node:
    """나눠진 그물의 셈틀을 나타낸다."""

    def __init__(self, node_id: int, neighbors: list[int]):
        self.node_id = node_id
        self.neighbors = neighbors
        self.distance = -1
        self.parent = -1
        self.visited = False
        self.inbox: list[tuple[int, int]] = []  # (보낸 이, 거리)

    def receive(self, sender: int, dist: int) -> None:
        """SEARCH 쪽지를 받는다."""
        self.inbox.append((sender, dist))

    def process_round(self) -> list[tuple[int, int]]:
        """쪽지를 다루고 내보낼 (이웃, 거리) 짝을 돌려준다."""
        outgoing = []
        if not self.visited and self.inbox:
            sender, dist = self.inbox[0]
            self.distance = dist + 1
            self.parent = sender
            self.visited = True
            for nb in self.neighbors:
                outgoing.append((nb, self.distance))
        self.inbox.clear()
        return outgoing

# === 나눠 하는 너비 우선 찾기 흉내내개 ===
def distributed_bfs(adj: dict[int, list[int]], source: int) -> dict[int, int]:
    """발맞춘 흩뿌린 너비 우선 찾기를 흉내 낸다. {마디: 거리}를 돌려준다."""
    nodes = {v: Node(v, neighbors) for v, neighbors in adj.items()}

    # 샘 첫 값 두기
    nodes[source].distance = 0
    nodes[source].visited = True

    # 0바퀴: 샘이 이웃에 보낸다
    messages = [(nb, source, 0) for nb in adj[source]]

    round_num = 0
    while messages:
        round_num += 1
        # 쪽지 배달
        for dest, sender, dist in messages:
            nodes[dest].receive(sender, dist)

        # 바퀴를 다루고 새 쪽지를 모은다
        new_messages = []
        for node in nodes.values():
            outgoing = node.process_round()
            for nb, dist in outgoing:
                new_messages.append((nb, node.node_id, dist))
        messages = new_messages

    return {v: node.distance for v, node in nodes.items()}

# === 보기 ===
if __name__ == "__main__":
    graph = {
        0: [1, 2],
        1: [0, 3],
        2: [0, 3, 4],
        3: [1, 2],
        4: [2],
    }
    distances = distributed_bfs(graph, source=0)
    print("Distributed BFS distances from node 0:")
    for node, dist in sorted(distances.items()):
        print(f"  Node {node}: distance {dist}")
```

**출력:**

```
Distributed BFS distances from node 0:
  Node 0: distance 0
  Node 1: distance 1
  Node 2: distance 1
  Node 3: distance 2
  Node 4: distance 2
```

---

## 6. 응용

- **길 잡기 표.** 너비 우선 나무는 무게 없는 그물에서 최단 길 잡기를
  준다.
- **위상 알아내기.** 나눠 하는 너비 우선 찾기는 가운데 앎 없이도 그물
  얼개를 드러낸다.
- **쌓기 벽돌.** 많은 나눠 하는 알고리즘(우두머리 뽑기, 뻗은 나무 짓기)이
  너비 우선 찾기를 부속 절차로 쓴다.

---

## 연습문제

**연습문제 1.**
나눠 하는 너비 우선 찾기 알고리즘과 그 쪽지 복잡도를 밝혀라.

??? success "연습문제 1 풀이"
    나눠 하는 너비 우선 찾기에서 정해진 뿌리가 너비 우선 나무 짓기를 시작한다. 마디마다 뿌리에서의 거리를 지닌다. 뿌리가 모든 이웃에 쪽지를 보낸다. 마디가 너비 우선 쪽지를 처음 받으면 보낸 이를 어버이로 삼고 거리를 적은 뒤 다른 모든 이웃에 넘긴다. 겹치는 쪽지(이미 들른 마디에서 온 것)는 버린다. 쪽지 복잡도: 변마다 많아야 쪽지 2개(방향마다 하나)를 나르므로 $O(|E|)$이다. 바퀴 복잡도: $D$이 지름일 때 $O(D)$이다.

---

**연습문제 2.**
주고받기 바퀴로 볼 때 나눠 하는 너비 우선 찾기의 때 복잡도는 얼마인가?

??? success "연습문제 2 풀이"
    나눠 하는 너비 우선 찾기는 $D$이 그래프 지름일 때 맞춘 바퀴 $O(D)$번에 끝난다. 바퀴마다 지금 너비 우선 앞자락의 모든 마디가 이웃에 쪽지를 보낸다. 앞자락이 바퀴마다 한 켜씩 나아간다. $D$바퀴 뒤에는 모든 마디를 찾아낸다. 이는 가장 좋은 값이다. 뿌리의 앎이 가장 먼 마디에 닿으려면 적어도 $D$번 건너야 하므로 어떤 알고리즘도 $\Omega(D)$바퀴가 든다.

---

**연습문제 3.**
나눠 하는 시스템에서 맞추지 않은 너비 우선 찾기는 맞춘 것과 어떻게 다른가?

??? success "연습문제 3 풀이"
    맞춘 너비 우선 찾기에서는 모든 마디가 발맞추어 바퀴를 돈다. 맞추지 않은 판에서는 쪽지가 아무만큼 늦는다. 어떤 마디는 더 먼 길로 어버이를 이미 정한 뒤에 '더 짧은 거리' 쪽지를 받을 수 있다. 맞추지 않은 너비 우선 찾기는 그런 고침을 다루어야 한다. 지금 어림보다 짧은 거리를 받으면 어버이를 고치고 다시 알린다. 가장 나쁜 경우 쪽지가 $O(n \cdot |E|)$개가 될 수 있다. 풀이: 맞춤개(알파, 베타, 감마)로 맞추지 않은 그물 위에서 맞춘 돌림을 흉내 낸다.

---

**연습문제 4.**
나눠 하는 너비 우선 찾기는 동무끼리 그물의 길 잡기에 어떻게 쓰이는가?

??? success "연습문제 4 풀이"
    동무끼리 그물에서 너비 우선 찾기는 동무 사이의 가장 적게 건너는 길을 찾는다. 마디가 들어오면 가까운 동무를 찾으려 (수명에 가둬진) 좁은 너비 우선 찾기를 한다. 짜임 없는 동무끼리 그물(Gnutella)에서 묻기 물 대기는 사실상 너비 우선 찾기다. 짜임 있는 동무끼리 그물(Chord, Kademlia)에서는 덧씌운 위상을 설계해 너비 우선 같은 길 잡기가 $O(\log n)$번 건너기로 끝난다. 나눠 하는 너비 우선 찾기는 그물 위상 알아내기에도 쓰이며 마디마다의 너비 우선 나무가 그물 얼개를 그 자리에서 보여 준다.

## 정리하며

이 마당은 모형、알고리즘(맞춘 판)、복잡도、맞추지 않은 판을 차례로 짚었다.

**참고 문헌**

- Lynch, N. *Distributed Algorithms*. Morgan Kaufmann, 1996.
- Peleg, D. *Distributed Computing: A Locality-Sensitive Approach*.
  SIAM, 2000.
