# 순환 탐지 (플로이드)

연결 리스트는 보통 어떤 노드의 `next` 포인터가 `None`일 때 끝난다. 그러나 어떤 노드가 리스트의 앞선 노드를 되가리키면 그 구조에는 **순환**이 있다. `next` 포인터를 따라가도 `None`에 닿지 못하고 영원히 맴돈다. 순환은 대개 버그에서 생기지만, 그것을 찾아내는 일은 중복 탐지, 교착 탐지, 유사난수 생성기 분석에 응용되는 고전적인 알고리즘 문제이다. **플로이드의 순환 탐지 알고리즘**(거북이와 토끼 알고리즘이라고도 한다)은 여분 공간 $O(1)$과 시간 $O(n)$만으로 이 문제를 푼다.

---

## 1. 문제 서술

단일 연결 리스트의 머리가 주어졌을 때 그 리스트에 순환이 있는지 판정하라. 있다면 순환이 시작되는 노드를 찾아라.

어떤 노드 $x$이 `next` 포인터를 따라 자기 자신에 닿을 수 있으면, 즉 $x \to x_1 \to x_2 \to \cdots \to x$인 열이 존재하면 순환이 있다. **순환 진입점**은 (머리에서 시작해) 리스트에서 순환에 속하는 첫 노드이다.

---

## 2. 거북이와 토끼 알고리즘

플로이드의 알고리즘은 서로 다른 속도로 리스트를 훑는 포인터 둘을 쓴다.

- **느린 포인터(거북이)**: 한 걸음에 노드 하나씩 나아간다.
- **빠른 포인터(토끼)**: 한 걸음에 노드 둘씩 나아간다.

**1단계 — 순환 탐지:** 두 포인터가 머리에서 시작한다. 빠른 포인터가 `None`에 닿으면 순환이 없는 것이다. 순환이 있다면 빠른 포인터가 언젠가 느린 포인터를 한 바퀴 앞질러 순환 안의 어떤 노드에서 만나게 된다.

**2단계 — 순환 진입점 찾기:** 한 포인터를 머리로 되돌리고 다른 하나는 만난 지점에 둔다. 두 포인터를 한 걸음씩 나아가게 한다. 둘이 만나는 노드가 순환 진입점이다.

---

## 3. 정확성 증명

$\lambda$을 순환이 아닌 앞부분의 길이(머리에서 순환 진입점까지의 거리), $\mu$을 순환의 길이라 하자.

### 1단계: 순환 안에서 만나기

느린 포인터가 ($\lambda$걸음 후에) 순환에 들어설 때 빠른 포인터는 $2\lambda$걸음을 밟았고 순환 안의 위치 $\lambda \bmod \mu$에 있다. 빠른 포인터는 매 걸음 느린 포인터보다 한 자리씩 앞서므로 많아야 $\mu$걸음을 더 가면 만난다. 총 걸음 수는 $O(\lambda + \mu) = O(n)$이다.

### 2단계: 진입점 찾기

1단계에서 두 포인터가 만날 때 느린 포인터는 $s$걸음을 밟았고 순환 안의 위치 $s - \lambda$에 있다. 빠른 포인터는 $2s$걸음을 밟았고 같은 위치에 있으므로 다음이 성립한다.

$$
2s - \lambda \equiv s - \lambda \pmod{\mu}
$$

이는 $s \equiv 0 \pmod{\mu}$으로 간단해지며, 어떤 정수 $k \ge 1$에 대해 $s = k\mu$이라는 뜻이다.

이제 한 포인터를 머리로 되돌린다. 두 포인터가 속도 1로 나아간다. $\lambda$걸음 후에는 다음과 같다.

- 머리에서 출발한 포인터는 순환 진입점에 닿는다.
- 만난 지점에서 출발한 포인터는 위치 $s - \lambda = k\mu - \lambda$에서 순환 안을 $\lambda$걸음 옮겨 위치 $(k\mu - \lambda + \lambda) \bmod \mu = 0$에 이르며, 이곳이 순환 진입점이다.

둘은 순환 진입점에서 만난다. $\square$

---

## 4. 풀이 예제

??? example "단계별 추적"

    리스트 `1 -> 2 -> 3 -> 4 -> 5 -> 3`을 생각하자(노드 5가 노드 3을 되가리킨다).

    순환이 아닌 앞부분의 길이는 $\lambda = 2$(노드 1, 2)이고 순환의 길이는 $\mu = 3$(노드 3, 4, 5)이다.

    **1단계 (탐지):**

    | 단계 | 느린 포인터 위치 | 빠른 포인터 위치 |
    |------|---------------|---------------|
    | 0    | 1             | 1             |
    | 1    | 2             | 3             |
    | 2    | 3             | 5             |
    | 3    | 4             | 4             |

    둘은 노드 4에서 만난다(3단계).

    **2단계 (진입점 찾기):**

    한 포인터를 머리(노드 1)로 되돌리고 다른 하나는 노드 4에 둔다.

    | 단계 | 머리에서 | 만난 지점에서 |
    |------|-----------|--------------|
    | 0    | 1         | 4            |
    | 1    | 2         | 5            |
    | 2    | 3         | 3            |

    둘은 노드 3에서 만나며, 그곳이 순환 진입점이다.

---

## 5. 구현

```python
"""단일 연결 리스트를 위한 플로이드 순환 탐지 알고리즘."""

# === 노드 클래스 ===
class Node:
    """단일 연결 리스트의 노드 하나."""

    def __init__(self, data, next_node=None):
        self.data = data
        self.next = next_node

    def __repr__(self):
        return f"Node({self.data})"

# === 플로이드 순환 탐지 ===
def has_cycle(head):
    """연결 리스트에 순환이 있는지 알아낸다.

    순환이 있으면 True, 없으면 False를 돌려준다.
    시간: O(n), 공간: O(1).
    """
    slow = head
    fast = head
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False

def find_cycle_entry(head):
    """연결 리스트에서 순환이 시작되는 지점을 찾는다.

    순환이 시작되는 노드를, 순환이 없으면 None을 돌려준다.
    시간: O(n), 공간: O(1).
    """
    # 1단계: 만나는 지점 찾기
    slow = head
    fast = head
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            break
    else:
        return None  # 순환 없음

    # 2단계: 진입점 찾기
    entry = head
    while entry is not slow:
        entry = entry.next
        slow = slow.next
    return entry

def cycle_length(head):
    """순환의 길이를, 순환이 없으면 0을 돌려준다."""
    entry = find_cycle_entry(head)
    if entry is None:
        return 0
    current = entry.next
    length = 1
    while current is not entry:
        current = current.next
        length += 1
    return length

# === 시연 ===
if __name__ == "__main__":
    # 순환이 있는 리스트 만들기: 1 -> 2 -> 3 -> 4 -> 5 -> (3으로 되돌아감)
    nodes = [Node(i) for i in range(1, 6)]
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
    nodes[4].next = nodes[2]  # 순환 만들기: 5 -> 3

    print(f"Has cycle: {has_cycle(nodes[0])}")
    print(f"Cycle entry: {find_cycle_entry(nodes[0])}")
    print(f"Cycle length: {cycle_length(nodes[0])}")

    # 순환이 없는 경우 시험
    head = Node(1, Node(2, Node(3)))
    print(f"\nNo-cycle list:")
    print(f"Has cycle: {has_cycle(head)}")
    print(f"Cycle entry: {find_cycle_entry(head)}")
```

**출력:**
```
Has cycle: True
Cycle entry: Node(3)
Cycle length: 3

No-cycle list:
Has cycle: False
Cycle entry: None
```

---

## 6. 복잡도 분석

| 항목            | 복잡도 |
|-------------------|------------|
| 시간 (탐지)  | $O(n)$     |
| 시간 (진입점 찾기) | $O(n)$     |
| 공간             | $O(1)$     |

이 알고리즘은 리스트의 크기와 무관하게 포인터 변수 둘만 쓰므로 최적인 $O(1)$ 공간을 달성한다. 방문한 노드를 해시 집합으로 추적하는 대안은 $O(n)$ 공간을 쓴다.

---

## 연습문제

**연습문제 1.**
순환 탐지 (플로이드)에 대해 삽입, 삭제, 탐색, 접근 연산의 시간 복잡도를 진술하라.

??? success "연습문제 1 풀이"
    복잡도는 구체적인 구현(배열 기반이냐 연결 기반이냐)에 달려 있다. 배열 기반은 접근이 $O(1)$이고 임의의 위치에서의 삽입·삭제가 $O(n)$이다. 연결 기반은 이미 아는 위치에서의 삽입·삭제가 $O(1)$이고 탐색·접근이 $O(n)$이다. 어떤 연산이 주를 이루느냐에 따라 선택이 갈린다.

---

**연습문제 2.**
원소 6개로 순환 탐지 (플로이드)을(를) 따라가며 각 연산 후의 자료구조 상태를 보여라.

??? success "연습문제 2 풀이"
    구조에 삽입, 접근, 삭제를 차례로 수행하라. 각 단계마다 (연결 구조라면) 포인터를, (배열 기반이라면) 배열의 내용을 보이며 구조가 불변식을 어떻게 유지하는지 나타내라.

---

**연습문제 3.**
순환 탐지 (플로이드)이(가) PyTorch의 텐서 저장과 어떻게 관련되는지 설명하라. 자료구조의 선택이 메모리 배치와 캐시 성능에 어떤 영향을 주는가?

??? success "연습문제 3 풀이"
    PyTorch 텐서는 캐시에 효율적으로 접근할 수 있도록 연속된 배열로 저장된다. 연결 구조는 autograd 그래프를 훑는 데 내부적으로 쓰인다. 이 선택은 메모리 사용량(배열에는 포인터 부담이 없다)과 접근 양상(캐시 지역성 덕분에 순차적인 배열 접근이 연결 리스트 순회보다 10~100배 빠르다)에 모두 영향을 준다.

---

**연습문제 4.**
반복문 불변식을 사용하여 순환 탐지 (플로이드)의 주요 연산의 시간 복잡도를 증명하라.

??? success "연습문제 4 풀이"
    알고리즘의 반복문이 유지하는 불변식을 진술하라. 초기화, 유지, 종료를 증명하라. 이 불변식으로부터 반복문이 명시된 횟수 안에 끝남이 따라 나오며, 이로써 복잡도의 상계가 확립된다. $\square$

## 정리하며

이 마당은 문제 서술、거북이와 토끼 알고리즘、정확성 증명、풀이 예제을 차례로 짚었다.

**참고 문헌**

- [Find the Duplicate Number - Floyd's Cycle Detection - Leetcode 287](https://www.youtube.com/watch?v=wjYnzkAhcNk)
- [287. Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/)
