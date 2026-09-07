# 스킵 리스트의 구조

정렬된 연결 리스트를 탐색하려면 모든 원소를 훑어야 하므로 탐색이 $O(n)$ 연산이 된다. 균형 이진 탐색 트리(AVL 트리, 적흑 트리)는 $O(\log n)$ 탐색을 이루지만 재균형 논리가 복잡하다. 1990년 윌리엄 퓨가 소개한 **스킵 리스트**는 우아한 대안을 제시한다. 빡빡한 구조적 불변식 대신 무작위성을 써서 탐색, 삽입, 삭제의 기대 시간을 $O(\log n)$으로 만든다. 이 페이지는 스킵 리스트의 구조를 설명하고 그 설계 뒤에 있는 직관을 밝힌다.

## 직관: 급행 차선

원소가 $n$개인 정렬된 연결 리스트를 생각하자. 모든 노드를 살펴야 하므로 탐색에 $O(n)$이 든다. 이제 한 칸 걸러 하나씩만 담은 두 번째 "급행" 연결 리스트를 더한다고 하자. 탐색은 먼저 급행 리스트를 훑어(노드 $n/2$개를 살핀다) 목표 가까이 간 뒤, 마지막 몇 걸음을 위해 전체 리스트로 내려올 수 있다.

네 칸마다 하나씩 담은 세 번째 리스트를 더하면 탐색이 더욱 줄어든다. 층 $i$이 $2^i$번째 원소마다 하나씩 담도록 $\log_2 n$개의 층으로 이 방식을 이어 가면, 탐색이 층마다 한두 개씩 총 $O(\log n)$개의 노드만 방문하는 이상적인 구조가 된다.

스킵 리스트는 이 층 구조를 **확률적으로** 이룬다. 층 $i$에 놓을 원소를 $2^i$번째마다 결정적으로 고르는 대신, 각 노드를 확률 $p$(보통 $p = 1/2$)로 무작위로 위 층에 올린다. 기댓값으로는 이상적인 구조와 같은 기하급수적 간격이 나온다.

## 구성 요소

스킵 리스트는 다음으로 이루어진다.

- **층**: (모든 원소를 담은 바닥 층인) 0부터 위로 번호를 매긴다. 위 층일수록 담는 원소가 점점 적어진다.
- **노드**: 각 노드는 키 하나와, 그 노드가 참여하는 층마다 하나씩인 앞쪽 포인터의 배열을 저장한다.
- **머리글**: 모든 층에 참여하며 모든 탐색의 출발점 노릇을 하는 감시 노드.
- **앞쪽 포인터**: `node.forward[i]`은 층 $i$에서 키가 `node.key`보다 큰 다음 노드를 가리킨다.

## 그림으로 보기

키가 $\{3, 6, 7, 9, 12, 19, 17, 21, 25, 26\}$인 스킵 리스트는 다음과 같은 모습일 수 있다.

```
Level 3: header ──────────────────────────────────────> 19 ──────────────────> None
Level 2: header ───────────> 6 ───────────────────────> 19 ──> 25 ──────────> None
Level 1: header ───────────> 6 ──> 9 ────> 17 ────────> 19 ──> 25 ──────────> None
Level 0: header ──> 3 ──> 6 ──> 7 ──> 9 ──> 12 ──> 17 ──> 19 ──> 21 ──> 25 ──> 26 ──> None
```

층 0은 완전한 정렬 연결 리스트이다. 위 층은 각각 아래 층의 부분집합이며, 점점 성글어지는 급행 차선 노릇을 한다.

## 노드 구조

각 노드는 키 하나와 앞쪽 포인터의 배열을 담는다. 배열의 길이는 그 노드의 층에 1을 더한 값이다(층은 0에서 시작한다).

```python
"""
건너뛰기 리스트의 노드와 기본 구조.

여러 층으로 된 노드 구조와, 층이 정렬된 연결 리스트 위에
급행 차선을 만드는 방식을 보인다.
"""

import random


# === 노드 정의 ===

class SkipNode:
    """건너뛰기 리스트의 노드.

    속성:
        key: 이 노드에 담긴 값.
        forward: 층마다 하나씩인 전진 포인터의 목록.
                 forward[i]는 층 i에서의 다음 노드를 가리킨다.
    """

    def __init__(self, key, level):
        self.key = key
        self.forward = [None] * (level + 1)

    def __repr__(self):
        return f"SkipNode({self.key}, levels={len(self.forward)})"


# === 층 생성 ===

def random_level(max_level, p=0.5):
    """기하 분포로 무작위 층을 만들어 낸다.

    0층 위의 각 층은 확률 p으로 독립적으로 더해진다.
    기대 층수는 1/(1-p)이며, p=0.5이면 2이다.
    """
    level = 0
    while random.random() < p and level < max_level:
        level += 1
    return level


# === 건너뛰기 리스트 만들기 ===

class SkipList:
    """확률적 균형으로 정렬된 차례를 유지하는 건너뛰기 리스트."""

    def __init__(self, max_level=16, p=0.5):
        self.max_level = max_level
        self.p = p
        self.level = 0   # 현재 쓰이는 최대 층
        self.header = SkipNode(-1, max_level)

    def insert(self, key):
        """건너뛰기 리스트에 키를 넣는다."""
        update = [None] * (self.max_level + 1)
        current = self.header
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current

        new_level = random_level(self.max_level, self.p)
        if new_level > self.level:
            for i in range(self.level + 1, new_level + 1):
                update[i] = self.header
            self.level = new_level

        new_node = SkipNode(key, new_level)
        for i in range(new_level + 1):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node

    def display(self):
        """건너뛰기 리스트를 층별로 출력한다."""
        for i in range(self.level, -1, -1):
            nodes = []
            current = self.header.forward[i]
            while current:
                nodes.append(str(current.key))
                current = current.forward[i]
            print(f"Level {i}: header -> {' -> '.join(nodes)} -> None")

    def level_counts(self):
        """각 층의 노드 수를 돌려준다."""
        counts = []
        for i in range(self.level + 1):
            count = 0
            current = self.header.forward[i]
            while current:
                count += 1
                current = current.forward[i]
            counts.append(count)
        return counts


# === 메인 ===

if __name__ == "__main__":
    random.seed(42)
    sl = SkipList(max_level=4, p=0.5)

    keys = [3, 6, 7, 9, 12, 17, 19, 21, 25, 26]
    for k in keys:
        sl.insert(k)

    print("Skip list structure:")
    sl.display()

    counts = sl.level_counts()
    print(f"\nNodes per level: {counts}")
    print(f"Total nodes: {counts[0]}")
    print(f"Height: {sl.level}")
```

**출력:**

```
Skip list structure:
Level 4: header -> 6 -> None
Level 3: header -> 6 -> 25 -> None
Level 2: header -> 6 -> 9 -> 25 -> None
Level 1: header -> 6 -> 9 -> 17 -> 19 -> 25 -> None
Level 0: header -> 3 -> 6 -> 7 -> 9 -> 12 -> 17 -> 19 -> 21 -> 25 -> 26 -> None

Nodes per level: [10, 5, 3, 2, 1]
Total nodes: 10
Height: 4
```

## 성질

잘 만들어진 스킵 리스트는 몇 가지 핵심 성질을 보인다.

1. **층 0은 완전하다**: 모든 원소가 층 0에 나타나 보통의 정렬 연결 리스트를 이룬다.

2. **층의 포함 관계**: 어떤 노드가 층 $i$에 나타나면 층 $0, 1, \ldots, i-1$에도 모두 나타난다.

3. **기하급수적으로 성글어짐**: 층 $i$의 노드 개수의 기댓값은 $n \cdot p^i$이므로, 각 층은 위 층의 대략 $1/p$배만큼 노드를 담는다.

4. **로그 높이**: 높이(최대 층)의 기댓값은 $O(\log_{1/p} n)$이다. 유도는 [기대 높이](height.md)를 보라.

5. **선형 공간**: 모든 노드에 걸친 앞쪽 포인터 개수의 기댓값은 $n / (1 - p) = O(n)$이다.

## 이상적인 스킵 리스트와의 비교

| 특징 | 이상적 (결정적) | 무작위 |
|---|---|---|
| 층 배정 | 층 $i$에 $2^i$번째 원소마다 | 확률 $p^i$으로 무작위 |
| 탐색 보장 | 최악의 경우 $O(\log n)$ | 기댓값 $O(\log n)$ |
| 삽입 | $O(n)$ (층을 다시 짜야 한다) | 기댓값 $O(\log n)$ |
| 삭제 | $O(n)$ (층을 다시 짜야 한다) | 기댓값 $O(\log n)$ |

이상적인 스킵 리스트는 완벽하게 균형 잡혀 있지만, 삽입이나 삭제를 할 때마다 층에 걸쳐 원소의 자리를 다시 잡아야 해서 비현실적이다. 무작위화는 최악의 경우 보장을 내주는 대신 간단하고 효율적인 갱신을 얻는다.

## 참고 문헌

- Pugh, W. "Skip Lists: A Probabilistic Alternative to Balanced Trees."
  *Communications of the ACM*, 33(6), 1990.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C.
  *Introduction to Algorithms* (4th ed.). MIT Press.


## 연습문제

**연습문제 1.**
스킵 리스트의 구조에 대해 삽입, 삭제, 탐색, 접근 연산의 시간 복잡도를 진술하라.

??? success "연습문제 1 풀이"
    복잡도는 구체적인 구현(배열 기반이냐 연결 기반이냐)에 달려 있다. 배열 기반은 접근이 $O(1)$이고 임의의 위치에서의 삽입·삭제가 $O(n)$이다. 연결 기반은 이미 아는 위치에서의 삽입·삭제가 $O(1)$이고 탐색·접근이 $O(n)$이다. 어떤 연산이 주를 이루느냐에 따라 선택이 갈린다.

---

**연습문제 2.**
원소 6개로 스킵 리스트의 구조을(를) 따라가며 각 연산 후의 자료구조 상태를 보여라.

??? success "연습문제 2 풀이"
    구조에 삽입, 접근, 삭제를 차례로 수행하라. 각 단계마다 (연결 구조라면) 포인터를, (배열 기반이라면) 배열의 내용을 보이며 구조가 불변식을 어떻게 유지하는지 나타내라.

---

**연습문제 3.**
스킵 리스트의 구조이(가) PyTorch의 텐서 저장과 어떻게 관련되는지 설명하라. 자료구조의 선택이 메모리 배치와 캐시 성능에 어떤 영향을 주는가?

??? success "연습문제 3 풀이"
    PyTorch 텐서는 캐시에 효율적으로 접근할 수 있도록 연속된 배열로 저장된다. 연결 구조는 autograd 그래프를 훑는 데 내부적으로 쓰인다. 이 선택은 메모리 사용량(배열에는 포인터 부담이 없다)과 접근 양상(캐시 지역성 덕분에 순차적인 배열 접근이 연결 리스트 순회보다 10~100배 빠르다)에 모두 영향을 준다.

---

**연습문제 4.**
반복문 불변식을 사용하여 스킵 리스트의 구조의 주요 연산의 시간 복잡도를 증명하라.

??? success "연습문제 4 풀이"
    알고리즘의 반복문이 유지하는 불변식을 진술하라. 초기화, 유지, 종료를 증명하라. 이 불변식으로부터 반복문이 명시된 횟수 안에 끝남이 따라 나오며, 이로써 복잡도의 상계가 확립된다. $\square$