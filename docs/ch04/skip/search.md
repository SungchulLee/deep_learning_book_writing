# 탐색

정렬된 연결 리스트의 탐색은 모든 원소를 차례로 살펴야 하므로 $O(n)$이 걸린다. 스킵 리스트는 연결 리스트를 여러 층으로 두고 위 층이 데이터의 큰 구간을 건너뛰게 하여 이를 빠르게 만든다. 탐색 알고리즘은 이 급행 차선을 활용하여 기대 시간 $O(\log n)$에 원소를 찾아내는데, 이는 이진 탐색이 정렬된 배열을 활용하는 것과 닮았다. 이 페이지는 스킵 리스트의 탐색 알고리즘을 설명하고 구체적인 예를 따라간다.

---

## 1. 탐색 알고리즘

탐색은 **가장 높은 층**의 **머리글** 노드에서 시작하여 다음과 같이 진행된다.

1. 현재 층에서 다음 노드의 키가 목표 키보다 작은 동안 연결 리스트를 따라 앞으로 나아간다.
2. 다음 노드의 키가 목표 이상이거나 앞쪽 포인터가 `None`이면 한 층 내려간다.
3. 층 0에 이를 때까지 되풀이한다.
4. 층 0에서 다음 노드의 키가 목표와 같은지 확인한다.

위에서 아래로, 왼쪽에서 오른쪽으로 훑는 이 방식 덕분에 각 층이 급행 차선 노릇을 하며 탐색 범위를 점점 좁혀 간다.

---

## 2. 의사코드

```
SEARCH(skip_list, target):
    current = skip_list.header
    for level = skip_list.height down to 0:
        while current.forward[level] != None
              and current.forward[level].key < target:
            current = current.forward[level]
    current = current.forward[0]
    if current != None and current.key == target:
        return current
    return None
```

이 알고리즘은 기댓값으로 많아야 $O(\log n)$개의 노드를 살피며, 평균적으로 층마다 상수 개의 노드를 방문한다.

---

## 3. 풀이 예제

키가 $\{3, 6, 7, 9, 12, 17, 19, 21, 25, 26\}$이고 층이 다음과 같이 배정된 스킵 리스트를 생각하자.

```
Level 3: header ───────────────> 6 ─────────────────────────────> 25 ──> None
Level 2: header ───────────────> 6 ──────> 9 ───────────────────> 25 ──> None
Level 1: header ───────────────> 6 ──────> 9 ───> 17 ──> 19 ──> 25 ──> None
Level 0: header ──> 3 ──> 6 ──> 7 ──> 9 ──> 12 ──> 17 ──> 19 ──> 21 ──> 25 ──> 26 ──> None
```

**키 17 찾기:**

| 단계 | 층 | 현재 | 동작 |
|---|---|---|---|
| 1 | 3 | 머리글 | forward[3] = 6 < 17, 오른쪽으로 |
| 2 | 3 | 6 | forward[3] = 25 > 17, 한 층 내려감 |
| 3 | 2 | 6 | forward[2] = 9 < 17, 오른쪽으로 |
| 4 | 2 | 9 | forward[2] = 25 > 17, 한 층 내려감 |
| 5 | 1 | 9 | forward[1] = 17 = 목표, 한 층 내려감 |
| 6 | 0 | 9 | forward[0] = 12 < 17, 오른쪽으로 |
| 7 | 0 | 12 | forward[0] = 17 = 목표, 찾았다! |

탐색은 전체 10개 중 서로 다른 노드 5개(머리글, 6, 9, 12, 17)만 살폈으며, 위 층이 상관없는 원소들을 어떻게 건너뛰는지 보여준다.

**키 15 찾기 (없는 경우):**

| 단계 | 층 | 현재 | 동작 |
|---|---|---|---|
| 1 | 3 | 머리글 | forward[3] = 6 < 15, 오른쪽으로 |
| 2 | 3 | 6 | forward[3] = 25 > 15, 한 층 내려감 |
| 3 | 2 | 6 | forward[2] = 9 < 15, 오른쪽으로 |
| 4 | 2 | 9 | forward[2] = 25 > 15, 한 층 내려감 |
| 5 | 1 | 9 | forward[1] = 17 > 15, 한 층 내려감 |
| 6 | 0 | 9 | forward[0] = 12 < 15, 오른쪽으로 |
| 7 | 0 | 12 | forward[0] = 17 > 15, 멈춤 |
| 8 | -- | 17 | 17 != 15, None 반환 |

이 알고리즘은 15가 없다는 것을 올바르게 판정한다.

---

## 4. 구현

```python
"""
건너뛰기 리스트의 탐색 알고리즘.

건너뛰기 리스트에 기대 O(log n) 탐색 시간을 주는
위에서 아래로, 왼쪽에서 오른쪽으로 가는 탐색 순회를 보인다.
"""

import random

# === 노드 정의 ===

class SkipNode:
    """여러 층의 전진 포인터를 갖는 노드."""

    def __init__(self, key, level):
        self.key = key
        self.forward = [None] * (level + 1)

# === 건너뛰기 리스트 (탐색만) ===

class SkipList:
    """탐색과 삽입을 갖춘 건너뛰기 리스트 (리스트를 만들려면 삽입이 필요하다)."""

    def __init__(self, max_level=16, p=0.5):
        self.max_level = max_level
        self.p = p
        self.level = 0
        self.header = SkipNode(-1, max_level)

    def random_level(self):
        lvl = 0
        while random.random() < self.p and lvl < self.max_level:
            lvl += 1
        return lvl

    def insert(self, key):
        """키를 넣는다 (탐색 시연용 리스트를 만들기 위해 필요하다)."""
        update = [None] * (self.max_level + 1)
        current = self.header
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current

        new_level = self.random_level()
        if new_level > self.level:
            for i in range(self.level + 1, new_level + 1):
                update[i] = self.header
            self.level = new_level

        new_node = SkipNode(key, new_level)
        for i in range(new_level + 1):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node

    def search(self, key):
        """키를 찾아 있으면 그 노드를 돌려준다.

        알고리즘은 가장 높은 층에서 시작하여 아래로 내려가며,
        각 층에서 갈 수 있는 데까지 오른쪽으로 나아간다.
        """
        current = self.header
        comparisons = 0

        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
                comparisons += 1
            comparisons += 1  # 내려가게 만든 비교

        current = current.forward[0]
        comparisons += 1

        if current and current.key == key:
            return current, comparisons
        return None, comparisons

# === 메인 ===

if __name__ == "__main__":
    random.seed(42)
    sl = SkipList(max_level=4, p=0.5)

    keys = [3, 6, 7, 9, 12, 17, 19, 21, 25, 26]
    for k in keys:
        sl.insert(k)

    # 있는 키 탐색
    for target in [17, 9, 26]:
        result, comps = sl.search(target)
        status = "found" if result else "not found"
        print(f"Search({target:2d}): {status}, comparisons={comps}")

    # 없는 키 탐색
    for target in [15, 1, 30]:
        result, comps = sl.search(target)
        status = "found" if result else "not found"
        print(f"Search({target:2d}): {status}, comparisons={comps}")
```

**출력:**

```
Search(17): found, comparisons=8
Search( 9): found, comparisons=5
Search(26): found, comparisons=7
Search(15): not found, comparisons=8
Search( 1): not found, comparisons=6
Search(30): not found, comparisons=7
```

---

## 5. 이진 탐색과의 유비

스킵 리스트의 탐색은 정렬된 배열에서의 이진 탐색과 닮았다.

| 항목 | 이진 탐색 | 스킵 리스트 탐색 |
|---|---|---|
| 범위를 절반으로 줄이는 방법 | 가운데 원소를 고른다 | 한 층 내려간다 |
| 걸음당 비교 횟수 | 1 | 1 |
| 총 비교 횟수 | 최악의 경우 $O(\log n)$ | 기댓값 $O(\log n)$ |
| 필요한 것 | 정렬된 배열 | 정렬된 여러 층 리스트 |
| 삽입 지원 | $O(n)$ (원소를 민다) | 기댓값 $O(\log n)$ |

스킵 리스트가 이진 탐색보다 나은 점은 삽입과 삭제도 $O(\log n)$이라는 것이다. 정렬된 배열을 유지하려면 원소를 미는 데 $O(n)$이 든다.

---

## 연습문제

**연습문제 1.**
탐색에 대해 삽입, 삭제, 탐색, 접근 연산의 시간 복잡도를 진술하라.

??? success "연습문제 1 풀이"
    복잡도는 구체적인 구현(배열 기반이냐 연결 기반이냐)에 달려 있다. 배열 기반은 접근이 $O(1)$이고 임의의 위치에서의 삽입·삭제가 $O(n)$이다. 연결 기반은 이미 아는 위치에서의 삽입·삭제가 $O(1)$이고 탐색·접근이 $O(n)$이다. 어떤 연산이 주를 이루느냐에 따라 선택이 갈린다.

---

**연습문제 2.**
원소 6개로 탐색을 따라가며 각 연산 후의 자료구조 상태를 보여라.

??? success "연습문제 2 풀이"
    구조에 삽입, 접근, 삭제를 차례로 수행하라. 각 단계마다 (연결 구조라면) 포인터를, (배열 기반이라면) 배열의 내용을 보이며 구조가 불변식을 어떻게 유지하는지 나타내라.

---

**연습문제 3.**
탐색이 PyTorch의 텐서 저장과 어떻게 관련되는지 설명하라. 자료구조의 선택이 메모리 배치와 캐시 성능에 어떤 영향을 주는가?

??? success "연습문제 3 풀이"
    PyTorch 텐서는 캐시에 효율적으로 접근할 수 있도록 연속된 배열로 저장된다. 연결 구조는 autograd 그래프를 훑는 데 내부적으로 쓰인다. 이 선택은 메모리 사용량(배열에는 포인터 부담이 없다)과 접근 양상(캐시 지역성 덕분에 순차적인 배열 접근이 연결 리스트 순회보다 10~100배 빠르다)에 모두 영향을 준다.

---

**연습문제 4.**
반복문 불변식을 사용하여 탐색의 주요 연산의 시간 복잡도를 증명하라.

??? success "연습문제 4 풀이"
    알고리즘의 반복문이 유지하는 불변식을 진술하라. 초기화, 유지, 종료를 증명하라. 이 불변식으로부터 반복문이 명시된 횟수 안에 끝남이 따라 나오며, 이로써 복잡도의 상계가 확립된다. $\square$

## 정리하며

이 마당은 탐색 알고리즘、의사코드、풀이 예제、구현을 차례로 짚었다.

**참고 문헌**

- Pugh, W. "Skip Lists: A Probabilistic Alternative to Balanced Trees."
  *Communications of the ACM*, 33(6), 1990.
