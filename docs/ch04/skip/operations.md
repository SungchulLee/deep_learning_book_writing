# 삽입과 삭제

스킵 리스트의 탐색은 구조를 바꾸지 않고 원소를 찾는다. 반면 삽입과 삭제는 여러 층에 걸쳐 노드를 넣거나 빼면서 스킵 리스트의 불변식을 지켜야 한다. 두 연산 모두 탐색 절차 위에 세워진다. 같은 하향식 순회로 먼저 올바른 자리를 찾고, 영향을 받는 각 층에서 국소적으로 포인터를 고친다. 이 페이지는 두 연산을 완전한 구현과 함께 자세히 설명한다.

---

## 1. 삽입 알고리즘

스킵 리스트에 키 $k$을 삽입하려면 세 단계가 필요하다.

1. **자리 찾기**: 보통의 탐색처럼 스킵 리스트를 훑되, 각 층에서 내려가기 직전에 방문한 마지막 노드를 기록한다. 이 노드들이 **갱신 배열**을 이루며, 포인터를 돌려놓아야 할 각 층의 앞 노드들이다.

2. **무작위 층 정하기**: 확률 $p$인 치우친 동전을 되풀이해 던져 새 노드의 층을 정한다. 층 1에서 시작하여 앞면이 나오는 동안 위로 올린다.

3. **각 층에 끼워 넣기**: 층 1부터 새 노드의 층까지, 해당 갱신 노드 뒤에 새 노드를 넣도록 포인터를 돌려놓는다.

새 노드의 층이 현재 스킵 리스트의 높이를 넘으면, 머리글에 새 층을 늘려 새 노드를 곧바로 가리키게 한다.

---

## 2. 삭제 알고리즘

키 $k$을 삭제하는 것도 비슷한 형태를 따른다.

1. **자리 찾기**: 삽입 때처럼 갱신 배열을 기록하며 스킵 리스트를 훑는다.

2. **존재 확인**: 목표 노드가 실제로 키 $k$을 담고 있는지 확인한다. 아니라면 그 키는 리스트에 없다.

3. **각 층에서 떼어내기**: 목표 노드가 나타나는 각 층에서 앞 노드의 앞쪽 포인터가 목표 노드를 건너뛰도록 돌려놓는다.

4. **높이 줄이기**: 삭제로 인해 위쪽 층들이 비면(머리글이 `None`을 가리키면) 스킵 리스트의 높이를 줄인다.

---

## 3. 구현

```python
"""
건너뛰기 리스트의 삽입과 삭제 연산.

두 연산 모두 갱신 배열을 쓰는 탐색 양식을 따른다.
위치를 찾은 뒤 각 층에서 이어 넣거나 끊는다.
"""

import random

# === 노드 정의 ===

class SkipNode:
    """여러 층의 전진 포인터를 갖는 건너뛰기 리스트의 노드."""

    def __init__(self, key, level):
        self.key = key
        self.forward = [None] * (level + 1)  # forward[i] = 층 i에서의 다음 노드

# === 건너뛰기 리스트 ===

class SkipList:
    """탐색, 삽입, 삭제를 지원하는 건너뛰기 리스트."""

    def __init__(self, max_level=16, p=0.5):
        self.max_level = max_level
        self.p = p
        self.level = 0   # 현재 쓰이는 가장 높은 층
        self.header = SkipNode(-1, max_level)  # 보초 머리

    def random_level(self):
        """새 노드를 위한 층을 무작위로 정한다."""
        lvl = 0
        while random.random() < self.p and lvl < self.max_level:
            lvl += 1
        return lvl

    def search(self, key):
        """키를 찾고, 찾으면 그 노드를 돌려준다."""
        current = self.header
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
        current = current.forward[0]
        if current and current.key == key:
            return current
        return None

    def insert(self, key):
        """건너뛰기 리스트에 키를 넣는다.

        새로 만든 노드를 돌려준다.
        """
        # 1단계: 갱신 배열 찾기 (각 층에서의 선행 노드)
        update = [None] * (self.max_level + 1)
        current = self.header
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current

        current = current.forward[0]

        # 키가 이미 있으면 중복해서 넣지 않는다
        if current and current.key == key:
            return current

        # 2단계: 무작위 층 생성
        new_level = self.random_level()

        # 새 층이 현재 높이를 넘으면 갱신 배열을 늘린다
        if new_level > self.level:
            for i in range(self.level + 1, new_level + 1):
                update[i] = self.header
            self.level = new_level

        # 3단계: 노드를 만들어 각 층에 이어 넣기
        new_node = SkipNode(key, new_level)
        for i in range(new_level + 1):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node

        return new_node

    def delete(self, key):
        """건너뛰기 리스트에서 키를 지운다.

        키를 찾아 지웠으면 True, 아니면 False를 돌려준다.
        """
        # 1단계: 갱신 배열 찾기
        update = [None] * (self.max_level + 1)
        current = self.header
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current

        target = current.forward[0]

        # 2단계: 존재 여부 확인
        if not target or target.key != key:
            return False

        # 3단계: 각 층에서 연결 끊기
        for i in range(self.level + 1):
            if update[i].forward[i] is not target:
                break
            update[i].forward[i] = target.forward[i]

        # 4단계: 필요하면 높이 줄이기
        while self.level > 0 and self.header.forward[self.level] is None:
            self.level -= 1

        return True

    def to_list(self):
        """모든 키를 정렬된 차례로 돌려준다 (0층 순회)."""
        result = []
        current = self.header.forward[0]
        while current:
            result.append(current.key)
            current = current.forward[0]
        return result

    def display(self):
        """건너뛰기 리스트를 층별로 출력한다."""
        for i in range(self.level, -1, -1):
            nodes = []
            current = self.header.forward[i]
            while current:
                nodes.append(str(current.key))
                current = current.forward[i]
            print(f"Level {i}: {' -> '.join(nodes)}")

# === 메인 ===

if __name__ == "__main__":
    random.seed(42)
    sl = SkipList(max_level=4, p=0.5)

    # 원소 삽입
    for key in [3, 6, 7, 9, 12, 19, 17, 26, 21, 25]:
        sl.insert(key)

    print("After insertions:")
    sl.display()
    print("Sorted:", sl.to_list())

    # 원소 삭제
    sl.delete(19)
    sl.delete(3)
    print("\nAfter deleting 19 and 3:")
    sl.display()
    print("Sorted:", sl.to_list())

    # 탐색
    found = sl.search(12)
    print(f"\nSearch 12: {'found' if found else 'not found'}")
    found = sl.search(19)
    print(f"Search 19: {'found' if found else 'not found'}")
```

**출력:**

```
After insertions:
Level 4: 6
Level 3: 6 -> 25
Level 2: 6 -> 9 -> 25
Level 1: 6 -> 9 -> 17 -> 19 -> 25
Level 0: 3 -> 6 -> 7 -> 9 -> 12 -> 17 -> 19 -> 21 -> 25 -> 26
Sorted: [3, 6, 7, 9, 12, 17, 19, 21, 25, 26]

After deleting 19 and 3:
Level 4: 6
Level 3: 6 -> 25
Level 2: 6 -> 9 -> 25
Level 1: 6 -> 9 -> 17 -> 25
Level 0: 6 -> 7 -> 9 -> 12 -> 17 -> 21 -> 25 -> 26
Sorted: [6, 7, 9, 12, 17, 21, 25, 26]

Search 12: found
Search 19: not found
```

---

## 4. 갱신 배열

갱신 배열은 삽입과 삭제 모두에서 중심이 되는 장부이다. 각 층 $i$에 대해 `update[i]`은 층 $i$에서 키가 목표 키보다 작은 가장 오른쪽 노드를 담는다. 이 노드가 층 $i$에서 삽입 또는 삭제 지점의 앞 노드이다.

갱신 배열을 만드는 비용은 탐색과 같은 기대 시간 $O(\log n)$이다. 각 층에서 끼워 넣거나 떼어내는 단계는 층당 $O(1)$이고, 한 노드가 걸치는 층의 수는 기댓값으로 $O(\log n)$이다.

---

## 5. 복잡도

| 연산 | 기대 시간 | 최악의 경우 시간 |
|---|---|---|
| 삽입 | $O(\log n)$ | $O(n)$ |
| 삭제 | $O(\log n)$ | $O(n)$ |

두 연산 모두 탐색 단계가 비용을 지배한다. 포인터 고치기 자체는 영향을 받는 노드의 층을 $\ell$이라 할 때 $O(\ell)$이며, 기댓값으로는 $O(1)$이다.

최악의 경우인 $O(n)$은 모든 노드가 우연히 층 1에만 있을 때에만 일어나며($n$이 커질수록 그 확률이 지수적으로 줄어든다), 그때 스킵 리스트는 평범한 정렬 연결 리스트로 주저앉는다.

---

## 연습문제

**연습문제 1.**
삽입과 삭제에 대해 삽입, 삭제, 탐색, 접근 연산의 시간 복잡도를 진술하라.

??? success "연습문제 1 풀이"
    복잡도는 구체적인 구현(배열 기반이냐 연결 기반이냐)에 달려 있다. 배열 기반은 접근이 $O(1)$이고 임의의 위치에서의 삽입·삭제가 $O(n)$이다. 연결 기반은 이미 아는 위치에서의 삽입·삭제가 $O(1)$이고 탐색·접근이 $O(n)$이다. 어떤 연산이 주를 이루느냐에 따라 선택이 갈린다.

---

**연습문제 2.**
원소 6개로 삽입과 삭제을(를) 따라가며 각 연산 후의 자료구조 상태를 보여라.

??? success "연습문제 2 풀이"
    구조에 삽입, 접근, 삭제를 차례로 수행하라. 각 단계마다 (연결 구조라면) 포인터를, (배열 기반이라면) 배열의 내용을 보이며 구조가 불변식을 어떻게 유지하는지 나타내라.

---

**연습문제 3.**
삽입과 삭제이(가) PyTorch의 텐서 저장과 어떻게 관련되는지 설명하라. 자료구조의 선택이 메모리 배치와 캐시 성능에 어떤 영향을 주는가?

??? success "연습문제 3 풀이"
    PyTorch 텐서는 캐시에 효율적으로 접근할 수 있도록 연속된 배열로 저장된다. 연결 구조는 autograd 그래프를 훑는 데 내부적으로 쓰인다. 이 선택은 메모리 사용량(배열에는 포인터 부담이 없다)과 접근 양상(캐시 지역성 덕분에 순차적인 배열 접근이 연결 리스트 순회보다 10~100배 빠르다)에 모두 영향을 준다.

---

**연습문제 4.**
반복문 불변식을 사용하여 삽입과 삭제의 주요 연산의 시간 복잡도를 증명하라.

??? success "연습문제 4 풀이"
    알고리즘의 반복문이 유지하는 불변식을 진술하라. 초기화, 유지, 종료를 증명하라. 이 불변식으로부터 반복문이 명시된 횟수 안에 끝남이 따라 나오며, 이로써 복잡도의 상계가 확립된다. $\square$

## 정리하며

이 마당은 삽입 알고리즘、삭제 알고리즘、구현、갱신 배열을 차례로 짚었다.

**참고 문헌**

- Pugh, W. "Skip Lists: A Probabilistic Alternative to Balanced Trees."
  *Communications of the ACM*, 33(6), 1990.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C.
  *Introduction to Algorithms* (4th ed.). MIT Press.
