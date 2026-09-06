# 영속 구간 트리

자료 구조의 지난 상태에 대한 질의에 답해야 할 때가 있다. 이를테면 갱신을 잇달아 받는 배열이 있을 때 $k$번째 갱신 뒤의 상태로 어떤 범위의 합을 묻고 싶을 수 있다. **영속 구간 트리**는 트리의 지난 판을 모두 지녀, 갱신마다 트리를 통째로 베끼지 않고도 어떤 지난 판에든 질의할 수 있게 한다.

## 경로 베끼기

핵심 기법은 **경로 베끼기**이다. 자리 하나를 고칠 때 뿌리에서 잎까지의 경로 위 노드만 바뀐다. 영속 갱신은 그 $O(\log n)$개 노드만 새로 베끼고, 바뀌지 않은 부분 트리는 앞 판과 함께 쓴다. 판마다 그 뿌리 포인터로 가려낸다.

!!! note "짜임 함께 쓰기"
    트리에 잎이 $n$개이고 높이가 $h = O(\log n)$이면 갱신마다 새 노드가 꼭 $h + 1$개 생긴다. 나머지 노드는 모두 앞 판과 함께 쓴다. 갱신 $q$번에 걸쳐 전체 공간은 $O(n + q \log n)$인데, 매번 트리를 통째로 베끼면 $O(nq)$이다.

## 노드에 바탕한 표현

배열에 바탕한 구간 트리와 달리 영속 구간 트리는 **포인터에 바탕한 노드**를 쓰는데, 배열 색인이 판마다 부딪히기 때문이다. 노드마다 다음을 담는다.

- `value`: 그 범위의 모은 값.
- `left`, `right`: 자식 노드를 가리키는 것 (더 옛 판에 속할 수도 있다).

## 영속 갱신

판 $v$에서 자리 $i$을 새 값으로 두어 판 $v+1$을 만들려면 다음과 같이 한다.

1. 판 $v$의 뿌리에서 시작한다.
2. 지금 노드가 잎이면 고친 값으로 새 잎을 만든다.
3. 그렇지 않으면 새 내부 노드를 만든다. $i$이 왼쪽 자식의 범위에 들면 왼쪽 자식을 재귀적으로 고치고 옛 오른쪽 자식을 그대로 쓴다(반대도 마찬가지).
4. 새 노드의 값을 (새로울 수도 있는) 자식에서 다시 셈한다.
5. 이 새 경로의 뿌리가 판 $v+1$의 뿌리가 된다.

## 구현

```python
"""
경로 베끼기를 쓰는 영속 구간 트리.

갱신마다 뿌리-잎 경로 위의 O(log n)개 노드만 베껴 새 판을 만들고
바뀌지 않은 부분 트리는 모두 앞 판과
함께 쓴다.
"""


# === 노드 정의 ===

class Node:
    """영속 구간 트리의 바뀌지 않는 노드."""

    __slots__ = ('value', 'left', 'right')

    def __init__(self, value: int = 0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right


# === 영속 구간 트리 ===

class PersistentSegTree:
    """점 갱신과 범위 질의를 받쳐 주는 영속 구간 트리.

    갱신마다 새 뿌리(판)를 낸다. 어떤 판에든
    질의할 수 있다.
    """

    def __init__(self, data: list):
        """입력 배열로 판 0을 세운다."""
        self.n = len(data)
        self.roots = []
        if self.n > 0:
            root = self._build(data, 0, self.n - 1)
            self.roots.append(root)

    def _build(self, data: list, lo: int, hi: int) -> Node:
        """처음 트리를 재귀적으로 세운다."""
        if lo == hi:
            return Node(value=data[lo])
        mid = (lo + hi) // 2
        left = self._build(data, lo, mid)
        right = self._build(data, mid + 1, hi)
        return Node(value=left.value + right.value, left=left, right=right)

    def update(self, version: int, idx: int, val: int) -> int:
        """자리 idx를 val으로 두어 새 판을 만든다.

        새 판의 색인을 돌려준다.
        """
        new_root = self._update(self.roots[version], 0, self.n - 1, idx, val)
        self.roots.append(new_root)
        return len(self.roots) - 1

    def _update(self, node: Node, lo: int, hi: int,
                idx: int, val: int) -> Node:
        """경로 베끼기 갱신: idx까지의 경로를 따라 새 노드를 만든다."""
        if lo == hi:
            return Node(value=val)
        mid = (lo + hi) // 2
        if idx <= mid:
            new_left = self._update(node.left, lo, mid, idx, val)
            return Node(value=new_left.value + node.right.value,
                        left=new_left, right=node.right)
        else:
            new_right = self._update(node.right, mid + 1, hi, idx, val)
            return Node(value=node.left.value + new_right.value,
                        left=node.left, right=new_right)

    def query(self, version: int, l: int, r: int) -> int:
        """특정 판에 대한 범위 합 질의."""
        return self._query(self.roots[version], 0, self.n - 1, l, r)

    def _query(self, node: Node, lo: int, hi: int,
               l: int, r: int) -> int:
        """재귀 범위 질의."""
        if r < lo or hi < l:
            return 0
        if l <= lo and hi <= r:
            return node.value
        mid = (lo + hi) // 2
        return (self._query(node.left, lo, mid, l, r)
                + self._query(node.right, mid + 1, hi, l, r))

    def version_count(self) -> int:
        """담긴 판의 수를 돌려준다."""
        return len(self.roots)


# === 시연 ===

if __name__ == "__main__":
    data = [1, 3, 5, 7, 9]
    pst = PersistentSegTree(data)

    print(f"Version 0 (original): {data}")
    print(f"  Sum [0,4]: {pst.query(0, 0, 4)}")
    print(f"  Sum [1,3]: {pst.query(0, 1, 3)}")
    print()

    # 판 1: 자리 2를 50으로 둔다
    v1 = pst.update(0, 2, 50)
    print(f"Version 1 (set a[2]=50):")
    print(f"  Sum [0,4]: {pst.query(v1, 0, 4)}")
    print(f"  Sum [1,3]: {pst.query(v1, 1, 3)}")
    print()

    # 판 2: (판 1에 바탕하여) 자리 0을 100으로 둔다
    v2 = pst.update(v1, 0, 100)
    print(f"Version 2 (set a[0]=100, from v1):")
    print(f"  Sum [0,4]: {pst.query(v2, 0, 4)}")
    print()

    # 본디 판에도 여전히 닿을 수 있다
    print(f"Version 0 still intact:")
    print(f"  Sum [0,4]: {pst.query(0, 0, 4)}")
    print()

    print(f"Total versions: {pst.version_count()}")
```

**출력:**
```
Version 0 (original): [1, 3, 5, 7, 9]
  Sum [0,4]: 25
  Sum [1,3]: 15

Version 1 (set a[2]=50):
  Sum [0,4]: 70
  Sum [1,3]: 60

Version 2 (set a[0]=100, from v1):
  Sum [0,4]: 169

Version 0 still intact:
  Sum [0,4]: 25

Total versions: 3
```

## 복잡도

| 연산 | 시간 | 판마다의 공간 |
|-----------|------|-------------------|
| 세우기 (판 0) | $O(n)$ | 노드 $O(n)$개 |
| 점 갱신 (새 판) | $O(\log n)$ | 새 노드 $O(\log n)$개 |
| 범위 질의 | $O(\log n)$ | $O(1)$ |
| 갱신 $q$번 뒤 모두 | — | 노드 $O(n + q \log n)$개 |

공간 효율은 짜임을 함께 쓰는 데서 온다. 갱신마다 새 노드가 $O(\log n)$개만 생긴다.

## 응용

- **범위에서 $k$번째로 작은 값.** 정렬된 값 위에 영속 구간 트리를 세운다. 판 $i$은 앞의 $i$개 원소를 넣은 뒤의 상태를 나타낸다. 두 판의 차이가 $k$번째로 작은 값 질의에 답한다.
- **판 관리.** 모은 데이터에 되돌리기·다시하기가 필요한 어떤 응용에나 쓴다.
- **오프라인 질의.** 갱신 차례의 서로 다른 시점을 가리키는 질의에 답한다.

## 참고 문헌

- Driscoll, J. R., Sarnak, N., Sleator, D. D., & Tarjan, R. E. (1989). Making Data Structures Persistent. *Journal of Computer and System Sciences*, 38(1), 86-124.


## 연습문제

**연습문제 1.**
영속 구간 트리의 짜임을 설명하고 질의와 갱신의 복잡도를 밝혀라.

??? success "연습문제 1 풀이"
    이 짜임은 위계적 분해를 이용해 일차보다 빠른 질의 시간을 이룬다. 대개 질의와 갱신이 모두 $O(\log n)$이고 세우는 데 $O(n)$이나 $O(n\log n)$이 든다.

---

**연습문제 2.**
입력 $[3, 1, 4, 1, 5, 9, 2, 6]$으로 영속 구간 트리를 세워라. 마지막 짜임을 보여라.

??? success "연습문제 2 풀이"
    세우기 알고리즘을 적용하며 중간 상태를 보여라. 트리 짜임이면 트리를 그려라. 배열에 바탕한 짜임이면 부모-자식 관계를 덧붙여 배열의 내용을 보여라.

---

**연습문제 3.**
질의 연산이 올바른 어떤 질의 범위에 대해서도 옳은 결과를 돌려줌을 증명하라.

??? success "연습문제 3 풀이"
    증명에는 대개 트리의 높이에 대한 귀납법을 쓴다. 노드마다 질의가 한 부분 트리 안에 온전히 들거나(재귀한다) 두 부분 트리에 걸친다(부분 결과를 모은다). 모으는 함수(합, 최솟값, 최댓값)가 결합적이므로 올바로 모아진다. $\square$

---

**연습문제 4.**
영속 구간 트리가 질의마다의 계산을 $O(n)$에서 $O(\log n)$으로 빠르게 하는 딥러닝 응용을 설명하라.

??? success "연습문제 4 풀이"
    응용으로는 누적 어텐션 가중치를 위한 접두사 합 질의, 길이가 제각각인 수열에서 효율적인 풀링을 위한 범위 질의, 검색 증강 생성을 위한 최근접 이웃 찾기, 3차원 딥러닝의 점 구름 처리를 위한 공간 색인이 있다.