# 겹침 질의

[구간 트리](structure.md)가 받쳐 주는 주된 연산은 **겹침 질의**이다. 질의 구간 $[q_{low}, q_{high}]$이 주어지면 그것과 겹치는 구간을 트리에서 찾는다. 노드마다 담은 [증강한 최대 끝점 칸](augmented.md)이 겹치는 구간을 담을 만한 부분 트리로 찾기를 이끌고 그렇지 않은 것은 쳐 내어 이를 $O(\log n)$ 시간에 가능하게 한다.

## 두 구간은 언제 겹치는가

닫힌 구간 $[a, b]$과 $[c, d]$은 다음일 때만 겹친다.

$$
a \le d \quad \text{and} \quad c \le b
$$

달리 말해 $b < c$이거나 $d < a$이면(한쪽이 다른 쪽이 시작하기 전에 끝나면) 겹치지 **않는다**.

## 겹침 찾기 알고리즘

질의 구간 $q = [q_{low}, q_{high}]$과 $x$을 뿌리로 하는 구간 트리가 주어졌을 때 다음과 같이 한다.

```
INTERVAL-SEARCH(T, q):
    x = T.root
    while x != T.nil and q does not overlap x.interval:
        if x.left != T.nil and x.left.max >= q.low:
            x = x.left
        else:
            x = x.right
    return x
```

알고리즘은 뿌리에서 내려가며 노드마다 왼쪽이나 오른쪽을 고른다. 겹치는 구간을 찾거나 $x$이 nil 노드에 닿을 때(겹치는 것이 없을 때) 끝난다.

## 결정 규칙

내부 노드 $x$마다 알고리즘이 다음을 살핀다.

**왼쪽으로 갈 조건:** $x.left \ne \text{nil}$이고 $x.left.max \ge q_{low}$.

**그렇지 않으면 오른쪽으로 간다.**

직관은 이렇다. 왼쪽 부분 트리의 최대 끝점이 $q_{low}$ 이상이면 왼쪽 부분 트리의 어떤 구간이 오른쪽으로 충분히 뻗어 $q$과 겹칠 *수도* 있다. 왼쪽 부분 트리의 max가 $q_{low}$보다 작으면 왼쪽 부분 트리의 어떤 구간도 $q$과 겹칠 수 없으므로(모두 $q$이 시작하기 전에 끝난다) 알고리즘이 오른쪽으로 간다.

## 올바름

!!! note "옳음 정리"
    알고리즘이 왼쪽으로 가면, 왼쪽 부분 트리에 겹치는 구간이 있거나 트리 전체에 $q$과 겹치는 구간이 없다. 알고리즘이 오른쪽으로 가면 왼쪽 부분 트리에는 겹치는 구간이 없다.

**오른쪽으로 가는 경우의 증명.** $x.left.max < q_{low}$이면 왼쪽 부분 트리의 모든 구간 $[a, b]$이 $b \le x.left.max < q_{low}$을 만족하므로 $b < q_{low}$이고 겹칠 수 없다. $\square$

**왼쪽으로 가는 경우의 증명.** $x.left.max \ge q_{low}$이어서 알고리즘이 왼쪽으로 간다고 하자. $[a, b]$을 왼쪽 부분 트리에서 최대 끝점을 이루는 구간($b = x.left.max$)이라 하자. $q$이 $[a, b]$과 겹치지 않으면 $q_{high} < a$이다. 왼쪽 부분 트리의 모든 구간은 왼쪽 끝점이 $a$ 이하이다(왼쪽 끝점에 대한 이진 탐색 트리 순서로… 사실 $a$은 어떤 구간의 왼쪽 끝점일 뿐이지만, 이진 탐색 트리 순서에 따라 오른쪽 부분 트리의 모든 구간은 왼쪽 끝점이 $x.key$ 이상이다). $q_{high} < a \le x.low$(노드 구간의 왼쪽 끝점)이고 오른쪽 부분 트리의 모든 구간은 왼쪽 끝점이 $x.low \ge a > q_{high}$ 이상이므로 오른쪽 부분 트리의 어떤 구간도 $q$과 겹치지 않는다. $\square$

## 겹치는 구간 모두 찾기

기본 알고리즘은 겹치는 구간 하나를 돌려준다. $q$과 겹치는 $k$개 구간을 **모두** 찾으려면 겹칠 수 있을 때 두 부분 트리를 모두 살피도록 고친다.

1. $x$의 구간이 $q$과 겹치면 $x$을 보고한다.
2. $x.left \ne \text{nil}$이고 $x.left.max \ge q_{low}$이면 왼쪽 부분 트리로 재귀한다.
3. $x.right \ne \text{nil}$이고 $x.right.max \ge q_{low}$이며 $x.key \le q_{high}$이면(오른쪽 부분 트리에 $q$이 끝나기 전에 시작하는 구간이 있으면) 오른쪽 부분 트리로 재귀한다.

이 변형은 겹치는 구간의 수를 $k$이라 할 때 $O(k \log n)$ 시간에 돈다. (노드마다 정렬된 목록을 둔 증강 구간 트리처럼) 더 정교한 짜임은 $O(\log n + k)$을 이룰 수 있다.

## 구현

```python
"""겹침 질의를 하는 구간 트리."""

from __future__ import annotations


# === 노드 정의 ===

class IntervalNode:
    """[low, high]과 증강한 max를 담는 구간 트리 노드."""

    def __init__(self, low: int, high: int):
        self.low = low
        self.high = high
        self.max = high
        self.left: IntervalNode | None = None
        self.right: IntervalNode | None = None


# === 삽입 ===

def insert(root: IntervalNode | None, low: int, high: int) -> IntervalNode:
    """구간 [low, high]을 구간 트리에 넣는다."""
    if root is None:
        return IntervalNode(low, high)
    if low < root.low:
        root.left = insert(root.left, low, high)
    else:
        root.right = insert(root.right, low, high)
    root.max = max(root.max, high)
    return root


# === 겹침 찾기 ===

def overlap_search(root: IntervalNode | None,
                   q_low: int, q_high: int) -> IntervalNode | None:
    """[q_low, q_high]과 겹치는 구간을 찾는다. 없으면 None을 돌려준다."""
    x = root
    while x is not None:
        if x.low <= q_high and q_low <= x.high:
            return x  # 겹침을 찾았다
        if x.left is not None and x.left.max >= q_low:
            x = x.left
        else:
            x = x.right
    return None


# === 시연 ===

if __name__ == "__main__":
    root: IntervalNode | None = None
    intervals = [(15, 20), (10, 30), (17, 19), (5, 20), (12, 15), (30, 40)]
    for lo, hi in intervals:
        root = insert(root, lo, hi)

    query = (14, 16)
    result = overlap_search(root, *query)
    if result:
        print(f"Query {query} overlaps [{result.low}, {result.high}]")
    else:
        print(f"Query {query}: no overlap found")
```

## 복잡도

| 연산 | 시간 |
|-----------|------|
| 겹침 하나 찾기 | $O(\log n)$ |
| 겹침 $k$개 모두 찾기 | $O(k \log n)$ |
| 삽입 | $O(\log n)$ |
| 삭제 | $O(\log n)$ |

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Section 14.3. MIT Press.
- de Berg, M., Cheong, O., van Kreveld, M., & Overmars, M. (2008). *Computational Geometry: Algorithms and Applications* (3rd ed.), Chapter 10. Springer.


## 연습문제

**연습문제 1.**
겹침 질의의 짜임을 설명하고 질의와 갱신의 복잡도를 밝혀라.

??? success "연습문제 1 풀이"
    이 짜임은 위계적 분해를 이용해 일차보다 빠른 질의 시간을 이룬다. 대개 질의와 갱신이 모두 $O(\log n)$이고 세우는 데 $O(n)$이나 $O(n\log n)$이 든다.

---

**연습문제 2.**
입력 $[3, 1, 4, 1, 5, 9, 2, 6]$으로 겹침 질의를 세워라. 마지막 짜임을 보여라.

??? success "연습문제 2 풀이"
    세우기 알고리즘을 적용하며 중간 상태를 보여라. 트리 짜임이면 트리를 그려라. 배열에 바탕한 짜임이면 부모-자식 관계를 덧붙여 배열의 내용을 보여라.

---

**연습문제 3.**
질의 연산이 올바른 어떤 질의 범위에 대해서도 옳은 결과를 돌려줌을 증명하라.

??? success "연습문제 3 풀이"
    증명에는 대개 트리의 높이에 대한 귀납법을 쓴다. 노드마다 질의가 한 부분 트리 안에 온전히 들거나(재귀한다) 두 부분 트리에 걸친다(부분 결과를 모은다). 모으는 함수(합, 최솟값, 최댓값)가 결합적이므로 올바로 모아진다. $\square$

---

**연습문제 4.**
겹침 질의가 질의마다의 계산을 $O(n)$에서 $O(\log n)$으로 빠르게 하는 딥러닝 응용을 설명하라.

??? success "연습문제 4 풀이"
    응용으로는 누적 어텐션 가중치를 위한 접두사 합 질의, 길이가 제각각인 수열에서 효율적인 풀링을 위한 범위 질의, 검색 증강 생성을 위한 최근접 이웃 찾기, 3차원 딥러닝의 점 구름 처리를 위한 공간 색인이 있다.