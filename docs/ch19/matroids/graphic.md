# 그래프 매트로이드

그래프 매트로이드는 가장 자연스럽고 역사적으로 중요한 매트로이드의 보기이다. 아무 방향 없는 그래프에서든 숲(순환이 없는 아래그래프)을 이루는 변 모음이 매트로이드 공리를 채운다. 크러스컬 같은 욕심쟁이 알고리즘이 가장 좋은 뻗은 나무를 내는 것은 바로 이 매트로이드 짜임 때문이다. 그래프 매트로이드를 알면 어떤 그래프 문제는 욕심쟁이 전략에 넘어가고 다른 것은 그렇지 않은 까닭이 드러난다.

## 정의

방향 없는 그래프 $G = (V, E)$이 주어졌을 때 **그래프 매트로이드**(**순환 매트로이드**라고도 한다)는 $M(G) = (E, \mathcal{I})$이며 다음과 같다.

- **바탕 모음**은 변 모음 $E$이다.
- 부분 모음 $F \subseteq E$이 **홀로 선** 것은 $F$이 비순환인 것, 곧 $F$이 숲을 이루는 것과 같은 뜻이다.

핵심 조각이 그래프 개념에 곧바로 대응된다:

| 매트로이드 개념 | 그래프에서의 뜻 |
|-----------------|---------------------|
| 바탕 모음 $E$ | $G$의 모든 변 |
| 얽히지 않는 모음 | 숲(순환 없는 변 부분 모음) |
| 회로 | 단순 순환 |
| 기저 | 뻗은 숲 |
| $A \subseteq E$의 계수 | $A$이 이끄는 아래그래프의 뻗는 숲에 든 변의 수 |

## 매트로이드 공리 확인하기

$E$의 순환 없는 부분 모음이 매트로이드 공리 셋을 채우는지 살펴야 한다.

**공리 1(비지 않음).** 빈 모음 $\emptyset$에는 변이 없으므로 저절로 비순환이다. 따라서 $\emptyset \in \mathcal{I}$이다.

**공리 2(물림 성질).** $F$이 비순환이고 $F' \subseteq F$이면 $F'$도 비순환이다. 숲에서 변을 없앤다고 순환이 생길 수는 없다.

**공리 3(맞바꿈 성질).** $|F_1| < |F_2|$인 $F_1, F_2 \in \mathcal{I}$을 보자. 둘 다 숲이므로 $F_1$은 많아야 $|V| - |F_1|$개의 이어진 조각에 걸치고 $F_2$은 $|V| - |F_2|$개의 조각에 걸친다. $|F_1| < |F_2|$이므로 숲 $F_2$의 조각이 더 적고, 따라서 어떤 변 $e \in F_2 \setminus F_1$이 $F_1$의 조각 둘을 잇는다. $F_1$에 $e$을 더해도 순환이 생기지 않으므로(조각 둘을 다리처럼 잇는다) $F_1 \cup \{e\} \in \mathcal{I}$이다.

!!! note "맞바꿈 성질이 통하는 까닭"
    꼭짓점 $n$개에 변 $k$개인 숲은 이어진 조각이 정확히 $n - k$개다. $F_2$의 변이 $F_1$보다 많으면 조각은 더 적다. 비둘기집 원리로 $F_2$의 어떤 변 하나는 $F_1$의 서로 다른 두 조각을 잇고, 그 변을 더해도 순환이 생기지 않는다.

## 계수와 기저

그래프 매트로이드의 **계수**는 $|V| - c(G)$이며, 여기서 $c(G)$은 $G$의 이어진 조각 수이다. 이어진 그래프에서는 계수가 $|V| - 1$이다.

모든 기저(가장 큰 얽히지 않는 모음)는 **뻗은 숲**이다. 이어진 그래프에서는 기저마다 변이 정확히 $|V| - 1$개인 뻗은 나무이다. 매트로이드 공리가 모든 기저의 크기가 같음을 보장하는데, 그래프의 말로 하면 이어진 그래프의 모든 뻗은 나무는 변의 수가 같다는 뜻이다.

## 회로

그래프 매트로이드의 **회로**는 가장 작은 얽힌 모음이며, 그래프의 단순 순환에 맞대응된다. 회로의 근본 성질이 그래프 이론과 이어진다:

!!! note "회로가 하나뿐이라는 성질"
    $F$이 숲이고 $e \notin F$이면 $F \cup \{e\}$에는 순환이 많아야 하나 있다. 그런 순환이 있다면 그것이 $F$에 대해 $e$을 담은 하나뿐인 회로다.

이 성질은 뻗은 나무에 변을 하나 더하면 순환이 정확히 하나 생기고, 그 순환에서 아무 변이나 없애면 또 다른 뻗은 나무가 되는 까닭을 이해하는 데 꼭 필요하다.

## 최소 뻗은 나무와의 이음

그래프 매트로이드는 크러스컬 알고리즘이 통하는 까닭을 밝혀 준다. 매트로이드 욕심쟁이 정리에 따르면, 원소를 무게가 커지는 차례로 다루며 얽히지 않음을 지키면 더하는 욕심쟁이 알고리즘은 무게가 가장 큰 기저를 낸다. 무게의 부호를 뒤집은 그래프 매트로이드에서는 이것이 최소 뻗은 나무가 된다.

구체적으로 크러스컬 알고리즘은:

1. 변을 무게로 정렬한다.
2. 변을 차례대로 다룬다.
3. 순환을 만들지 않으면(곧 그래프 매트로이드에서 얽히지 않으면) 그 변을 더한다.

그 결과 모음이 무게가 가장 작은 기저이며, 이는 바로 최소 뻗은 나무이다.

## 구현

```python
"""
그래프 매트로이드 연산.

공리를 확인하고 계수를 셈하고 매트로이드 욕심쟁이 알고리즘으로
가장 가벼운 바탕(최소 뻗음 나무)을 찾아
그래프 매트로이드를 보인다.
"""

from itertools import combinations

# === 돌기 알아내기를 위한 합치기-찾기 ===

class UnionFind:
    """계수로 합치기와 길 눌러 담기를 쓴 합치기-찾기."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        """x와 y가 든 모임을 합친다. 이미 같은 모임이면 거짓을 돌려준다."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


# === 그래프 매트로이드 ===

def is_independent(n: int, edges: list[tuple[int, int]]) -> bool:
    """변의 모임이 숲을 이루는지(홀로서기인지) 살핀다.

    인수:
        n: 꼭짓점의 개수.
        edges: (u, v) 변의 목록.

    반환값:
        변 모임에 돌기가 없으면 참.
    """
    uf = UnionFind(n)
    for u, v in edges:
        if not uf.union(u, v):
            return False
    return True


def matroid_rank(n: int, edges: list[tuple[int, int]]) -> int:
    """그래프 매트로이드에서 변 모임의 계수를 셈한다.

    계수는 돌기 없는 가장 큰 부분 모임의 크기와 같다.

    인수:
        n: 꼭짓점의 개수.
        edges: (u, v) 변의 목록.

    반환값:
        주어진 변 안에서 가장 큰 숲의 크기.
    """
    uf = UnionFind(n)
    rank = 0
    for u, v in edges:
        if uf.union(u, v):
            rank += 1
    return rank


def minimum_spanning_tree(
    n: int,
    weighted_edges: list[tuple[int, int, float]]
) -> list[tuple[int, int, float]]:
    """매트로이드 욕심쟁이 알고리즘(크러스컬)으로 최소 뻗음 나무를 찾는다.

    인수:
        n: 꼭짓점의 개수.
        weighted_edges: (u, v, 무게) 짝의 목록.

    반환값:
        최소 뻗음 나무에 든 변의 목록.
    """
    sorted_edges = sorted(weighted_edges, key=lambda e: e[2])
    uf = UnionFind(n)
    mst = []

    for u, v, w in sorted_edges:
        if uf.union(u, v):
            mst.append((u, v, w))
            if len(mst) == n - 1:
                break

    return mst


# === 시연 ===

if __name__ == "__main__":
    # 그래프: 꼭짓점 4개, 변 5개
    #   0 --1-- 1
    #   |      /|
    #   4    3  2
    #   |  /    |
    #   2 --5-- 3
    n = 4
    edges = [(0, 1), (1, 3), (1, 2), (0, 2), (2, 3)]

    # 여러 부분 모임의 홀로서기를 살핀다
    print("Independence checks:")
    forest = [(0, 1), (1, 2), (2, 3)]
    print(f"  {forest}: {is_independent(n, forest)}")

    cycle = [(0, 1), (1, 2), (0, 2)]
    print(f"  {cycle}: {is_independent(n, cycle)}")

    # 계수
    print(f"\nRank of all edges: {matroid_rank(n, edges)}")
    print(f"Expected (|V| - 1): {n - 1}")

    # 최소 뻗음 나무
    weighted = [(0, 1, 1), (1, 3, 3), (1, 2, 2), (0, 2, 4), (2, 3, 5)]
    mst = minimum_spanning_tree(n, weighted)
    total = sum(w for _, _, w in mst)
    print(f"\nMST edges: {[(u, v, w) for u, v, w in mst]}")
    print(f"MST weight: {total}")
```

**출력:**

```
Independence checks:
  [(0, 1), (1, 2), (2, 3)]: True
  [(0, 1), (1, 2), (0, 2)]: False

Rank of all edges: 3
Expected (|V| - 1): 3

MST edges: [(0, 1, 1), (1, 2, 2), (1, 3, 3)]
MST weight: 6
```

숲 $\{(0,1), (1,2), (2,3)\}$은 홀로 서 있고(비순환이다) $\{(0,1), (1,2), (0,2)\}$은 세모를 이루어 매여 있다. 이 이어진 그래프의 계수는 $|V| - 1 = 3$이다. 크러스컬 알고리즘(매트로이드 욕심쟁이 알고리즘)이 짐이 6인 가장 작은 뻗는 나무를 찾는다.

## 쌍대 그래프 매트로이드

그래프 매트로이드의 **짝**은 **짝 그래프 매트로이드** $M^*(G) = (E, \mathcal{I}^*)$이며, $F \subseteq E$이 홀로 선 것은 $G \setminus F$(변 $F$을 없앤 그래프)이 이어진 채로 남는 것과 같은 뜻이다. $M^*(G)$의 밑틀은 뻗는 나무의 여집합이다. 곧 $T$이 뻗는 나무이면 $E \setminus T$이 짝 그래프 매트로이드의 밑틀이다.

## 참고 문헌

- Whitney, H. (1935). On the abstract properties of linear dependence. *American Journal of Mathematics*, 57(3), 509--533.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), 16장: Greedy Algorithms.
- Oxley, J. G. (2011). *Matroid Theory* (2nd ed.). Oxford University Press.

## 연습문제

**연습문제 1.**
그래프 매트로이드에서 욕심쟁이 고름이 무엇인지 가려내고 왜 가장 좋은 풀이로 이어지는지 밝혀라.

??? success "연습문제 1 풀이"
    욕심쟁이 고름은 걸음마다 그 자리에서 가장 좋은 것을 고른다. Graphic Matroid에서는 이 고름이 욕심쟁이 고름 성질을 채운다. 곧 이 욕심쟁이 고름을 담은 가장 좋은 풀이가 있다. 여기에 가장 좋은 아래 짜임(욕심쟁이 고름 뒤 남은 아래 문제도 같은 전략으로 가장 좋게 풀린다)을 더하면 욕심쟁이 알고리즘이 두루 가장 좋은 풀이를 내놓는다. $\square$

---

**연습문제 2.**
그래프 매트로이드이 가장 좋은 아래 짜임을 갖는지 증명하거나 반증하여라.

??? success "연습문제 2 풀이"
    가장 좋은 아래 짜임이란 문제의 가장 좋은 풀이가 그 아래 문제의 가장 좋은 풀이를 담는다는 뜻이다. Graphic Matroid에서는 욕심쟁이 고름을 하고 나면 남은 문제가 같은 갈래의 더 작은 사례가 된다. 아래 문제의 풀이가 가장 좋지 않다면 그것을 갈음해 온 풀이를 더 낫게 할 수 있는데, 이는 온 풀이가 가장 좋다는 것과 어긋난다. 그러므로 가장 좋은 아래 짜임이 이루어진다. $\square$

---

**연습문제 3.**
그래프 매트로이드의 시간 복잡도는 무엇인가? 가장 값비싼 단계를 가려내어라.

??? success "연습문제 3 풀이"
    시간 복잡도는 (필요하다면) 줄 세우는 걸음과 욕심쟁이 고름 되돌이에 달렸다. 흔히 줄 세우기가 $O(n \log n)$으로 가장 크다. 욕심쟁이 되돌이는 원소마다 한 번씩 다루어 $O(n)$이다. 모두 $O(n \log n)$이다. 들임이 미리 줄 세워져 있으면 알고리즘은 $O(n)$에 돈다. $\square$

---

**연습문제 4.**
(그래프 매트로이드에서 쓴 것이 아닌) 다른 욕심쟁이 전략은 가장 좋은 풀이를 내지 못함을 보이는 반례를 들어라.

??? success "연습문제 4 풀이"
    문제의 짜임과 맞지 않는 다른 욕심쟁이 잣대를 생각해 보자. 이 잣대는 뒤에 더 좋은 고름을 막는 원소를 고를 수 있다. 어긋나는 보기는 잘못된 욕심쟁이 잣대가 가장 좋지 않은 결과를 낼 수 있음을 보이며, 문제마다 그에 맞는 욕심쟁이 고름 성질을 반드시 증명해야 하는 까닭을 드러낸다. $\square$
