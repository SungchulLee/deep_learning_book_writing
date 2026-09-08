# 합치기-찾기의 쓰임새

합치기-찾기는 뜻밖의 곳에 나타나는 자료 짜임 가운데 하나이다. 거의 상수 시간 연산으로 바뀌는 이어진 조각을 지닐 수 있어 그래프 이론, 그림 다루기, 그물 살피기, 그리고 많은 겨루기 프로그래밍 문제에서 알고리즘의 등뼈가 된다. 이 쪽에서는 가장 중요한 쓰임새를 훑어본다.

---

## 1. 크러스컬의 최소 뻗은 나무 알고리즘

합치기-찾기의 가장 고전적인 쓰임새는 최소 뻗은 나무를 구하는 크러스컬 알고리즘이다. 이 알고리즘은 변을 무게가 커지는 차례로 다루며 $u$과 $v$이 서로 다른 조각에 있을 때만 변 $(u, v)$을 최소 뻗은 나무에 더한다. 합치기-찾기는 이 이어짐 살피기를 효율적으로 만든다:

- `find(u) != find(v)`이 그 변이 순환을 만드는지 정한다.
- 변을 더할 때 `union(u, v)`이 두 조각을 합친다.

합치기-찾기를 쓰면 크러스컬 알고리즘은 (변 정렬이 좌우하는) $O(m \log m)$ 시간에 돌아가며, 합치기-찾기 연산은 $O(m \cdot \alpha(n))$만 보탠다.

---

## 2. 바뀌는 이어짐

변이 하나씩 더해지는 그래프에서 합치기-찾기는 더할 때마다 "$u$과 $v$이 이어져 있는가?"에 답한다. 이것이 흐름 속 이어진 조각 문제이며, 합치기-찾기가 이를 가장 좋게 푼다.

---

## 3. 방향 없는 그래프에서 순환 알아내기

방향 없는 그래프의 변을 다룰 때 변 $(u, v)$은 $u$과 $v$이 이미 같은 조각에 있을 때 그리고 오직 그때만 순환을 만든다. 이로써 $O(m \cdot \alpha(n))$의 순환 알아내기 알고리즘을 얻으며, 어떤 쓰임새에서는 깊이 우선 찾기에 바탕을 둔 방식보다 단순하다.

---

## 4. 그림 나누기(이어진 조각에 이름표 붙이기)

그림 다루기에서 **이어진 조각에 이름표 붙이기**는 같은 성질(이를테면 색이나 밝기)을 나눠 갖는 화소가 이어진 구역을 가려낸다. 합치기-찾기는 화소를 훑는 차례로 다루며, 화소가 이웃과 맞으면 둘을 합친다. 화소를 모두 다루고 나면 조각마다 오직 하나뿐인 이름표를 갖는다. 이는 화소 수에 거의 비례하는 시간에 돌아간다.

---

## 5. 그물 스며듦

스며듦 이론에서는 격자의 자리를 무작위로 "열고" 위에서 아래로 가는 길이 있는지 묻는다. 합치기-찾기는 자리가 열릴 때마다 이어진 뭉치를 좇는다. 위와 아래에 가상 마디를 더하면 `connected(top, bottom)` 물음 한 번으로 스며듦이 일어났는지 정할 수 있다.

---

## 6. 구현

```python
"""
합치기-찾기의 쓰임새: 크러스컬의 최소 뻗은 나무와 순환 알아내기.

그래프 알고리즘에서 합치기-찾기의 흔한 두 쓰임을 보인다:
곧 방향 없는 그래프에서 최소 뻗은 나무를 세우는 것과
순환을 알아내는 것이다.
"""

# === 합치기-찾기 ===

class UnionFind:
    """길 줄이기와 계급으로 합치기를 쓴 합치기-찾기."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n

    def find(self, x: int) -> int:
        """길 누르기로 뿌리를 찾는다."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> bool:
        """계급으로 합치기. 합침이 일어났으면 True를 돌려준다."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.components -= 1
        return True

    def connected(self, a: int, b: int) -> bool:
        """a과 b이 같은 조각에 있는지 살핀다."""
        return self.find(a) == self.find(b)

# === 크러스컬의 최소 뻗은 나무 ===

def kruskal_mst(n: int, edges: list) -> list:
    """합치기-찾기를 쓴 크러스컬 알고리즘으로 최소 뻗은 나무 찾기.

    인수:
        n: 꼭짓점의 개수.
        edges: (무게, u, v) 짝의 목록.

    반환값:
        최소 뻗은 나무의 (무게, u, v) 변 목록.
    """
    edges_sorted = sorted(edges)
    uf = UnionFind(n)
    mst = []

    for w, u, v in edges_sorted:
        if uf.union(u, v):
            mst.append((w, u, v))
            if len(mst) == n - 1:
                break

    return mst

# === 고리 알아내기 ===

def has_cycle(n: int, edges: list) -> bool:
    """합치기-찾기로 방향 없는 그래프에 순환이 있는지 알아내기.

    인수:
        n: 꼭짓점의 개수.
        edges: (u, v) 짝의 목록.

    반환값:
        그래프에 순환이 있으면 True.
    """
    uf = UnionFind(n)
    for u, v in edges:
        if not uf.union(u, v):
            return True
    return False

# === 시연 ===

if __name__ == "__main__":
    # 크러스컬의 최소 뻗은 나무
    edges = [
        (4, 0, 1), (8, 0, 7), (11, 1, 7), (8, 1, 2),
        (7, 2, 3), (4, 2, 5), (2, 2, 8), (9, 3, 4),
        (14, 3, 5), (10, 4, 5), (2, 5, 6), (1, 6, 7),
        (6, 6, 8), (7, 7, 8)
    ]
    mst = kruskal_mst(9, edges)
    total = sum(w for w, u, v in mst)
    print("Kruskal's MST:")
    for w, u, v in mst:
        print(f"  ({u},{v}) weight={w}")
    print(f"Total MST weight: {total}")
    print()

    # 순환 알아내기
    print("Cycle detection:")
    edges_no_cycle = [(0, 1), (1, 2), (2, 3)]
    print(f"  Tree edges {edges_no_cycle}: "
          f"has_cycle={has_cycle(4, edges_no_cycle)}")

    edges_with_cycle = [(0, 1), (1, 2), (2, 3), (3, 0)]
    print(f"  Cycle edges {edges_with_cycle}: "
          f"has_cycle={has_cycle(4, edges_with_cycle)}")
```

**출력:**
```
Kruskal's MST:
  (6,7) weight=1
  (2,8) weight=2
  (5,6) weight=2
  (0,1) weight=4
  (2,5) weight=4
  (2,3) weight=7
  (0,7) weight=8
  (3,4) weight=9
Total MST weight: 37

Cycle detection:
  Tree edges [(0, 1), (1, 2), (2, 3)]: has_cycle=False
  Cycle edges [(0, 1), (1, 2), (2, 3), (3, 0)]: has_cycle=True
```

---

## 7. 응용 요약

| 쓰임새 | 합치기-찾기가 하는 일 | 전체 시간 |
|-------------|----------------|------------|
| 크러스컬 최소 뻗은 나무 | 순환 피하기 | $O(m \log m)$ |
| 바뀌는 이어짐 | 흐름 속 조각 좇기 | $O(m \cdot \alpha(n))$ |
| 순환 알아내기 | 같은 조각인지 살피기 | $O(m \cdot \alpha(n))$ |
| 그림 나누기 | 화소 구역 합치기 | $O(\text{pixels} \cdot \alpha(\text{pixels}))$ |
| 그물 스며듦 | 뭉치 좇기 | $O(\text{sites} \cdot \alpha(\text{sites}))$ |

---

## 연습문제

**연습문제 1.**
변이 하나씩 더해질 때 합치기-찾기가 방향 없는 그래프의 순환을 어떻게 알아내는지 설명하여라.

??? success "연습문제 1 풀이"
    꼭짓점마다 저마다의 모음으로 첫값을 잡는다. 변을 하나씩 다룬다. 곧 변 $(u, v)$에 대해 Find(u)과 Find(v)을 부른다. 같은 대표를 돌려주면 $u$과 $v$이 이미 이어져 있으므로 이 변을 더하면 순환이 생긴다. 다른 대표를 돌려주면 Union(u, v)을 불러 조각을 합친다. 이러면 순환을 처음 만드는 변을 $O(E \cdot \alpha(V))$ 시간에 알아낸다. $\square$

---

**연습문제 2.**
최소 뻗은 나무를 세우는 크러스컬 알고리즘에서 합치기-찾기를 어떻게 쓰는지 설명하여라.

??? success "연습문제 2 풀이"
    변을 무게로 정렬한다. 꼭짓점마다 저마다의 모음으로 첫값을 잡는다. 변을 차례로 다룬다. 곧 변 $(u, v, w)$에 대해 Find(u) $\neq$ Find(v)이면 그 변을 최소 뻗은 나무에 더하고 Union(u, v)을 부른다. Find(u) $=$ Find(v)이면 (순환이 생기므로) 그 변을 건너뛴다. 변 $V - 1$개를 받아들이면 최소 뻗은 나무가 완성된다. 합치기-찾기는 변마다 이어짐 살피기를 거의 $O(1)$으로 만들므로 병목은 $O(E \log E)$의 정렬이다. $\square$

---

**연습문제 3.**
변이 더해지기만 하고 없어지지는 않는 바뀌는 그래프에서 합치기-찾기로 이어진 조각의 개수를 어떻게 셀 수 있는가?

??? success "연습문제 3 풀이"
    세는 값 $c = V$으로 첫값을 잡는다(꼭짓점마다 저마다의 조각이다). 더해지는 변 $(u, v)$마다 Find(u) $\neq$ Find(v)이면 Union(u, v)을 부르고 $c$을 1 줄인다. 어느 때든 $c$이 지금의 이어진 조각 개수를 준다. Union마다 조각 딱 둘을 하나로 합쳐 개수를 1 줄이므로 이것이 통한다. 변을 모두 다루고 나면 $c$이 마지막 그래프의 이어진 조각 개수와 같다. $\square$

---

**연습문제 4.**
방향 그래프에도 합치기-찾기를 쓸 수 있는가? 어떤 한계가 있는가?

??? success "연습문제 4 풀이"
    표준 합치기-찾기는 방향 없는 이어짐(대칭 관계)을 좇는다. 방향 그래프에서 닿음은 대칭이 아니다. 곧 $v$이 $u$에 닿지 못해도 $u$은 $v$에 닿을 수 있다. 합치기-찾기는 강한 이어짐이나 방향 있는 닿음을 곧바로 셈하지 못한다. 다만 밑에 깔린 방향 없는 그래프에 합치기-찾기를 써서 약한 이어짐은 살필 수 있다. 강한 조각에는 타잔이나 코사라주 같은 알고리즘이 필요하다. 방향 그래프의 비실시간 바뀌는 이어짐을 위한 합치기-찾기의 특수한 갈래가 있기는 하지만 훨씬 복잡하다. $\square$

## 정리하며

이 마당은 크러스컬의 최소 뻗은 나무 알고리즘、바뀌는 이어짐、방향 없는 그래프에서 순환 알아내기、그림 나누기(이어진 조각에 이름표 붙이기)을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 19장과 21장. MIT Press.
- Sedgewick, R., & Wayne, K. (2011). *Algorithms* (4th ed.), 1.5장. Addison-Wesley.
