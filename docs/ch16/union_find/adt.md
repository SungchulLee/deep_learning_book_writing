# 서로 겹치지 않는 모음 추상 자료형

많은 문제에서 서로 겹치지 않는 무리의 모임을 좇으며 두 물음에 효율적으로 답해야 한다. 곧 "원소 $x$과 $y$이 같은 무리에 드는가?"와 "$x$과 $y$을 담은 무리를 합쳐라"이다. 이런 무리는 크러스컬 알고리즘의 이어진 조각, 하나로 맞추기 문제의 동치류, 그림 나누기의 구역으로 자연스럽게 나타난다. **서로 겹치지 않는 모음**(또는 **합치기-찾기**) 추상 자료형이 이 연산들에 깔끔한 창구를 준다.

## 정의

**서로 겹치지 않는 모음 자료 짜임**은 서로 겹치지 않고 바뀌는 모음의 모임 $\mathcal{S} = \{S_1, S_2, \ldots, S_k\}$을 지닌다. 모음 $S_i$마다 그 모음에서 뽑은 한 원소인 **대표**로 가려낸다. 이 자료 짜임은 연산 셋을 받쳐 준다:

### MAKE-SET(x)

원소 $x$만 담은 새 모음 $\{x\}$을 만든다. 이 홑원소 모음의 대표는 $x$ 자신이다.

**앞조건**: $x$이 아직 $\mathcal{S}$의 어떤 모음에도 들어 있지 않다.

**뒷조건**: $\mathcal{S} \leftarrow \mathcal{S} \cup \{\{x\}\}$.

### FIND(x)

$x$을 담은 오직 하나뿐인 모음의 대표를 돌려준다.

**앞조건**: $x$이 $\mathcal{S}$의 어떤 모음에 들어 있다.

**뒷조건**: $x \in S_i$인 $S_i$의 대표를 돌려준다. 모임 $\mathcal{S}$은 바뀌지 않는다(다만 효율을 위해 안의 짜임은 바뀔 수 있다).

**핵심 성질**: $x$과 $y$이 같은 모음에 있을 때 그리고 오직 그때만 $\text{FIND}(x) = \text{FIND}(y)$이다.

### UNION(x, y)

$x$과 $y$을 담은 두 모음을 하나로 합친다. 합쳐진 모음의 대표는 그 안의 아무 원소나 될 수 있다.

**앞조건**: $x$과 $y$이 $\mathcal{S}$의 (같을 수도 있는) 모음에 들어 있다.

**뒷조건**: $x \in S_i$이고 $y \in S_j$이며 $S_i \ne S_j$이면 $\mathcal{S} \leftarrow (\mathcal{S} \setminus \{S_i, S_j\}) \cup \{S_i \cup S_j\}$이다.

## 파이썬 창구

다음은 안의 나타냄을 정하지 않은 채 추상 자료형의 창구를 정한다. 이 절의 다음 쪽들에서 점점 더 빠른 구현을 세워 간다.

```python
"""
서로 겹치지 않는 모음 추상 자료형.

구현 방식을 정하지 않은 채
합치기-찾기 연산의 창구를 정한다.
"""


# === 추상 자료형 창구 ===

class DisjointSetADT:
    """서로 겹치지 않는 모음 자료 짜임의 추상 창구."""

    def __init__(self, n):
        """홑원소 모음 {0}, {1}, ..., {n-1} 만들기."""
        self.parent = list(range(n))  # 원소마다 저마다의 대표

    def find(self, x):
        """x을 담은 모음의 대표를 돌려준다."""
        while self.parent[x] != x:
            x = self.parent[x]
        return x

    def union(self, a, b):
        """
        a과 b을 담은 모음 합치기.

        a과 b이 서로 다른 모음에 있었으면 True를 돌려준다,
        이미 같은 모음에 있었으면 False.
        """
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return False
        self.parent[root_b] = root_a  # 어수룩하게: 한 뿌리가 다른 뿌리를 가리키게만 한다
        return True

    def connected(self, a, b):
        """a과 b이 같은 모음에 있는지 살핀다."""
        return self.find(a) == self.find(b)


# === 보기 ===

if __name__ == "__main__":
    uf = DisjointSetADT(5)
    print(f"Initially: 0 and 1 connected? {uf.connected(0, 1)}")

    uf.union(0, 1)
    uf.union(2, 3)
    print(f"After union(0,1) and union(2,3): 0 and 1 connected? {uf.connected(0, 1)}")
    print(f"0 and 3 connected? {uf.connected(0, 3)}")

    uf.union(1, 3)
    print(f"After union(1,3): 0 and 3 connected? {uf.connected(0, 3)}")
```

**출력:**
```
Initially: 0 and 1 connected? False
After union(0,1) and union(2,3): 0 and 1 connected? True
0 and 3 connected? False
After union(1,3): 0 and 3 connected? True
```

## 어수룩한 구현의 복잡도

위의 어수룩한 구현은(최적화가 없을 때) 값이 다음과 같다:

| 연산 | 시간 |
|-----------|------|
| MAKE-SET | $O(1)$ |
| FIND | 최악일 때 $O(n)$ |
| UNION | 최악일 때 $O(n)$(FIND 때문에) |

최악의 경우는 UNION이 늘 같은 사슬에 이어 붙여 깊이 $n$의 이음 목록을 만들 때이다. 다음 쪽들에서는 **계급으로 합치기**와 **길 줄이기**라는 최적화 둘을 소개한다. 이는 FIND과 UNION의 고르게 친 값을 $O(\alpha(n))$으로 줄이며, 여기서 $\alpha$은 거꿀 애커만 함수이다.

## 나타냄의 고름

고전 구현이 둘 있다:

| 방식 | FIND | UNION | 생각 |
|----------|------|-------|---------|
| **빠른 찾기** | $O(1)$ | $O(n)$ | 납작한 배열: `id[x]`이 모음의 대표를 곧바로 담는다 |
| **빠른 합치기** | 최악일 때 $O(n)$ | 최악일 때 $O(n)$ | 숲: `parent[x]`이 $x$의 어버이를 담고, 사슬을 따라 뿌리로 간다 |

다음 두 쪽에서 이 방식들을 자세히 살펴보고, 이어서 숲에 바탕을 둔 방식을 거의 최적으로 만드는 최적화를 다룬다.

## 참고 문헌

- [Introduction to Algorithms (CLRS), 21장](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Tarjan, R. E. (1975). Efficiency of a good but not linear set union algorithm. *JACM*, 22(2), 215--225.

## 연습문제

**연습문제 1.**
서로 겹치지 않는 모음(합치기-찾기) 추상 자료형의 핵심 연산 셋과 그 쓰임을 정의하여라.

??? success "연습문제 1 풀이"
    (1) **MakeSet(x)**: 원소 $x$만 담은 새 모음을 만든다. 원소마다 저마다의 조각으로 첫값을 잡는 데 쓴다. (2) **Find(x)**: $x$을 담은 모음의 대표(뿌리)를 돌려준다. $x$이 어느 조각에 드는지 알아내는 데 쓴다. (3) **Union(x, y)**: $x$과 $y$을 담은 모음을 하나로 합친다. 조각 둘을 합치는 데 쓴다. 두 원소가 같은 모음에 있을 때 그리고 오직 그때만 Find(x) == Find(y)이다. $\square$

---

**연습문제 2.**
합치기-찾기가 쪼개기나 지우기 연산을 왜 효율적으로 받쳐 주지 못하는가?

??? success "연습문제 2 풀이"
    합치기-찾기의 나무 짜임은 합치기(Union)와 묻기(Find)를 위해 만들어졌다. 모음을 쪼개려면 부분 나무의 모든 원소를 가려내고 나무 짜임을 다시 세워야 하는데, 이는 $O(n)$의 일이 될 수 있다. 길 줄이기는 나무를 납작하게 만들어 어느 원소가 어느 부분 조각에 있었는지 알아내는 데 필요한 본디 층위를 잃게 하므로 쪼개기를 더 까다롭게 한다. 지우기도 비슷한 말썽이 있다. 곧 나무 가운데서 원소를 없애려면 짜임을 다시 세워야 한다. 쪼개기나 지우기가 필요한 쓰임새에는 링크-컷 나무나 다른 자료 짜임이 더 알맞다. $\square$

---

**연습문제 3.**
최소 뻗은 나무 알고리즘 말고 서로 겹치지 않는 모음 추상 자료형의 실제 쓰임새를 설명하여라.

??? success "연습문제 3 풀이"
    **그림 나누기**: 컴퓨터 시각에서 화소는 처음에 저마다 따로 된 모음이다. 색이 비슷한 이웃 화소를 합쳐 구역을 이룬다. 마지막에 남은 조각이 그림의 조각을 나타낸다. 합치기-찾기는 거의 상수 시간 연산으로 수백만 화소를 효율적으로 다룬다. 다른 쓰임새로는 크러스컬 최소 뻗은 나무, 이어진 조각에 이름표 붙이기, 스며듦 흉내내기, 동치류 셈하기, 방향 없는 그래프의 순환 알아내기가 있다. $\square$

---

**연습문제 4.**
파이썬으로 MakeSet, (길 줄이기 없는) Find, (계급 없는) Union을 구현하여라. 최악의 경우 복잡도를 살펴라.

??? success "연습문제 4 풀이"
    ```python
    parent = list(range(n))  # MakeSet: 원소마다 저마다의 어버이

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry
    ```

    최적화가 없으면 (늘 한쪽에만 붙여) 나무가 이음 목록으로 무너질 수 있다. 최악의 경우 Find은 $O(n)$이 든다. Union과 Find 연산 $n$번의 늘어놓음은 모두 합해 $O(n^2)$이 들 수 있다. $\square$
