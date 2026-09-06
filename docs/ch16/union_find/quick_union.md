# 빠른 합치기

서로 겹치지 않는 모음의 빠른 찾기 방식에서는 원소마다 자기 뿌리를 곧바로 담아 `find`이 $O(1)$에 돌아가지만, 한 조각의 모든 원소를 고쳐야 하므로 `union`은 $O(n)$이 든다. 빠른 합치기는 이 주고받음을 뒤집는다. 곧 원소마다 뿌리 있는 나무에서 자기 **어버이**만 담으므로 `union`은 뿌리 하나를 다른 뿌리에 잇기만 해서 $O(1)$이지만, `find`은 나무를 거슬러 뿌리까지 올라간다. 그 값은 `find`이 이제 나무 높이에 비례하는 시간이 든다는 것이며, 더 손보지 않으면 최악의 경우 $O(n)$까지 나빠질 수 있다.

## 나무로 나타내기

빠른 합치기는 서로 겹치지 않는 모음마다 **뿌리 있는 나무**로 나타낸다. 원소 $x$마다 어버이 가리개 $\text{parent}[x]$을 지닌다. 나무의 뿌리는 $\text{parent}[r] = r$을 만족하는 오직 하나뿐인 원소이다. 두 원소가 같은 뿌리를 나눠 가질 때 그리고 오직 그때만 같은 모음에 든다.

처음에는 원소마다 저마다의 뿌리이다:

$$
\text{parent}[x] = x \quad \text{for all } x \in \{0, 1, \dots, n-1\}
$$

## 핵심 연산

### Find

원소 $x$이 어느 모음에 드는지 알아내려면 $x$에서 뿌리에 이를 때까지 어버이 가리개를 따라간다:

$$
\text{find}(x) = \begin{cases} x & \text{if } \text{parent}[x] = x \\ \text{find}(\text{parent}[x]) & \text{otherwise} \end{cases}
$$

`find`의 값은 $O(d)$이며 여기서 $d$은 그 나무에서 $x$의 깊이이다.

### Union

원소 $a$과 $b$을 담은 모음을 합치려면 저마다의 뿌리를 찾아 한 뿌리가 다른 뿌리를 가리키게 한다:

$$
\text{union}(a, b): \quad r_a = \text{find}(a),\; r_b = \text{find}(b),\; \text{parent}[r_b] \leftarrow r_a
$$

이는 `find` 두 번의 값 말고는 $O(1)$이 든다.

## 최악의 경우 분석

균형을 잡는 전략이 없으면 합치기 $n - 1$번의 늘어놓음이 높이 $n - 1$의 무너진 사슬을 만들 수 있다. 이를테면 어수룩한 판에서 $\text{union}(0,1),\, \text{union}(1,2),\, \dots,\, \text{union}(n{-}2, n{-}1)$을 하면 길 $0 \to 1 \to 2 \to \cdots \to n{-}1$ 하나가 생긴다. 그다음 원소 $0$에 대한 `find`은 마디 $n$개를 모두 지나간다.

따라서 원소 $n$개에 대한 연산 $m$번의 늘어놓음에서 어수룩한 빠른 합치기는 최악의 값이 $O(mn)$이다.

## 최적화

핵심 최적화 둘이 연산마다의 고르게 친 값을 거의 $O(1)$까지 낮춘다:

**계급으로 합치기.** 더 낮은 나무를 더 높은 나무의 뿌리 아래에 붙인다. 마디마다 계급(높이의 위 경계)을 담는다. 두 뿌리의 계급이 같으면 새 뿌리의 계급이 1 커진다. 이러면 나무 높이가 많아야 $O(\log n)$임이 보장된다.

**길 줄이기.** `find` 동안 들른 모든 마디가 뿌리를 곧바로 가리키게 한다. 이러면 앞으로의 물음을 위해 나무가 납작해진다. **길 쪼개기**라는 더 가벼운 갈래는 들른 마디마다 자기 할아비를 가리키게 하여 더 단순한 고리로 비슷한 고르게 친 성능을 이룬다.

두 최적화를 모두 쓰면 원소 $n$개에 대한 연산 $m$번의 어떤 늘어놓음도 $O(m \, \alpha(n))$ 시간에 돌아간다. 여기서 $\alpha$은 거꿀 애커만 함수이며 아주 느리게 자라 사실상 상수이다(실전에서 다룰 만한 모든 $n$에 대해 $\alpha(n) \le 4$).

## 구현

다음 구현은 계급으로 합치기와 길 쪼개기를 어우른 것으로 실전에서 가장 흔히 쓰는 꼴이다.

```python
"""
계급으로 합치기와 길 쪼개기를 쓴 빠른 합치기.

서로 겹치지 않는 모음의 빠른 합치기 방식을 보인다. 여기서는
원소마다 어버이 가리개를 담고 find이 나무를 거슬러 뿌리까지 올라간다.
최적화 둘이 나무를 거의 납작하게 지킨다:
  - 계급으로 합치기: 더 낮은 나무를 더 높은 뿌리 아래에 붙인다.
  - 길 쪼개기: find 동안 마디마다 자기 할아비를 가리키게 다시 잇는다.
"""

# === 합치기-찾기 클래스 ===

class UnionFind:
    """빠른 합치기를 쓰는, 서로 겹치지 않는 모음 자료 짜임."""

    def __init__(self, n: int):
        """홑원소 모음 {0}, {1}, ..., {n-1} 만들기."""
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """x을 담은 모음의 뿌리를 돌려준다(길 쪼개기와 함께)."""
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # 길 쪼개기
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        """a과 b을 담은 모음 합치기. 이미 같은 모음이면 False를 돌려준다."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        # 계급으로 합치기: 계급이 작은 나무를 계급이 큰 뿌리 아래에 붙인다
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True

    def connected(self, a: int, b: int) -> bool:
        """a과 b이 같은 모음에 드는지 살핀다."""
        return self.find(a) == self.find(b)


# === 시연 ===

if __name__ == "__main__":
    uf = UnionFind(6)

    # 짝 셋 세우기
    print(uf.union(0, 1))  # True  — {0,1}
    print(uf.union(2, 3))  # True  — {2,3}
    print(uf.union(4, 5))  # True  — {4,5}

    # 짝 둘 합치기
    print(uf.union(1, 3))  # True  — {0,1,2,3}

    # 이어짐 묻기
    print(f"0 and 3 connected: {uf.connected(0, 3)}")  # 참
    print(f"0 and 4 connected: {uf.connected(0, 4)}")  # False
```

**출력:**

```
참
참
참
참
0 and 3 connected: True
0 and 4 connected: False
```

처음 세 번의 합치기는 저마다 새 짝을 만든다. 네 번째 합치기는 $1$과 $3$을 담은 모음을 합쳐 $\{0,1\}$과 $\{2,3\}$을 한 조각으로 잇는다. 물음들은 이제 $0$과 $3$이 이어졌고 $0$과 $4$은 여전히 서로 다른 조각에 있음을 확인해 준다.

## 복잡도 요약

| 연산 | 어수룩한 빠른 합치기 | 계급 + 길 줄이기를 쓸 때 |
|-----------|:-----------------:|:----------------------------:|
| `find`    | $O(n)$            | 고르게 친 $O(\alpha(n))$     |
| `union`   | $O(n)$            | 고르게 친 $O(\alpha(n))$     |
| 공간      | $O(n)$            | $O(n)$                       |

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), 21장: Data Structures for Disjoint Sets.
- Tarjan, R. E. (1975). Efficiency of a good but not linear set union algorithm. *Journal of the ACM*, 22(2), 215--225.

## 연습문제

**연습문제 1.**
빠른 합치기 방식을 설명하여라. 모음마다 어떻게 나타내는가?

??? success "연습문제 1 풀이"
    빠른 합치기는 어버이 배열 `parent[0..n-1]`으로 모음마다 나무로 나타낸다. 처음에는 `parent[i] = i`이다(원소마다 저마다의 뿌리이다). **Find(x)**: 뿌리(`parent[x] = x`인 $x$)에 이를 때까지 어버이 가리개를 따라간다. **Union(x, y)**: 뿌리 $r_x$과 $r_y$을 찾아 `parent[r_x] = r_y`으로 둔다. 이러면 한 나무가 다른 나무의 부분 나무가 된다. $\square$

---

**연습문제 2.**
최적화가 없는 빠른 합치기가 높이 $n - 1$의 나무를 만들 수 있음을 보여라. 구체적인 연산 늘어놓음을 들어라.

??? success "연습문제 2 풀이"
    Union을 사슬로 한다. 곧 Union(0, 1), Union(1, 2), Union(2, 3), ..., Union($n-2$, $n-1$)이다. Union마다 첫 뿌리를 둘째 뿌리 아래에 붙이면 나무가 $0 \to 1 \to 2 \to \cdots \to n-1$이 된다. Find(0)은 사슬 전체를 지나가므로 $O(n)$이다. 그런 Find $n$번의 늘어놓음은 모두 $O(n^2)$이 든다. $\square$

---

**연습문제 3.**
빠른 찾기, 빠른 합치기, 그리고 두 최적화를 모두 쓴 빠른 합치기의 점근 복잡도를 견주어라.

??? success "연습문제 3 풀이"
    | 연산 | 빠른 찾기 | 빠른 합치기 | 빠른 합치기 + 계급 + 길 줄이기 |
    |---|---|---|---|
    | MakeSet | $O(1)$ | $O(1)$ | $O(1)$ |
    | Find | $O(1)$ | 최악 $O(n)$ | 고르게 친 $O(\alpha(n))$ |
    | Union | $O(n)$ | 최악 $O(n)$ | 고르게 친 $O(\alpha(n))$ |
    | 원소 $n$개에 연산 $m$번 | $O(mn)$ | $O(mn)$ | $O(m \cdot \alpha(n))$ |

    계급과 길 줄이기를 모두 쓴 빠른 합치기는 거의 최적이다. 곧 연산 $m$번의 어떤 늘어놓음에도 $O(m \cdot \alpha(n))$이다. $\square$

---

**연습문제 4.**
최적화의 출발점으로 빠른 찾기보다 빠른 합치기를 즐겨 쓰는 까닭은 무엇인가?

??? success "연습문제 4 풀이"
    빠른 합치기의 나무 짜임은 계급으로 합치기와 길 줄이기를 모두 쓸 수 있게 하며, 이 둘이 함께 거의 상수인 고르게 친 시간을 이룬다. 빠른 찾기의 납작한 배열 짜임은 길 줄이기를 받쳐 주지 못한다(줄일 어버이-자식 관계가 없다). 빠른 찾기는 Find이 $O(1)$이지만 무게 붙인 갈래를 써도 $O(n)$인 Union을 고르게 친 $O(\log n)$ 아래로 낫게 할 수 없다. 두 최적화를 모두 쓴 빠른 합치기는 두 연산 모두 $O(\alpha(n))$을 이루며, 이는 어떤 빠른 찾기 갈래보다도 낫다. $\square$
