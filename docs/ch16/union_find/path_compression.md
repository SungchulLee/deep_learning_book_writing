# 길 줄이기

최적화가 없으면 `find` 연산은 마디에서 뿌리까지 어버이 가리개의 사슬을 지나간다. 나무가 깊으면 이 지나감이 느리다. **길 줄이기**는 `find`마다 나무를 납작하게 만들어 앞으로의 `find`을 빠르게 한다. 곧 뿌리를 찾은 뒤 그 길 위의 모든 마디가 뿌리를 곧바로 가리키도록 고친다. 그러면 그 마디들에 대한 다음 `find`은 $O(1)$이 든다. 연산 $m$번의 늘어놓음에서 (계급으로 합치기와 어우러진) 길 줄이기는 모두 $O(m \cdot \alpha(n))$ 시간을 이룬다.

---

## 1. 세 갈래

흔한 길 줄이기 전략이 셋 있으며 모두 같은 고르게 친 경계를 이룬다:

### 온전한 길 줄이기

뿌리 $r$을 찾은 뒤 $x$에서 $r$까지의 길 위의 모든 마디를 $r$의 곧바른 자식으로 만든다. 이것이 표준 "교과서" 판이며 가장 납작한 나무를 만든다.

```
Before find(5):     After find(5):
    1                    1
    |               / / | \
    2              2  3  4  5
    |
    3
    |
    4
    |
    5
```

### 길 쪼개기

길 위의 마디마다 자기 **할아비**를 가리키도록 고친다. 이는 한 번 훑기로 끝나며(뿌리를 먼저 찾을 필요가 없다) 되풀이 꼴로 구현하기가 조금 더 단순하다.

### 길 반으로 줄이기

길 위의 한 마디 건너 하나씩 자기 할아비를 가리키도록 고친다. 이는 길 쪼개기의 절반만큼 마디를 건드리지만 같은 고르게 친 경계를 이룬다.

---

## 2. 갈래 견주기

| 갈래 | 훑기 횟수 | 고치는 마디 | 구현 |
|---------|--------|---------------|----------------|
| 온전한 줄이기 | 2(뿌리 찾고 나서 고침) | 길 위 전부 | 되돌이나 두 번 훑기 |
| 길 쪼개기 | 1 | 길 위 전부 | 되풀이 고리 하나 |
| 길 반으로 줄이기 | 1 | 길의 절반 | 되풀이 고리 하나 |

셋 모두 계급으로 합치기와 어우러지면 고르게 친 $O(m \cdot \alpha(n))$ 시간을 이룬다. 실전에서의 차이는 무시할 만하다.

---

## 3. 구현

```python
"""
합치기-찾기의 길 줄이기 갈래.

온전한 길 줄이기, 길 쪼개기, 길 반으로
줄이기를 보인다. 계급으로 합치기와 어우러지면 셋 모두
연산마다 고르게 친 O(alpha(n))을 이룬다.
"""

# === 온전한 길 줄이기(되돌이) ===

class UnionFindFullCompression:
    """온전한 길 줄이기를 쓴 합치기-찾기(되돌이 find)."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """온전한 길 줄이기로 뿌리 찾기.

        이 부름 뒤에 x에서 뿌리까지의 길 위의 모든 마디가
        뿌리를 곧바로 가리키게 된다.
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> bool:
        """계급으로 합치기."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True

# === 길 쪼개기 ===

class UnionFindPathSplitting:
    """길 쪼개기를 쓴 합치기-찾기(되풀이 find)."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """길 쪼개기로 뿌리 찾기.

        길 위의 마디마다 자기
        할아비를 가리키도록 한 번 훑기로 고친다.
        """
        while self.parent[x] != x:
            next_x = self.parent[x]
            self.parent[x] = self.parent[next_x]  # 할아비를 가리키게 하기
            x = next_x
        return x

    def union(self, a: int, b: int) -> bool:
        """계급으로 합치기."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True

# === 길 반으로 줄이기 ===

class UnionFindPathHalving:
    """길 반으로 줄이기를 쓴 합치기-찾기(되풀이 find)."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """길 반으로 줄이기로 뿌리 찾기.

        길 위의 한 마디 건너 하나씩
        할아비를 가리키게 된다.
        """
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        """계급으로 합치기."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True

# === 시연 ===

if __name__ == "__main__":
    # 깊은 사슬 세우기: 0 <- 1 <- 2 <- 3 <- 4 <- 5 <- 6 <- 7
    def build_chain(uf_class, n):
        uf = uf_class(n)
        # 사슬을 손수 만들기(합치기를 건너뛰고)
        for i in range(1, n):
            uf.parent[i] = i - 1
        return uf

    n = 8
    print("Parent arrays before and after find(7):")
    print()

    for name, cls in [("Full compression", UnionFindFullCompression),
                       ("Path splitting", UnionFindPathSplitting),
                       ("Path halving", UnionFindPathHalving)]:
        uf = build_chain(cls, n)
        print(f"{name}:")
        print(f"  Before: {uf.parent}")
        root = uf.find(n - 1)
        print(f"  After:  {uf.parent}  (root={root})")
        print()

    # 셋 다 올바른 결과를 내는지 확인
    for cls in [UnionFindFullCompression, UnionFindPathSplitting,
                UnionFindPathHalving]:
        uf = cls(6)
        uf.union(0, 1)
        uf.union(2, 3)
        uf.union(4, 5)
        uf.union(0, 2)
        uf.union(0, 4)
        assert uf.find(5) == uf.find(1)
    print("All variants produce correct connectivity results.")
```

**출력:**
```
Parent arrays before and after find(7):

Full compression:
  Before: [0, 0, 1, 2, 3, 4, 5, 6]
  After:  [0, 0, 0, 0, 0, 0, 0, 0]  (root=0)

Path splitting:
  Before: [0, 0, 1, 2, 3, 4, 5, 6]
  After:  [0, 0, 0, 1, 2, 3, 4, 5]  (root=0)

Path halving:
  Before: [0, 0, 1, 2, 3, 4, 5, 6]
  After:  [0, 0, 0, 2, 2, 4, 4, 6]  (root=0)

All variants produce correct connectivity results.
```

!!! tip "어느 갈래를 쓸까?"
    온전한 길 줄이기는 가장 납작한 나무를 만들고 따져 보기도 가장 단순하다. 길 쪼개기와 반으로 줄이기는 되풀이 꼴이라(되돌이 쌓임이 없고) 메모리에 덜 써서 실전에서 조금 더 빠르다. 대부분의 쓰임새에서 차이는 무시할 만하니 자기 코드에서 가장 또렷한 것을 고르면 된다.

---

## 연습문제

**연습문제 1.**
길 줄이기를 설명하고 마디 5개의 사슬에서 그 효과를 보여라.

??? success "연습문제 1 풀이"
    길 줄이기는 찾기 길 위의 모든 마디가 뿌리를 곧바로 가리키게 한다. 사슬 $0 \leftarrow 1 \leftarrow 2 \leftarrow 3 \leftarrow 4$에서 Find(4) 전에는 길이가 4이다. 길 줄이기를 쓴 Find(4) 뒤에는 마디 1, 2, 3, 4이 모두 뿌리 0을 곧바로 가리킨다. 나무가 별 모양이 된다. 이 마디들에 대한 뒤이은 Find 연산은 $O(1)$이 든다. $\square$

---

**연습문제 2.**
(계급으로 합치기 없이) 길 줄이기만 써도 연산마다 고르게 친 $O(\log n)$ 시간이 됨을 증명하여라.

??? success "연습문제 2 풀이"
    계급으로 합치기가 없으면 나무가 처음에 높을 수 있다(최대 $O(n)$). 그러나 길 줄이기를 쓴 Find마다 물음이 지나간 길을 따라 나무를 납작하게 만든다. 값이 $k$인(길 길이가 $k$인) Find 뒤에는 $k$개 마디가 모두 뿌리를 가리키므로 이 마디들이 끼는 앞으로의 Find은 $O(1)$이 든다. 고르게 친 살피기를 쓰면(이를테면 마디에 외상을 매기면) 연산 $m$번의 전체 값은 $O(m \log n)$이며, 이는 연산마다 고르게 친 $O(\log n)$을 준다. 계급으로 합치기와 어우러지면 $O(\alpha(n))$으로 나아진다. $\square$

---

**연습문제 3.**
길 줄이기를 길 반으로 줄이기와 길 쪼개기라는 다른 두 길 짧게 하기 기법과 견주어라.

??? success "연습문제 3 풀이"
    **길 줄이기**: 길 위의 모든 마디가 뿌리를 곧바로 가리키게 한다. 두 번 훑어야 한다(뿌리를 찾는 한 번, 가리개를 고치는 한 번). **길 반으로 줄이기**: 길 위의 한 마디 건너 하나씩 자기 할아비를 가리키게 한다. 한 번 훑기로 끝나고 더 단순하다. **길 쪼개기**: 모든 마디가 자기 할아비를 가리키게 한다. 한 번 훑기이다. 셋 모두 계급으로 합치기와 어우러지면 고르게 친 $O(\alpha(n))$ 시간을 이룬다. 반으로 줄이기와 쪼개기는 구현이 더 단순하며(한 번 훑기, 되돌이 필요 없음) 실전 성능도 못지않다. $\square$

---

**연습문제 4.**
길 줄이기가 어떤 마디의 계급이라도 바꾸는가? 왜 그런가, 또는 왜 아닌가?

??? success "연습문제 4 풀이"
    아니다. 길 줄이기는 어버이 가리개만 바꾸고 계급은 바꾸지 않는다. 줄인 뒤에는 (예전 자식들이 이제 뿌리를 곧바로 가리키므로) 마디의 계급이 실제 부분 나무 높이보다 클 수 있다. 그래서 계급은 정확한 높이가 아니라 높이의 위 경계이다. 줄일 때 계급을 고치지 않는 것은 일부러 그렇게 짠 것이다. 이러면 Union 연산이 단순하게 남고(계급만 견주면 된다) 고르게 친 살피기도 옳게 남는다. 정확한 높이를 좇으려면 줄일 때마다 많은 마디의 높이를 고쳐야 해서 비쌀 것이다. $\square$

## 정리하며

이 마당은 세 갈래、갈래 견주기、구현을 차례로 짚었다.

**참고 문헌**

- Tarjan, R. E., & van Leeuwen, J. (1984). Worst-case analysis of set union algorithms. *Journal of the ACM*, 31(2), 245-281.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 19장. MIT Press.
