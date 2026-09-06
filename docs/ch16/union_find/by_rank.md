# 계급으로 합치기

어수룩한 합치기-찾기 구현에서 `union` 연산은 나무의 짜임을 살피지 않고 한 나무의 뿌리를 다른 나무의 자식으로 붙인다. 이러면 긴 사슬(높이 $n - 1$인 무너진 나무)이 생겨 `find` 연산이 $O(n)$ 시간이 든다. **계급으로 합치기**는 늘 더 낮은 나무를 더 높은 나무의 뿌리 아래에 붙여 이를 막고 나무 높이를 $O(\log n)$으로 묶는다. 길 줄이기와 어우러지면 거의 최적인 고르게 친 $O(\alpha(n))$ 경계를 준다.

## 계급이란 무엇인가?

마디의 **계급**은 그 부분 나무 높이의 위 경계이다. 처음에는 모든 마디가 계급 0인 저마다의 뿌리이다(마디 하나짜리 나무의 높이는 0이다). 나무 둘을 합칠 때는:

- **계급이 다르면** 계급이 작은 뿌리가 계급이 큰 뿌리의 자식이 된다. 합쳐진 나무의 높이가 늘지 않으므로 큰 뿌리의 계급은 그대로이다.
- **계급이 같으면** 한 뿌리가 다른 뿌리의 자식이 되고 살아남은 뿌리의 계급이 1 커진다. 높이 $r$인 나무를 높이 $r$인 다른 나무 아래에 붙이면 높이 $r + 1$인 나무가 된다.

## 계급의 성질

계급으로 합치기는 여러 중요한 불변식을 지킨다:

1. 계급 $r$인 마디에는 자손이 적어도 $2^r$개 있다. 이는 귀납법으로 따라 나온다. 곧 계급이 커지는 것은 계급 $r$인 부분 나무 둘이 합쳐질 때뿐이며 그때 최소 크기가 곱절이 된다.
2. $2^r \leq n$에서 $r \leq \log_2 n$이 따라 나오므로 최대 계급은 많아야 $\lfloor \log_2 n \rfloor$이다.
3. 계급이 $\geq r$인 마디는 많아야 $n / 2^r$개이며, 이는 계급 층마다 마디가 몇 개나 있을 수 있는지를 묶는다.

이 성질들 덕분에 길 줄이기가 없어도 `find` 연산이 많아야 $O(\log n)$개의 어버이 가리개를 지나감이 보장된다.

## 계급으로 합치기와 크기로 합치기의 견줌

다른 길로 **크기로 합치기**가 있다. 이는 (마디 수로) 더 작은 나무를 더 큰 나무의 뿌리 아래에 붙인다. 두 전략 모두 나무 높이 $O(\log n)$을 이룬다. 차이는 미묘하다:

| 성질 | 계급으로 합치기 | 크기로 합치기 |
|----------|--------------|---------------|
| 담는 값 | 높이의 위 경계 | 정확한 부분 나무 마디 수 |
| 새로 고치는 규칙 | 계급이 같을 때 1 늘림 | 크기를 더함 |
| 길 줄이기를 쓰면 | 계급이 높이를 넘겨 잡을 수 있음 | 크기는 정확한 채로 |
| 고르게 친 경계 | $O(\alpha(n))$ | $O(\alpha(n))$ |

계급의 성질이 증명하기 더 단순하므로 이론으로 살필 때는 계급으로 합치기를 즐겨 쓴다. 크기 값은 (이를테면 조각의 크기를 알려 주는) 덤 쓰임이 있어 실전에서는 크기로 합치기를 즐겨 쓰기도 한다.

## 구현

```python
"""
합치기-찾기의 계급으로 합치기 최적화.

더 낮은 나무를 더 높은 나무의 뿌리 아래에 붙여 나무의
높이를 고르게 지킨다. 길 줄이기 없이도 find이 O(log n)이고,
길 줄이기를 쓰면 고르게 친 O(alpha(n))이다.
"""


# === 계급으로 합치기를 쓴 합치기-찾기 ===

class UnionFindByRank:
    """계급으로 합치기와 온전한 길 줄이기를 쓴 합치기-찾기."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """길 줄이기로 x의 뿌리 찾기."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> bool:
        """계급으로 합치기. 합침이 일어났으면 True를 돌려준다."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        # 계급이 작은 나무를 계급이 큰 뿌리 아래에 붙이기
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


# === 크기로 합치기를 쓴 합치기-찾기(견줌) ===

class UnionFindBySize:
    """크기로 합치기와 온전한 길 줄이기를 쓴 합치기-찾기."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        """길 줄이기로 x의 뿌리 찾기."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> bool:
        """크기로 합치기. 합침이 일어났으면 True를 돌려준다."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True

    def component_size(self, x: int) -> int:
        """x을 담은 조각의 크기를 돌려준다."""
        return self.size[self.find(x)]


# === 시연 ===

if __name__ == "__main__":
    print("=== Union by Rank ===")
    uf_rank = UnionFindByRank(8)
    merges = [(0, 1), (2, 3), (4, 5), (6, 7),
              (0, 2), (4, 6), (0, 4)]
    for a, b in merges:
        uf_rank.union(a, b)
        print(f"union({a},{b}): ranks={uf_rank.rank}")

    print()
    print("=== Union by Size ===")
    uf_size = UnionFindBySize(8)
    for a, b in merges:
        uf_size.union(a, b)
        print(f"union({a},{b}): sizes={uf_size.size}")

    print()
    print(f"Component size of 5: {uf_size.component_size(5)}")

    # 둘이 같은 이어짐을 내는지 보이기
    for i in range(8):
        for j in range(i + 1, 8):
            rank_conn = uf_rank.find(i) == uf_rank.find(j)
            size_conn = uf_size.find(i) == uf_size.find(j)
            assert rank_conn == size_conn
    print("Both methods produce identical connectivity.")
```

**출력:**
```
=== Union by Rank ===
union(0,1): ranks=[1, 0, 0, 0, 0, 0, 0, 0]
union(2,3): ranks=[1, 0, 1, 0, 0, 0, 0, 0]
union(4,5): ranks=[1, 0, 1, 0, 1, 0, 0, 0]
union(6,7): ranks=[1, 0, 1, 0, 1, 0, 1, 0]
union(0,2): ranks=[2, 0, 1, 0, 1, 0, 1, 0]
union(4,6): ranks=[2, 0, 1, 0, 2, 0, 1, 0]
union(0,4): ranks=[3, 0, 1, 0, 2, 0, 1, 0]

=== Union by Size ===
union(0,1): sizes=[2, 1, 1, 1, 1, 1, 1, 1]
union(2,3): sizes=[2, 1, 2, 1, 1, 1, 1, 1]
union(4,5): sizes=[2, 1, 2, 1, 2, 1, 1, 1]
union(6,7): sizes=[2, 1, 2, 1, 2, 1, 2, 1]
union(0,2): sizes=[4, 1, 2, 1, 2, 1, 2, 1]
union(4,6): sizes=[4, 1, 2, 1, 4, 1, 2, 1]
union(0,4): sizes=[8, 1, 2, 1, 4, 1, 2, 1]

Component size of 5: 8
Both methods produce identical connectivity.
```

## 참고 문헌

- Tarjan, R. E. (1975). Efficiency of a good but not linear set union algorithm. *Journal of the ACM*, 22(2), 215-225.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 19장. MIT Press.

## 연습문제

**연습문제 1.**
계급으로 합치기에서 계급을 정의하고 그것이 나무 높이를 어떻게 다스리는지 설명하여라.

??? success "연습문제 1 풀이"
    마디 $x$의 **계급**은 $x$을 뿌리로 하는 부분 나무 높이의 위 경계이다. 처음에는 모든 마디의 계급이 0이다. 뿌리가 $r_1$과 $r_2$인 나무 둘을 합칠 때 $\text{rank}(r_1) < \text{rank}(r_2)$이면 $r_1$을 $r_2$의 자식으로 만든다(계급은 그대로). $\text{rank}(r_1) > \text{rank}(r_2)$이면 $r_2$을 $r_1$의 자식으로 만든다. 같으면 아무 쪽이나 뿌리로 고르고 그 계급을 1 늘린다. 이러면 나무 높이가 많아야 $\log_2 n$이 되어 Find 연산이 효율적으로 남는다. $\square$

---

**연습문제 2.**
계급으로 합치기로 만든, 계급이 $r$인 나무에 마디가 적어도 $2^r$개 있음을 증명하여라.

??? success "연습문제 2 풀이"
    계급에 대한 귀납법으로 보인다. 바닥: 계급 0인 나무의 마디는 $1 = 2^0$개다. 귀납 걸음: 계급 $r$인 나무는 계급 $r-1$인 나무 둘을 합쳐야만 생긴다. 가정에 따라 저마다 마디가 적어도 $2^{r-1}$개 있다. 합쳐진 나무의 마디는 적어도 $2^{r-1} + 2^{r-1} = 2^r$개다. 마디 $n$개의 나무는 계급이 많아야 $\log_2 n$이므로 길 줄이기 없는 Find은 $O(\log n)$ 시간이 든다. $\square$

---

**연습문제 3.**
(길 줄이기 없이) 계급으로 합치기만 써도 Find 연산마다 $O(\log n)$이 되는 까닭을 설명하여라.

??? success "연습문제 3 풀이"
    Find은 마디에서 뿌리까지의 길을 지나간다. 이 길의 길이는 나무 높이로 묶인다. 계급 성질(계급 $\leq \log_2 n$)에 따라 나무 높이는 많아야 $\log_2 n$이다. 그러므로 Find마다 $O(\log n)$ 시간이 든다. 길 줄이기를 더하면 고르게 친 값이 $O(\alpha(n))$으로 떨어지지만, 그것 없이도 $O(\log n)$은 어수룩한 Union의 최악 $O(n)$보다 크게 나아진 것이다. $\square$

---

**연습문제 4.**
계급으로 합치기와 크기로 합치기를 견주어라. 점근 복잡도가 같은가?

??? success "연습문제 4 풀이"
    **크기로 합치기**: 더 작은 나무를 더 큰 나무의 뿌리 아래에 붙인다. **계급으로 합치기**: 계급이 낮은 나무를 계급이 높은 뿌리 아래에 붙인다. 둘 다 나무 높이 $O(\log n)$을 이룬다. 길 줄이기를 쓰면 둘 다 연산마다 고르게 친 $O(\alpha(n))$을 준다. 핵심 차이는 계급이 높이의 위 경계인 반면(길 줄이기 뒤에는 넘겨 잡을 수 있다) 크기는 정확하다는 것이다. 실전에서는 둘의 성능이 비슷하다. 계급이 같은 나무를 합칠 때만 계급을 새로 고치면 되므로 계급으로 합치기가 조금 더 단순하다. $\square$
