# 범위 갱신

실제 문제 가운데 이어진 데이터 덩어리를 한꺼번에 고쳐야 하는 것이 많다. 반 전체의 시험 점수를 조정하거나 화소 한 줄에 밝기 값을 더하는 일을 생각해 보라. 점 갱신 구간 트리는 원소를 하나씩 $O(\log n)$에 다루므로 원소 $k$개를 고치면 $O(k \log n)$이 든다. **게으른 전파**를 등에 업은 범위 갱신은 아무 구간 $[l, r]$이든 그 길이와 무관하게 $O(\log n)$ 연산 한 번으로 줄인다.

## 범위 더하기 갱신

가장 흔한 범위 갱신은 $[l, r]$의 모든 원소에 값 $\delta$을 더하는 것이다.

$$
a[i] \leftarrow a[i] + \delta \quad \text{for all } l \leq i \leq r
$$

구간 트리에서 범위가 $[l, r]$ 안에 온전히 드는 노드는 갱신을 곧바로 받는다. 일부만 겹치는 노드는 게으른 전파로 갱신을 자식에 밀어 넣는다. 이 미룬 전파가 연산마다의 비용을 로그로 지킨다.

## 알고리즘

범위 갱신 절차는 $[lo, hi]$을 덮는 노드를 들러 세 경우 가운데 하나로 갈라진다.

1. **겹치지 않음** ($r < lo$이거나 $hi < l$): 곧바로 돌아간다.
2. **온전히 담김** ($l \leq lo$이고 $hi \leq r$): 노드에 담긴 합에 $\delta \cdot (hi - lo + 1)$을 더하고 게으른 꼬리표에 $\delta$을 적는다. 재귀하지 않고 돌아간다.
3. **일부만 겹침**: 이미 있는 게으른 꼬리표를 두 자식에 밀어 내리고 두 자식으로 재귀한 뒤 고쳐진 자식에서 노드의 값을 다시 셈한다.

노드의 게으른 꼬리표는 아직 그 자식에 퍼지지 않은, 원소마다의 미룬 더하기를 나타낸다. 밀어 내릴 때마다 꼬리표가 한 층 더 깊이 옮겨 가고 지금 노드에서는 지워진다.

## 더 간단한 대안

(범위 질의 없이) 범위 갱신과 **점 질의**만 필요하면 펜윅 트리(BIT)를 등에 업은 **차이 배열**이 같은 $O(\log n)$ 한계로 더 간단한 코드로 푼다. 다만 범위 갱신과 범위 질의가 모두 필요하면 차이 배열만으로는 범위 합 질의에 효율적으로 답할 수 없으므로 게으른 구간 트리가 표준이다.

## 구현

```python
"""
범위 더하기 갱신을 위해 게으른 전파를 쓰는 구간 트리.

연산 둘을 저마다 O(log n)에 받쳐 준다.
  - range_update(l, r, delta): [l, r]의 모든 원소에 delta를 더한다
  - range_query(l, r): [l, r]의 원소의 합을 돌려준다

노드마다의 게으른 꼬리표는 아직 자식에 밀어 넣지 않은
원소별 미룬 더하기를 담는다.
"""


# === 범위 갱신을 하는 구간 트리 ===

class RangeUpdateSegTree:
    """범위 더하기와 범위 합을 위해 게으른 전파를 쓰는 구간 트리."""

    def __init__(self, data: list):
        self.n = len(data)
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        if self.n > 0:
            self._build(data, 1, 0, self.n - 1)

    def _build(self, data: list, node: int, lo: int, hi: int) -> None:
        if lo == hi:
            self.tree[node] = data[lo]
            return
        mid = (lo + hi) // 2
        self._build(data, 2 * node, lo, mid)
        self._build(data, 2 * node + 1, mid + 1, hi)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def _push_down(self, node: int, lo: int, hi: int) -> None:
        """게으른 꼬리표를 자식으로 퍼뜨린다."""
        if self.lazy[node] != 0 and lo != hi:
            mid = (lo + hi) // 2
            left, right = 2 * node, 2 * node + 1

            self.tree[left] += self.lazy[node] * (mid - lo + 1)
            self.lazy[left] += self.lazy[node]

            self.tree[right] += self.lazy[node] * (hi - mid)
            self.lazy[right] += self.lazy[node]

            self.lazy[node] = 0

    def range_update(self, node: int, lo: int, hi: int,
                     l: int, r: int, delta: int) -> None:
        """[l, r]의 모든 원소에 delta를 더한다.

        인수:
            node: 트리 배열에서 지금 자리.
            lo, hi: 이 노드가 덮는 범위.
            l, r: 갱신 범위.
            delta: 원소마다 더할 값.
        """
        if r < lo or hi < l:
            return
        if l <= lo and hi <= r:
            self.tree[node] += delta * (hi - lo + 1)
            self.lazy[node] += delta
            return
        self._push_down(node, lo, hi)
        mid = (lo + hi) // 2
        self.range_update(2 * node, lo, mid, l, r, delta)
        self.range_update(2 * node + 1, mid + 1, hi, l, r, delta)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def range_query(self, node: int, lo: int, hi: int,
                    l: int, r: int) -> int:
        """[l, r]의 원소의 합을 돌려준다."""
        if r < lo or hi < l:
            return 0
        if l <= lo and hi <= r:
            return self.tree[node]
        self._push_down(node, lo, hi)
        mid = (lo + hi) // 2
        return (self.range_query(2 * node, lo, mid, l, r)
                + self.range_query(2 * node + 1, mid + 1, hi, l, r))

    def point_query(self, idx: int) -> int:
        """a[idx]의 지금 값을 돌려준다."""
        return self.range_query(1, 0, self.n - 1, idx, idx)


# === 시연 ===

if __name__ == "__main__":
    data = [1, 3, 5, 7, 9, 11]
    n = len(data)
    st = RangeUpdateSegTree(data)

    print(f"Original: {data}")
    print(f"Sum [0,5] = {st.range_query(1, 0, n-1, 0, 5)}")
    print()

    # 범위 갱신: [1, 4]에 10을 더한다
    print("Range update: add 10 to [1, 4]")
    st.range_update(1, 0, n - 1, 1, 4, 10)
    print(f"Sum [0,5] = {st.range_query(1, 0, n-1, 0, 5)}")
    print(f"Sum [1,4] = {st.range_query(1, 0, n-1, 1, 4)}")
    print(f"a[0] = {st.point_query(0)}")
    print(f"a[2] = {st.point_query(2)}")
    print(f"a[5] = {st.point_query(5)}")
    print()

    # 또 다른 범위 갱신: [0, 2]에 5를 더한다
    print("Range update: add 5 to [0, 2]")
    st.range_update(1, 0, n - 1, 0, 2, 5)
    print(f"Sum [0,5] = {st.range_query(1, 0, n-1, 0, 5)}")
    print(f"a[0] = {st.point_query(0)}")
    print(f"a[1] = {st.point_query(1)}")
    print(f"a[2] = {st.point_query(2)}")
```

**출력:**
```
Original: [1, 3, 5, 7, 9, 11]
Sum [0,5] = 36

Range update: add 10 to [1, 4]
Sum [0,5] = 76
Sum [1,4] = 64
a[0] = 1
a[2] = 15
a[5] = 11

Range update: add 5 to [0, 2]
Sum [0,5] = 91
a[0] = 6
a[1] = 18
a[2] = 20
```

## 복잡도

| 연산 | 시간 | 공간 |
|-----------|------|-------|
| 범위 갱신 ($[l, r]$에 $\delta$ 더하기) | $O(\log n)$ | 스택 $O(\log n)$ |
| 갱신 뒤의 범위 질의 | $O(\log n)$ | 스택 $O(\log n)$ |
| 전체 공간 | — | $O(n)$ (트리 배열과 게으른 배열) |

연산마다 노드를 많아야 $O(\log n)$개 들른다. 재귀 깊이는 트리의 높이 $\lceil \log_2 n \rceil$으로 한계 지어진다.

## 범위 덮어쓰기

쓸모 있는 변형은 $[l, r]$의 모든 원소에 $\delta$을 더하는 대신 값 $v$으로 **덮어쓰는** 것이다. 핵심 차이는 게으른 꼬리표의 뜻에 있다. 이제 꼬리표가 덮어쓸 값을 담고, (파이썬에서는 대개 `None`, 값이 음이 아니면 $-1$인) 파수 값이 "미룬 덮어쓰기 없음"과 "0으로 덮어쓰기"를 가른다. 밀어 내릴 때 자식의 값은 더해지는 것이 아니라 **바뀐다**.

!!! warning "서로 다른 연산 이어 붙이기"
    같은 트리에 범위 더하기와 범위 덮어쓰기가 함께 있으면 게으른 꼬리표가 두 성분을 모두 담아야 한다. 밀어 내릴 때 (덮어쓰는) 덮어쓰기를 먼저 적용하고 그 위에 더하기를 적용한다. 이 순서를 뒤집으면 틀린 결과가 나온다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.


## 연습문제

**연습문제 1.**
범위 갱신의 짜임을 설명하고 질의와 갱신의 복잡도를 밝혀라.

??? success "연습문제 1 풀이"
    이 짜임은 위계적 분해를 이용해 일차보다 빠른 질의 시간을 이룬다. 대개 질의와 갱신이 모두 $O(\log n)$이고 세우는 데 $O(n)$이나 $O(n\log n)$이 든다.

---

**연습문제 2.**
입력 $[3, 1, 4, 1, 5, 9, 2, 6]$으로 범위 갱신을 세워라. 마지막 짜임을 보여라.

??? success "연습문제 2 풀이"
    세우기 알고리즘을 적용하며 중간 상태를 보여라. 트리 짜임이면 트리를 그려라. 배열에 바탕한 짜임이면 부모-자식 관계를 덧붙여 배열의 내용을 보여라.

---

**연습문제 3.**
질의 연산이 올바른 어떤 질의 범위에 대해서도 옳은 결과를 돌려줌을 증명하라.

??? success "연습문제 3 풀이"
    증명에는 대개 트리의 높이에 대한 귀납법을 쓴다. 노드마다 질의가 한 부분 트리 안에 온전히 들거나(재귀한다) 두 부분 트리에 걸친다(부분 결과를 모은다). 모으는 함수(합, 최솟값, 최댓값)가 결합적이므로 올바로 모아진다. $\square$

---

**연습문제 4.**
범위 갱신가 질의마다의 계산을 $O(n)$에서 $O(\log n)$으로 빠르게 하는 딥러닝 응용을 설명하라.

??? success "연습문제 4 풀이"
    응용으로는 누적 어텐션 가중치를 위한 접두사 합 질의, 길이가 제각각인 수열에서 효율적인 풀링을 위한 범위 질의, 검색 증강 생성을 위한 최근접 이웃 찾기, 3차원 딥러닝의 점 구름 처리를 위한 공간 색인이 있다.