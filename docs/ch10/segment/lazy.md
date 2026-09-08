# 게으른 전파

표준 구간 트리는 **점 갱신**을 $O(\log n)$에 다룬다. 그런데 범위 $[l, r]$ 전체를 한꺼번에 고치려면 어떨까? 다듬지 않으면 범위 갱신이 노드를 $O(n)$개까지 건드린다. **게으른 전파**는 자식으로의 갱신을 정말 필요할 때까지 미루어 이를 풀고 범위 갱신과 범위 질의를 모두 $O(\log n)$에 해낸다.

---

## 1. 순진한 범위 갱신의 문제

범위 $[l, r]$의 모든 원소에 값 $\delta$을 더하고 싶다고 하자. 순진한 방법은 $r - l + 1$개 자리를 하나씩 고쳐 모두 $O((r - l + 1) \cdot \log n)$이 든다. 범위가 크면 $O(n \log n)$에 다가가 맨바닥부터 다시 셈하는 것보다 별로 낫지 않다.

---

## 2. 게으른 생각

갱신을 모든 자손 노드에 곧바로 밀어 넣는 대신, 범위가 $[l, r]$ 안에 온전히 드는 가장 높은 노드에 미룬 갱신을 담아 둔다. 이 "게으른" 꼬리표가 아직 자식에 적용하지 않은 것을 적어 둔다. 뒤이은 질의나 갱신이 자식을 들러야 할 때 먼저 게으른 꼬리표를 자식으로 **밀어 내린** 뒤 나아간다.

!!! note "미룬 일의 원리"
    게으른 전파는 자료 구조 설계의 일반 원리를 따른다. 결과가 필요할 때까지 일을 미루는 것이다. 그러면 갱신의 비용이 앞으로의 연산에 걸쳐 분할 상환된다.

---

## 3. 밀어 내리기 얼개

게으른 꼬리표 $\text{lazy}[v]$을 가지고 범위 $[lo, hi]$을 덮는 노드 $v$의 밀어 내리기 연산은 다음과 같다.

1. $\text{mid} = \lfloor (lo + hi) / 2 \rfloor$을 셈한다.
2. **왼쪽 자식**($2v$, $[lo, \text{mid}]$을 덮음): 그 값에 $\text{lazy}[v] \cdot (\text{mid} - lo + 1)$을 더하고 그 게으른 꼬리표에 $\text{lazy}[v]$을 더한다.
3. **오른쪽 자식**($2v + 1$, $[\text{mid}+1, hi]$을 덮음): 그 값에 $\text{lazy}[v] \cdot (hi - \text{mid})$을 더하고 그 게으른 꼬리표에 $\text{lazy}[v]$을 더한다.
4. $v$의 게으른 꼬리표를 지운다: $\text{lazy}[v] = 0$.

---

## 4. 범위 갱신 알고리즘

$[l, r]$의 모든 원소에 $\delta$을 더하려면 다음과 같이 한다.

1. 노드의 범위 $[lo, hi]$이 $[l, r]$과 겹치지 않으면 돌아간다.
2. $[lo, hi] \subseteq [l, r]$이면 노드의 값에 $\delta \cdot (hi - lo + 1)$을 더하고 게으른 꼬리표에 $\delta$을 더한 뒤 돌아간다.
3. 그렇지 않으면 게으른 꼬리표를 밀어 내리고 두 자식을 재귀적으로 고친 뒤 자식에서 노드의 값을 다시 셈한다.

---

## 5. 게으른 꼬리표를 쓰는 범위 질의

$[l, r]$의 합을 물으려면 다음과 같이 한다.

1. 노드의 범위가 $[l, r]$과 겹치지 않으면 0을 돌려준다.
2. 노드의 범위가 $[l, r]$ 안에 온전히 들면 노드의 값을 돌려준다.
3. 그렇지 않으면 게으른 꼬리표를 먼저 밀어 내리고 두 자식에 재귀적으로 묻는다.

재귀 전에 밀어 내리면 자식에 닿을 때 그 값이 최신임이 보장된다.

---

## 6. 구현

```python
"""
범위 갱신을 위해 게으른 전파를 쓰는 구간 트리.

필요할 때만 퍼뜨리는 미룬(게으른) 갱신으로 O(log n)의
범위 더하기와 O(log n)의 범위 합 질의를
받쳐 준다.
"""

# === 게으른 전파를 쓰는 구간 트리 ===

class LazySegTree:
    """범위 갱신과 범위 질의를 받쳐 주는 구간 트리."""

    def __init__(self, data: list):
        """입력 배열로 O(n)에 세운다."""
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
        """게으른 꼬리표를 노드에서 자식으로 퍼뜨린다."""
        if self.lazy[node] != 0:
            mid = (lo + hi) // 2
            left, right = 2 * node, 2 * node + 1

            # 왼쪽 자식을 고친다
            self.tree[left] += self.lazy[node] * (mid - lo + 1)
            self.lazy[left] += self.lazy[node]

            # 오른쪽 자식을 고친다
            self.tree[right] += self.lazy[node] * (hi - mid)
            self.lazy[right] += self.lazy[node]

            # 이 노드의 게으른 꼬리표를 지운다
            self.lazy[node] = 0

    def range_update(self, node: int, lo: int, hi: int,
                     l: int, r: int, delta: int) -> None:
        """[l, r]의 모든 원소에 delta를 더한다."""
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

# === 시연 ===

if __name__ == "__main__":
    data = [1, 3, 5, 7, 9, 11]
    n = len(data)
    st = LazySegTree(data)

    print(f"Array: {data}")
    print(f"Sum [0,5]: {st.range_query(1, 0, n-1, 0, 5)}")
    print(f"Sum [1,3]: {st.range_query(1, 0, n-1, 1, 3)}")
    print()

    # 범위 갱신: 자리 [1, 4]에 10을 더한다
    print("Adding 10 to every element in [1, 4]...")
    st.range_update(1, 0, n - 1, 1, 4, 10)

    print(f"Sum [0,5]: {st.range_query(1, 0, n-1, 0, 5)}")
    print(f"Sum [1,3]: {st.range_query(1, 0, n-1, 1, 3)}")
    print(f"Sum [0,0]: {st.range_query(1, 0, n-1, 0, 0)}")
    print(f"Sum [5,5]: {st.range_query(1, 0, n-1, 5, 5)}")
    print()

    # 또 다른 범위 갱신
    print("Adding 5 to every element in [0, 2]...")
    st.range_update(1, 0, n - 1, 0, 2, 5)
    print(f"Sum [0,5]: {st.range_query(1, 0, n-1, 0, 5)}")
    print(f"Sum [0,2]: {st.range_query(1, 0, n-1, 0, 2)}")
```

**출력:**
```
Array: [1, 3, 5, 7, 9, 11]
Sum [0,5]: 36
Sum [1,3]: 15

Adding 10 to every element in [1, 4]...
Sum [0,5]: 76
Sum [1,3]: 45
Sum [0,0]: 1
Sum [5,5]: 11

Adding 5 to every element in [0, 2]...
Sum [0,5]: 91
Sum [0,2]: 39
```

---

## 7. 복잡도

| 연산 | 시간 | 공간 |
|-----------|------|-------|
| 세우기 | $O(n)$ | $O(n)$ |
| 범위 갱신 | $O(\log n)$ | $O(1)$ |
| 범위 질의 | $O(\log n)$ | $O(1)$ |

게으른 배열이 기억 사용을 두 배로(정수 $4n$개에서 $8n$개로) 늘리지만 점근 공간은 $O(n)$ 그대로이다.

---

## 8. 게으른 전파가 통하는 경우

게으른 전파는 갱신 연산이 **이어 붙일 수 있을** 때, 곧 한 노드의 미룬 갱신 여러 개를 꼬리표 하나로 합칠 수 있을 때 통한다. 덧셈은 $\delta_1$을 더한 뒤 $\delta_2$을 더하는 것이 $\delta_1 + \delta_2$을 더하는 것과 같으므로 이를 만족한다. 이 기법은 다음으로 넓어진다.

- **범위 덮어쓰기** (범위의 모든 원소를 어떤 값으로 둔다).
- **범위 곱하기** (모든 원소에 배수를 곱한다).
- **섞은 연산** (이를테면 곱한 뒤 더하기). 다만 서로 다른 두 연산을 이어 붙이려면 꼬리표의 순서를 조심해야 한다.

---

## 연습문제

**연습문제 1.**
게으른 전파의 짜임을 설명하고 질의와 갱신의 복잡도를 밝혀라.

??? success "연습문제 1 풀이"
    이 짜임은 위계적 분해를 이용해 일차보다 빠른 질의 시간을 이룬다. 대개 질의와 갱신이 모두 $O(\log n)$이고 세우는 데 $O(n)$이나 $O(n\log n)$이 든다.

---

**연습문제 2.**
입력 $[3, 1, 4, 1, 5, 9, 2, 6]$으로 게으른 전파를 세워라. 마지막 짜임을 보여라.

??? success "연습문제 2 풀이"
    세우기 알고리즘을 적용하며 중간 상태를 보여라. 트리 짜임이면 트리를 그려라. 배열에 바탕한 짜임이면 부모-자식 관계를 덧붙여 배열의 내용을 보여라.

---

**연습문제 3.**
질의 연산이 올바른 어떤 질의 범위에 대해서도 옳은 결과를 돌려줌을 증명하라.

??? success "연습문제 3 풀이"
    증명에는 대개 트리의 높이에 대한 귀납법을 쓴다. 노드마다 질의가 한 부분 트리 안에 온전히 들거나(재귀한다) 두 부분 트리에 걸친다(부분 결과를 모은다). 모으는 함수(합, 최솟값, 최댓값)가 결합적이므로 올바로 모아진다. $\square$

---

**연습문제 4.**
게으른 전파가 질의마다의 계산을 $O(n)$에서 $O(\log n)$으로 빠르게 하는 딥러닝 응용을 설명하라.

??? success "연습문제 4 풀이"
    응용으로는 누적 어텐션 가중치를 위한 접두사 합 질의, 길이가 제각각인 수열에서 효율적인 풀링을 위한 범위 질의, 검색 증강 생성을 위한 최근접 이웃 찾기, 3차원 딥러닝의 점 구름 처리를 위한 공간 색인이 있다.

## 정리하며

이 마당은 순진한 범위 갱신의 문제、게으른 생각、밀어 내리기 얼개、범위 갱신 알고리즘을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.
