# 점 갱신

구간 트리를 세운 뒤에도 바탕 배열이 바뀔 수 있다. **점 갱신**은 원소 하나 $a[\text{idx}]$을 고치고 고친 잎에서 뿌리까지의 경로 위 모든 조상 노드를 다시 셈한다. 트리의 높이가 $O(\log n)$이므로 이 경로에는 노드가 많아야 $\lfloor \log_2 n \rfloor + 1$개 있어 갱신이 $O(\log n)$이다.

## 알고리즘

$a[\text{idx}]$을 새 값 $v$으로 두려면 다음과 같이 한다.

1. **잎까지 내려간다.** 뿌리(범위 $[0, n-1]$)에서 시작해 $\text{mid} = \lfloor (lo + hi) / 2 \rfloor$을 셈한다. $\text{idx} \leq \text{mid}$이면 왼쪽 자식으로, 아니면 오른쪽 자식으로 재귀한다.
2. **잎을 고친다.** $lo = hi = \text{idx}$일 때 `tree[node] = v`으로 둔다.
3. **조상을 다시 셈한다.** 재귀를 거슬러 올라오며 내부 노드마다 두 자식의 모은 값으로 다시 셈한다: `tree[node] = tree[2*node] + tree[2*node+1]`.

재귀가 층마다 꼭 자식 하나를 들르므로 들르는 노드는 트리의 높이를 $h$이라 할 때 모두 $h + 1$개이다.

## 한 걸음씩 따라가기

구간 트리에 담긴 배열 $a = [1, 3, 5, 7, 9]$을 생각하자. 자리 2를 5에서 50으로 고친다.

**갱신 전:**

| 노드 | 범위 | 값 |
|:----:|:-----:|:-----:|
| 1 | $[0,4]$ | 25 |
| 2 | $[0,2]$ | 9 |
| 3 | $[3,4]$ | 16 |
| 4 | $[0,1]$ | 4 |
| 5 | $[2,2]$ | 5 |

**Update(idx=2, val=50):**

1. 노드 1 ($[0,4]$): $\text{mid}=2$, $\text{idx}=2 \leq 2$ → 왼쪽 노드 2로 간다.
2. 노드 2 ($[0,2]$): $\text{mid}=1$, $\text{idx}=2 > 1$ → 오른쪽 노드 5로 간다.
3. 노드 5 ($[2,2]$): 잎이므로 값을 50으로 둔다.
4. 노드 2로 돌아와 $4 + 50 = 54$으로 다시 셈한다.
5. 노드 1로 돌아와 $54 + 16 = 70$으로 다시 셈한다.

**갱신 뒤:**

| 노드 | 범위 | 값 |
|:----:|:-----:|:-----:|
| 1 | $[0,4]$ | 70 |
| 2 | $[0,2]$ | 54 |
| 5 | $[2,2]$ | 50 |

노드 3개만 바뀌었고 모두 자리 2의 뿌리-잎 경로 위에 있다.

## 구현

```python
"""
구간 트리의 점 갱신.

잎을 고치고 뿌리-잎 경로의 모든 조상을 다시 셈하는
O(log n)의 점 갱신 연산을 보인다.
"""


# === 점 갱신을 하는 구간 트리 ===

class SegmentTree:
    """점 갱신을 하는, 합 질의를 위한 구간 트리."""

    def __init__(self, data: list):
        """입력 데이터로 구간 트리를 세운다."""
        self.n = len(data)
        self.tree = [0] * (4 * self.n)
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

    def update(self, node: int, lo: int, hi: int,
               idx: int, val: int) -> None:
        """a[idx] = val으로 두고 모든 조상을 다시 셈한다.

        인수:
            node: 트리 배열에서 지금 노드의 색인.
            lo, hi: 이 노드가 덮는 범위.
            idx: 고칠 본디 배열의 자리.
            val: a[idx]의 새 값.
        """
        if lo == hi:
            self.tree[node] = val
            return
        mid = (lo + hi) // 2
        if idx <= mid:
            self.update(2 * node, lo, mid, idx, val)
        else:
            self.update(2 * node + 1, mid + 1, hi, idx, val)
        # 자식에서 이 노드를 다시 셈한다
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, node: int, lo: int, hi: int,
              l: int, r: int) -> int:
        """[l, r]에 대한 범위 합 질의."""
        if r < lo or hi < l:
            return 0
        if l <= lo and hi <= r:
            return self.tree[node]
        mid = (lo + hi) // 2
        return (self.query(2 * node, lo, mid, l, r)
                + self.query(2 * node + 1, mid + 1, hi, l, r))


# === 시연 ===

if __name__ == "__main__":
    data = [1, 3, 5, 7, 9]
    n = len(data)
    st = SegmentTree(data)

    print(f"Original array: {data}")
    print(f"Sum [0,4] = {st.query(1, 0, n-1, 0, 4)}")
    print(f"Sum [1,3] = {st.query(1, 0, n-1, 1, 3)}")
    print()

    # 점 갱신: a[2] = 50으로 둔다
    print("Update: a[2] = 50")
    st.update(1, 0, n - 1, 2, 50)
    print(f"Sum [0,4] = {st.query(1, 0, n-1, 0, 4)}")
    print(f"Sum [1,3] = {st.query(1, 0, n-1, 1, 3)}")
    print(f"Sum [2,2] = {st.query(1, 0, n-1, 2, 2)}")
    print()

    # 또 다른 갱신: a[0] = 100으로 둔다
    print("Update: a[0] = 100")
    st.update(1, 0, n - 1, 0, 100)
    print(f"Sum [0,4] = {st.query(1, 0, n-1, 0, 4)}")
    print(f"Sum [0,0] = {st.query(1, 0, n-1, 0, 0)}")
```

**출력:**
```
Original array: [1, 3, 5, 7, 9]
Sum [0,4] = 25
Sum [1,3] = 15

Update: a[2] = 50
Sum [0,4] = 70
Sum [1,3] = 60
Sum [2,2] = 50

Update: a[0] = 100
Sum [0,4] = 169
Sum [0,0] = 100
```

## 복잡도

| 연산 | 시간 | 공간 |
|-----------|------|-------|
| 점 갱신 | $O(\log n)$ | 보조 공간 $O(1)$ |

갱신은 트리의 높이를 $h = \lfloor \log_2 n \rfloor$이라 할 때 꼭 $h + 1$개의 노드를 건드린다. 노드마다 (비교 한 번과 덧셈 한 번인) $O(1)$의 일이 든다.

!!! tip "더하기와 바꾸기"
    위 구현은 자리의 값을 **덮어쓴다**. 대신 차이값을 **더하려면** 잎 갱신을 `tree[node] = val`에서 `tree[node] += val`으로 바꾼다. 조상을 다시 셈하는 것은 그대로이다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.


## 연습문제

**연습문제 1.**
점 갱신의 짜임을 설명하고 질의와 갱신의 복잡도를 밝혀라.

??? success "연습문제 1 풀이"
    이 짜임은 위계적 분해를 이용해 일차보다 빠른 질의 시간을 이룬다. 대개 질의와 갱신이 모두 $O(\log n)$이고 세우는 데 $O(n)$이나 $O(n\log n)$이 든다.

---

**연습문제 2.**
입력 $[3, 1, 4, 1, 5, 9, 2, 6]$으로 점 갱신을 세워라. 마지막 짜임을 보여라.

??? success "연습문제 2 풀이"
    세우기 알고리즘을 적용하며 중간 상태를 보여라. 트리 짜임이면 트리를 그려라. 배열에 바탕한 짜임이면 부모-자식 관계를 덧붙여 배열의 내용을 보여라.

---

**연습문제 3.**
질의 연산이 올바른 어떤 질의 범위에 대해서도 옳은 결과를 돌려줌을 증명하라.

??? success "연습문제 3 풀이"
    증명에는 대개 트리의 높이에 대한 귀납법을 쓴다. 노드마다 질의가 한 부분 트리 안에 온전히 들거나(재귀한다) 두 부분 트리에 걸친다(부분 결과를 모은다). 모으는 함수(합, 최솟값, 최댓값)가 결합적이므로 올바로 모아진다. $\square$

---

**연습문제 4.**
점 갱신가 질의마다의 계산을 $O(n)$에서 $O(\log n)$으로 빠르게 하는 딥러닝 응용을 설명하라.

??? success "연습문제 4 풀이"
    응용으로는 누적 어텐션 가중치를 위한 접두사 합 질의, 길이가 제각각인 수열에서 효율적인 풀링을 위한 범위 질의, 검색 증강 생성을 위한 최근접 이웃 찾기, 3차원 딥러닝의 점 구름 처리를 위한 공간 색인이 있다.