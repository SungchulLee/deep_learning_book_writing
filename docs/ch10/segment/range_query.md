# 범위 질의

구간 트리의 주된 쓰임은 **범위 질의**에 답하는 것, 곧 이어진 어떤 부분 배열 $a[l..r]$의 모은 값(합, 최솟값, 최댓값, 최대공약수 따위)을 $O(\log n)$ 시간에 셈하는 것이다. 이 쪽은 재귀 질의 알고리즘이 $[l, r]$을 미리 셈해 둔 노드 범위의 가장 작은 묶음으로 나누어 어떻게 도는지 설명하고 $O(\log n)$ 한계를 증명한다.

---

## 1. 세 가지 경우

질의 범위 $[l, r]$에 대해 범위 $[lo, hi]$을 덮는 노드에 물을 때 세 상황 가운데 꼭 하나가 일어난다.

1. **겹치지 않음** ($r < lo$이거나 $hi < l$). 질의 범위가 이 노드의 범위와 만나지 않는다. 항등원(합이면 0, 최솟값이면 $+\infty$)을 돌려준다.

2. **온전히 담김** ($l \leq lo$이고 $hi \leq r$). 노드의 범위가 질의 범위 안에 온전히 든다. 미리 셈해 둔 노드의 값을 곧바로 돌려준다. 재귀가 필요 없다.

3. **일부만 겹침.** 질의 범위가 노드의 범위와 일부만 겹친다. 두 자식으로 재귀한 뒤 그 결과를 모은다.

!!! note "왜 세 경우로 넉넉한가"
    트리의 모든 노드가 이 세 갈래 가운데 꼭 하나에 든다. 경우 1과 2는 곧바로 끝나고 경우 3은 문제를 잎에 더 가까운 부분 문제 둘로 나눈다. 이것이 끝남과 옳음을 보장한다.

---

## 2. 한 걸음씩 따라가기

구간 트리에 담긴 배열 $a = [1, 3, 5, 7, 9]$을 생각하자.

**질의: [1, 3]의 합**

뿌리(노드 1, 범위 $[0, 4]$)에서 시작한다.

| 노드 | 범위 | 경우 | 하는 일 |
|:----:|:-----:|:----:|--------|
| 1 | $[0,4]$ | 일부 | 나눔: 자식에 묻는다 |
| 2 | $[0,2]$ | 일부 | 나눔: 자식에 묻는다 |
| 3 | $[3,4]$ | 일부 | 나눔: 자식에 묻는다 |
| 4 | $[0,1]$ | 일부 | 나눔: 자식에 묻는다 |
| 5 | $[2,2]$ | 온전 | 5를 돌려준다 |
| 6 | $[3,3]$ | 온전 | 7을 돌려준다 |
| 7 | $[4,4]$ | 겹치지 않음 | 0을 돌려준다 |
| 8 | $[0,0]$ | 겹치지 않음 | 0을 돌려준다 |
| 9 | $[1,1]$ | 온전 | 3을 돌려준다 |

결과: $3 + 5 + 7 = 15$.

질의가 노드 9개를 들르지만 그 가운데 5개에서는 (경우 1과 2로) 곧바로 돌아온다. 뜻있게 이바지하는 노드는 많아야 $O(\log n)$개이다.

---

## 3. 질의가 O(log n)인 까닭

트리의 층마다 질의가 노드를 많아야 4개 들르지만 그 가운데 많아야 2개만 더 재귀한다(경우 3). 논증은 다음과 같다.

- 맨 위 층에는 살아 있는 노드가 1개이다.
- 층마다 일부만 겹치는 노드가 이어 갈 자식을 많아야 2개 내고, 온전히 담기거나 겹치지 않는 자식은 멈춘다.
- 트리의 층이 $O(\log n)$개이므로 전체 일은 $O(\log n)$이다.

더 정확히, 어느 층에서도 $[l, r]$과 일부만 겹치는 노드는 많아야 2개이다. 왼쪽 경계에 하나, 오른쪽 경계에 하나이다. 그 사이의 노드는 모두 온전히 담긴다.

---

## 4. 구현

```python
"""
구간 트리의 범위 질의.

겹치지 않음, 온전히 담김, 일부만 겹침의 세 경우로 나누는
O(log n)의 범위 질의 알고리즘을
보인다.
"""

# === 범위 질의를 하는 구간 트리 ===

class SegmentTree:
    """범위 합 질의를 위한 구간 트리."""

    def __init__(self, data: list):
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

    def query(self, node: int, lo: int, hi: int,
              l: int, r: int) -> int:
        """세 경우로 나누어 a[l..r]의 합을 돌려준다.

        인수:
            node: 트리 배열에서 지금 자리.
            lo, hi: 이 노드가 덮는 범위.
            l, r: 질의 범위.
        """
        # 경우 1: 겹치지 않음
        if r < lo or hi < l:
            return 0

        # 경우 2: 온전히 담김
        if l <= lo and hi <= r:
            return self.tree[node]

        # 경우 3: 일부만 겹침 — 두 자식으로 재귀한다
        mid = (lo + hi) // 2
        left_sum = self.query(2 * node, lo, mid, l, r)
        right_sum = self.query(2 * node + 1, mid + 1, hi, l, r)
        return left_sum + right_sum

    def update(self, node: int, lo: int, hi: int,
               idx: int, val: int) -> None:
        """점 갱신: a[idx] = val으로 둔다."""
        if lo == hi:
            self.tree[node] = val
            return
        mid = (lo + hi) // 2
        if idx <= mid:
            self.update(2 * node, lo, mid, idx, val)
        else:
            self.update(2 * node + 1, mid + 1, hi, idx, val)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

# === 시연 ===

if __name__ == "__main__":
    data = [1, 3, 5, 7, 9]
    n = len(data)
    st = SegmentTree(data)

    print(f"Array: {data}")
    print()

    # 여러 가지 범위 질의
    queries = [(0, 4), (1, 3), (0, 0), (2, 4), (3, 3)]
    for l, r in queries:
        result = st.query(1, 0, n - 1, l, r)
        expected = sum(data[l:r + 1])
        print(f"  Sum [{l},{r}] = {result}  (expected {expected})")

    # 점 갱신 뒤
    print()
    st.update(1, 0, n - 1, 2, 50)
    data[2] = 50
    print("After setting a[2] = 50:")
    for l, r in queries:
        result = st.query(1, 0, n - 1, l, r)
        expected = sum(data[l:r + 1])
        print(f"  Sum [{l},{r}] = {result}  (expected {expected})")
```

**출력:**
```
Array: [1, 3, 5, 7, 9]

  Sum [0,4] = 25  (expected 25)
  Sum [1,3] = 15  (expected 15)
  Sum [0,0] = 1  (expected 1)
  Sum [2,4] = 21  (expected 21)
  Sum [3,3] = 7  (expected 7)

After setting a[2] = 50:
  Sum [0,4] = 70  (expected 70)
  Sum [1,3] = 60  (expected 60)
  Sum [0,0] = 1  (expected 1)
  Sum [2,4] = 66  (expected 66)
  Sum [3,3] = 7  (expected 7)
```

---

## 5. 복잡도

| 측면 | 한계 |
|--------|-------|
| 질의당 시간 | $O(\log n)$ |
| 질의당 공간 | 스택 틀 $O(\log n)$개 (재귀 깊이) |

$O(\log n)$개 층마다 일부만 겹치는 노드가 많아야 2개이므로 $O(\log n)$ 한계가 따라 나온다. 다른 노드는 모두 $O(1)$에 끝난다.

---

## 연습문제

**연습문제 1.**
범위 질의의 짜임을 설명하고 질의와 갱신의 복잡도를 밝혀라.

??? success "연습문제 1 풀이"
    이 짜임은 위계적 분해를 이용해 일차보다 빠른 질의 시간을 이룬다. 대개 질의와 갱신이 모두 $O(\log n)$이고 세우는 데 $O(n)$이나 $O(n\log n)$이 든다.

---

**연습문제 2.**
입력 $[3, 1, 4, 1, 5, 9, 2, 6]$으로 범위 질의를 세워라. 마지막 짜임을 보여라.

??? success "연습문제 2 풀이"
    세우기 알고리즘을 적용하며 중간 상태를 보여라. 트리 짜임이면 트리를 그려라. 배열에 바탕한 짜임이면 부모-자식 관계를 덧붙여 배열의 내용을 보여라.

---

**연습문제 3.**
질의 연산이 올바른 어떤 질의 범위에 대해서도 옳은 결과를 돌려줌을 증명하라.

??? success "연습문제 3 풀이"
    증명에는 대개 트리의 높이에 대한 귀납법을 쓴다. 노드마다 질의가 한 부분 트리 안에 온전히 들거나(재귀한다) 두 부분 트리에 걸친다(부분 결과를 모은다). 모으는 함수(합, 최솟값, 최댓값)가 결합적이므로 올바로 모아진다. $\square$

---

**연습문제 4.**
범위 질의가 질의마다의 계산을 $O(n)$에서 $O(\log n)$으로 빠르게 하는 딥러닝 응용을 설명하라.

??? success "연습문제 4 풀이"
    응용으로는 누적 어텐션 가중치를 위한 접두사 합 질의, 길이가 제각각인 수열에서 효율적인 풀링을 위한 범위 질의, 검색 증강 생성을 위한 최근접 이웃 찾기, 3차원 딥러닝의 점 구름 처리를 위한 공간 색인이 있다.

## 정리하며

이 마당은 세 가지 경우、한 걸음씩 따라가기、질의가 O(log n)인 까닭、구현을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.
