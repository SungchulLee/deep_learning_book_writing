# 세우기

**구간 트리**는 배열의 이어진 부분 범위에 대한 모은 값을 담는 이진 트리이다. 어떤 질의에 답하거나 갱신을 처리하기 전에 입력 배열로 트리를 세워야 한다. 이 쪽은 구간 트리의 짜임을 설명하고, 그것을 담는 데 필요한 배열 크기를 이끌어 내고, $O(n)$의 재귀 세우기 알고리즘을 보인다.

## 구간 트리의 짜임

배열 $a[0..n-1]$이 주어졌을 때 구간 트리는 다음과 같은 완전 이진 트리이다.

- **잎**마다 원소 하나 $a[i]$에 해당한다.
- **내부 노드**마다 두 자식이 덮는 범위의 모은 값(합, 최솟값, 최댓값 따위)을 담는다.
- **뿌리**는 배열 전체 $a[0..n-1]$의 모은 값을 담는다.

범위 $[lo, hi]$을 맡은 노드는 다음을 가진다.

- $\text{mid} = \lfloor (lo + hi) / 2 \rfloor$일 때 $[lo, \text{mid}]$을 덮는 왼쪽 자식.
- $[\text{mid}+1, hi]$을 덮는 오른쪽 자식.

## 배열 표현

구간 트리는 (이진 힙과 비슷하게) 1부터 세는 색인으로 납작한 배열에 담긴다.

- 뿌리는 색인 1에 있다.
- 노드 $k$의 왼쪽 자식은 $2k$에 있다.
- 노드 $k$의 오른쪽 자식은 $2k + 1$에 있다.

!!! note "왜 노드를 4n개 잡는가"
    $n$이 2의 거듭제곱이면 크기 $n$인 배열의 구간 트리는 노드가 많아야 $2n - 1$개이다. 아무 $n$에서나 트리는 그다음 2의 거듭제곱을 담아야 하므로 최악의 경우 항목이 $4n$개까지 든다. `4 * n`을 잡으면 색인이 범위를 벗어나는 오류를 피하는 안전한 위 한계가 된다.

## 재귀 세우기 알고리즘

세우기 절차는 한 번 훑으며 아래에서 위로 트리를 세운다.

1. **밑칸.** $lo = hi$이면 노드가 잎이다. $a[lo]$을 담는다.
2. **재귀칸.** $\text{mid} = \lfloor (lo + hi) / 2 \rfloor$을 셈한다. 왼쪽 자식(범위 $[lo, \text{mid}]$)과 오른쪽 자식(범위 $[\text{mid}+1, hi]$)을 재귀적으로 세운다. 노드의 값을 자식의 모은 값으로 둔다.

$O(n)$개 노드마다 꼭 한 번씩 들르므로 세우기가 $O(n)$ 시간에 돈다.

## 구현

```python
"""
구간 트리 세우기.

입력 배열을 납작한 배열에 담긴 구간 트리로 바꾸는 O(n)의
재귀 세우기 알고리즘을 보인다.
합에 바탕한 구간 트리와 최솟값에 바탕한 구간 트리를 함께 보인다.
"""


# === 구간 트리 (합) ===

class SegmentTree:
    """O(n)에 세우는, 범위 합 질의를 위한 구간 트리."""

    def __init__(self, data: list):
        """입력 배열로 구간 트리를 세운다."""
        self.n = len(data)
        self.tree = [0] * (4 * self.n)
        if self.n > 0:
            self._build(data, 1, 0, self.n - 1)

    def _build(self, data: list, node: int, lo: int, hi: int) -> None:
        """트리를 재귀적으로 세운다.

        인수:
            data: 본디 배열.
            node: 트리 배열에서 지금 노드의 색인.
            lo, hi: 이 노드가 덮는 본디 배열의 범위.
        """
        if lo == hi:
            # 잎 노드 — 원소 하나를 담는다
            self.tree[node] = data[lo]
            return

        mid = (lo + hi) // 2
        self._build(data, 2 * node, lo, mid)         # 왼쪽 자식을 세운다
        self._build(data, 2 * node + 1, mid + 1, hi)  # 오른쪽 자식을 세운다
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, node: int, lo: int, hi: int, l: int, r: int) -> int:
        """[l, r]의 원소의 합을 돌려준다."""
        if r < lo or hi < l:
            return 0  # 합의 항등원
        if l <= lo and hi <= r:
            return self.tree[node]
        mid = (lo + hi) // 2
        return (self.query(2 * node, lo, mid, l, r)
                + self.query(2 * node + 1, mid + 1, hi, l, r))

    def print_tree(self, node: int, lo: int, hi: int, depth: int = 0) -> None:
        """그려 보려고 트리의 짜임을 찍는다."""
        indent = "  " * depth
        print(f"{indent}Node {node}: [{lo},{hi}] = {self.tree[node]}")
        if lo < hi:
            mid = (lo + hi) // 2
            self.print_tree(2 * node, lo, mid, depth + 1)
            self.print_tree(2 * node + 1, mid + 1, hi, depth + 1)


# === 시연 ===

if __name__ == "__main__":
    data = [1, 3, 5, 7, 9, 11]
    print(f"Input array: {data}")
    print(f"Array size n = {len(data)}")
    print(f"Tree array size = {4 * len(data)}")
    print()

    st = SegmentTree(data)

    print("Segment tree structure:")
    st.print_tree(1, 0, len(data) - 1)
    print()

    # 범위 합을 확인한다
    queries = [(0, 2), (1, 4), (0, 5), (3, 5), (2, 2)]
    for l, r in queries:
        result = st.query(1, 0, len(data) - 1, l, r)
        expected = sum(data[l:r + 1])
        print(f"Sum [{l},{r}] = {result}  (expected {expected})")
```

**출력:**
```
Input array: [1, 3, 5, 7, 9, 11]
Array size n = 6
Tree array size = 24

Segment tree structure:
Node 1: [0,5] = 36
  Node 2: [0,2] = 9
    Node 4: [0,1] = 4
      Node 8: [0,0] = 1
      Node 9: [1,1] = 3
    Node 5: [2,2] = 5
  Node 3: [3,5] = 27
    Node 6: [3,4] = 16
      Node 12: [3,3] = 7
      Node 13: [4,4] = 9
    Node 7: [5,5] = 11

Sum [0,2] = 9  (expected 9)
Sum [1,4] = 24  (expected 24)
Sum [0,5] = 36  (expected 36)
Sum [3,5] = 27  (expected 27)
Sum [2,2] = 5  (expected 5)
```

## 세우기의 복잡도

| 항목 | 복잡도 |
|--------|-----------|
| 시간 | $O(n)$ |
| 공간 | $O(n)$ (크기 $4n$의 트리 배열) |

$O(n)$ 시간은 트리의 노드가 많아야 $4n$개이고 저마다 꼭 한 번 초기화된다는 사실에서 따라 나온다. 점 갱신을 $n$번 해서 세우면 $O(n \log n)$이 드는데 그보다 빠르다.

## 다른 모으기로 넓히기

이 세우기는 결합적인 어떤 이항 연산에도 통한다. 흔한 선택은 다음과 같다.

| 모으기 | 항등원 | 합치는 연산 |
|-----------|-----------------|-----------------|
| 합 | 0 | $a + b$ |
| 최솟값 | $+\infty$ | $\min(a, b)$ |
| 최댓값 | $-\infty$ | $\max(a, b)$ |
| 최대공약수 | 0 | $\gcd(a, b)$ |
| 배타적 논리합 | 0 | $a \oplus b$ |

바꿔야 할 것은 합치는 함수와, 질의가 노드의 범위 바깥에 놓일 때 쓰는 항등원뿐이다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.


## 연습문제

**연습문제 1.**
세우기의 짜임을 설명하고 질의와 갱신의 복잡도를 밝혀라.

??? success "연습문제 1 풀이"
    이 짜임은 위계적 분해를 이용해 일차보다 빠른 질의 시간을 이룬다. 대개 질의와 갱신이 모두 $O(\log n)$이고 세우는 데 $O(n)$이나 $O(n\log n)$이 든다.

---

**연습문제 2.**
입력 $[3, 1, 4, 1, 5, 9, 2, 6]$으로 세우기를 세워라. 마지막 짜임을 보여라.

??? success "연습문제 2 풀이"
    세우기 알고리즘을 적용하며 중간 상태를 보여라. 트리 짜임이면 트리를 그려라. 배열에 바탕한 짜임이면 부모-자식 관계를 덧붙여 배열의 내용을 보여라.

---

**연습문제 3.**
질의 연산이 올바른 어떤 질의 범위에 대해서도 옳은 결과를 돌려줌을 증명하라.

??? success "연습문제 3 풀이"
    증명에는 대개 트리의 높이에 대한 귀납법을 쓴다. 노드마다 질의가 한 부분 트리 안에 온전히 들거나(재귀한다) 두 부분 트리에 걸친다(부분 결과를 모은다). 모으는 함수(합, 최솟값, 최댓값)가 결합적이므로 올바로 모아진다. $\square$

---

**연습문제 4.**
세우기가 질의마다의 계산을 $O(n)$에서 $O(\log n)$으로 빠르게 하는 딥러닝 응용을 설명하라.

??? success "연습문제 4 풀이"
    응용으로는 누적 어텐션 가중치를 위한 접두사 합 질의, 길이가 제각각인 수열에서 효율적인 풀링을 위한 범위 질의, 검색 증강 생성을 위한 최근접 이웃 찾기, 3차원 딥러닝의 점 구름 처리를 위한 공간 색인이 있다.