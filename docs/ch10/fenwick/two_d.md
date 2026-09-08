# 2차원 펜윅 트리

1차원 BIT는 한 줄짜리 배열의 접두사 합 질의에 답한다. 영상 처리, 2차원 빈도 세기, 격자 기반 게임 같은 여러 응용은 2차원 행렬의 접두사 합이 필요하다. **2차원 펜윅 트리**는 1차원 짜임을 넓혀 $n \times m$ 격자에서 점 갱신과 직사각 접두사 합 질의를 모두 $O(\log n \cdot \log m)$ 시간에 다룬다.

---

## 1. 1차원에서 2차원으로

1차원 BIT에서 색인 $i$마다 잇따른 $\text{lowbit}(i)$개 원소의 범위를 맡는다. 2차원으로 넓힐 때는 이 생각을 행 차원과 열 차원에 따로 적용한다. 2차원 BIT는 2차원 배열 `tree[1..n][1..m]`에 담기며 다음과 같다.

$$
\texttt{tree}[i][j] = \sum_{\substack{i - \text{lowbit}(i) < r \leq i \\ j - \text{lowbit}(j) < c \leq j}} a[r][c]
$$

항목 `tree[i][j]`마다 행 범위는 $\text{lowbit}(i)$이, 열 범위는 $\text{lowbit}(j)$이 정하는 직사각 부분 영역의 합을 쌓는다.

---

## 2. 2차원 점 갱신

행렬의 자리 $(x, y)$에 $\delta$을 더하려면 $(x, y)$을 범위에 담는 모든 BIT 항목을 고친다. 겹친 고리가 필요하다. 바깥 고리는 $\text{lowbit}(i)$을 더해 행 색인 $i$을 나아가게 하고, 안쪽 고리는 $\text{lowbit}(j)$을 더해 열 색인 $j$을 나아가게 한다.

---

## 3. 2차원 접두사 질의

2차원 접두사 합 $\text{prefix}(x, y) = \sum_{r=1}^{x} \sum_{c=1}^{y} a[r][c]$은 차원마다 가장 낮은 켜진 비트를 떼는 겹친 고리로 셈한다.

$$
\text{prefix}(x, y) = \sum \texttt{tree}[i][j]
$$

여기서 $i$은 $\text{lowbit}(i)$을 없애며 $x$에서 내려가고, 그런 $i$마다 $j$은 $\text{lowbit}(j)$을 없애며 $y$에서 내려간다.

---

## 4. 포함-배제로 하는 2차원 범위 합

모서리가 $(r_1, c_1)$과 $(r_2, c_2)$인 직사각형의 합을 셈하려면 **포함-배제** 원리를 쓴다.

$$
\text{rangeSum}(r_1, c_1, r_2, c_2) = \text{prefix}(r_2, c_2) - \text{prefix}(r_1 - 1, c_2) - \text{prefix}(r_2, c_1 - 1) + \text{prefix}(r_1 - 1, c_1 - 1)
$$

1차원 공식 $\text{rangeSum}(l, r) = \text{prefix}(r) - \text{prefix}(l-1)$의 2차원 판이다.

!!! note "포함-배제 그려 보기"
    직사각형 안의 합을 찾으려면 오른쪽 아래 모서리까지의 접두사 합에서 시작해, 너무 멀리 뻗은 두 영역(왼쪽과 위쪽)을 빼고, 두 번 빼진 겹친 영역을 도로 더한다.

---

## 5. 구현

```python
"""
2차원 이진 색인 트리 (펜윅 트리).

겹친 가장 낮은 켜진 비트 순회와 포함-배제 원리로
2차원 격자에서 점 갱신과 직사각 범위 합 질의를
받쳐 준다.
"""

# === 2차원 펜윅 트리 ===

class FenwickTree2D:
    """점 갱신과 직사각 합 질의를 위한 2차원 BIT."""

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.tree = [[0] * (cols + 1) for _ in range(rows + 1)]

    def update(self, x: int, y: int, delta: int) -> None:
        """자리 (x, y)에 delta를 더한다. 둘 다 1부터 센다."""
        i = x
        while i <= self.rows:
            j = y
            while j <= self.cols:
                self.tree[i][j] += delta
                j += j & (-j)
            i += i & (-i)

    def prefix(self, x: int, y: int) -> int:
        """직사각형 [1..x, 1..y]의 모든 원소의 합을 돌려준다."""
        s = 0
        i = x
        while i > 0:
            j = y
            while j > 0:
                s += self.tree[i][j]
                j -= j & (-j)
            i -= i & (-i)
        return s

    def range_sum(self, r1: int, c1: int, r2: int, c2: int) -> int:
        """직사각형 [r1..r2, c1..c2]의 원소의 합을 돌려준다.

        포함-배제 공식을 쓴다.
          prefix(r2,c2) - prefix(r1-1,c2) - prefix(r2,c1-1) + prefix(r1-1,c1-1)
        """
        return (self.prefix(r2, c2)
                - self.prefix(r1 - 1, c2)
                - self.prefix(r2, c1 - 1)
                + self.prefix(r1 - 1, c1 - 1))

# === 시연 ===

if __name__ == "__main__":
    # 4×4 행렬
    matrix = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]

    rows, cols = len(matrix), len(matrix[0])
    ft = FenwickTree2D(rows, cols)

    # 2차원 BIT를 세운다
    for i in range(rows):
        for j in range(cols):
            ft.update(i + 1, j + 1, matrix[i][j])

    print("Matrix:")
    for row in matrix:
        print(f"  {row}")
    print()

    # 접두사 합 질의
    print(f"prefix(2, 3) = {ft.prefix(2, 3)}")  # [1..2, 1..3]의 합
    expected = sum(matrix[r][c] for r in range(2) for c in range(3))
    print(f"  Expected: {expected}")
    print()

    # 범위 합 질의
    print(f"rangeSum(2, 2, 3, 4) = {ft.range_sum(2, 2, 3, 4)}")
    expected = sum(matrix[r][c] for r in range(1, 3) for c in range(1, 4))
    print(f"  Expected: {expected}")
    print()

    print(f"rangeSum(1, 1, 4, 4) = {ft.range_sum(1, 1, 4, 4)}")
    expected = sum(matrix[r][c] for r in range(4) for c in range(4))
    print(f"  Expected: {expected}")
```

**출력:**
```
Matrix:
  [1, 2, 3, 4]
  [5, 6, 7, 8]
  [9, 10, 11, 12]
  [13, 14, 15, 16]

prefix(2, 3) = 24
  Expected: 24

rangeSum(2, 2, 3, 4) = 54
  Expected: 54

rangeSum(1, 1, 4, 4) = 136
  Expected: 136
```

---

## 6. 복잡도 분석

| 연산 | 시간 | 공간 |
|-----------|------|-------|
| 세우기 | $O(nm \log n \log m)$ | $O(nm)$ |
| 점 갱신 | $O(\log n \cdot \log m)$ | $O(1)$ |
| 접두사 질의 | $O(\log n \cdot \log m)$ | $O(1)$ |
| 범위 합 질의 | $O(\log n \cdot \log m)$ | $O(1)$ |

BIT 배열의 크기가 행렬과 같으므로 공간의 짐은 행렬 자체를 담는 것과 같다.

---

## 7. 더 높은 차원

같은 생각이 $d$차원으로 일반화된다. $d$차원 BIT는 $O(n^d)$의 공간으로 점 갱신과 접두사 질의를 $O(\log^d n)$ 시간에 받쳐 준다. 다만 상수 배가 빠르게 커져 실제로는 2차원과 이따금 3차원이 가장 흔하다.

---

## 연습문제

**연습문제 1.**
2차원 펜윅 트리의 짜임을 설명하고 질의와 갱신의 복잡도를 밝혀라.

??? success "연습문제 1 풀이"
    이 짜임은 위계적 분해를 이용해 일차보다 빠른 질의 시간을 이룬다. 대개 질의와 갱신이 모두 $O(\log n)$이고 세우는 데 $O(n)$이나 $O(n\log n)$이 든다.

---

**연습문제 2.**
입력 $[3, 1, 4, 1, 5, 9, 2, 6]$으로 2차원 펜윅 트리를 세워라. 마지막 짜임을 보여라.

??? success "연습문제 2 풀이"
    세우기 알고리즘을 적용하며 중간 상태를 보여라. 트리 짜임이면 트리를 그려라. 배열에 바탕한 짜임이면 부모-자식 관계를 덧붙여 배열의 내용을 보여라.

---

**연습문제 3.**
질의 연산이 올바른 어떤 질의 범위에 대해서도 옳은 결과를 돌려줌을 증명하라.

??? success "연습문제 3 풀이"
    증명에는 대개 트리의 높이에 대한 귀납법을 쓴다. 노드마다 질의가 한 부분 트리 안에 온전히 들거나(재귀한다) 두 부분 트리에 걸친다(부분 결과를 모은다). 모으는 함수(합, 최솟값, 최댓값)가 결합적이므로 올바로 모아진다. $\square$

---

**연습문제 4.**
2차원 펜윅 트리가 질의마다의 계산을 $O(n)$에서 $O(\log n)$으로 빠르게 하는 딥러닝 응용을 설명하라.

??? success "연습문제 4 풀이"
    응용으로는 누적 어텐션 가중치를 위한 접두사 합 질의, 길이가 제각각인 수열에서 효율적인 풀링을 위한 범위 질의, 검색 증강 생성을 위한 최근접 이웃 찾기, 3차원 딥러닝의 점 구름 처리를 위한 공간 색인이 있다.

## 정리하며

이 마당은 1차원에서 2차원으로、2차원 점 갱신、2차원 접두사 질의、포함-배제로 하는 2차원 범위 합을 차례로 짚었다.

**참고 문헌**

- Fenwick, P. M. (1994). A New Data Structure for Cumulative Frequency Tables. *Software: Practice and Experience*, 24(3), 327-336.
- Mishra, S. (2013). 2D Binary Indexed Trees. *TopCoder Tutorials*.
