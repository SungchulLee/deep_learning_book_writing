# 나란한 행렬 곱하기

행렬 곱하기는 과학 셈, 깊은 배움, 그래프 알고리즘에서 가장 중요한 연산 가운데 하나이다. $n \times n$ 행렬 둘을 곱하는 여느 알고리즘은 셈 연산을 $O(n^3)$번 하지만 이 연산들은 크게 나란하다. 내놓기 칸이 서로 독립임을 살리면 나란한 행렬 곱하기는 뻗음 $O(\log n)$을 이루어 가장 잘 나란해지는 근본 알고리즘 가운데 하나가 된다.

## 문제 서술

$n \times n$ 행렬 $A$과 $B$이 주어질 때 다음과 같은 $C = A \times B$을 셈한다:

$$
C[i][j] = \sum_{k=0}^{n-1} A[i][k] \cdot B[k][j]
$$

칸 $C[i][j]$마다 $A$의 $i$번째 줄과 $B$의 $j$번째 열의 안쪽 곱이다. 칸 $n^2$개가 모두 독립이므로 나란히 셈할 수 있다.

## 나란히 하는 길

### 되돌이 나란히 하기

가장 단순한 셈속은 $C$의 칸이 서로 독립임을 살린다. 바깥 되돌이 둘($i$과 $j$)을 나란히 돌리고 안쪽 되돌이($k$)가 안쪽 곱을 셈한다.

- **일**: $T_1 = O(n^3)$으로 차례 알고리즘과 같다.
- **뻗음**: 안쪽 곱 합마다 나란한 줄임을 쓰면 $T_\infty = O(\log n)$.
- **나란함**: $P = O(n^3 / \log n)$.

### 되돌이(나누어 정복하기)

행렬마다 $n/2 \times n/2$ 덩이 넷으로 가른다:

$$
\begin{bmatrix} C_{11} & C_{12} \\ C_{21} & C_{22} \end{bmatrix}
= \begin{bmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{bmatrix}
\cdot \begin{bmatrix} B_{11} & B_{12} \\ B_{21} & B_{22} \end{bmatrix}
$$

이는 되돌이 곱하기 8번과 행렬 더하기 4번을 낳는다:

$$
C_{ij} = A_{i1} \cdot B_{1j} + A_{i2} \cdot B_{2j}
$$

되돌이 곱하기 8번을 나란히 돌리고(갈라짐) 짝 더하기 4번이 합침을 이룬다.

**일**: $T_1(n) = 8 \cdot T_1(n/2) + O(n^2) = O(n^3)$.

**뻗음**: 곱하기 8번이 나란히 도므로 가지 하나만 뻗음에 이바지한다. 더하기는 원소마다 나란히 하면 뻗음이 $O(\log n)$이다:

$$
T_\infty(n) = T_\infty(n/2) + O(\log n) = O(\log^2 n)
$$

**나란함**: $P = O(n^3 / \log^2 n)$.

## 구현

```python
"""
나란한 행렬 곱하기 흉내내기.

어수룩한 세 겹 되돌이와 되돌이 나누어 정복하기를 견주며
저마다의 일과 뻗음을 좇는다.
"""

# ===================================================================
# 어수룩한 나란한 행렬 곱하기
# ===================================================================

def matmul_naive(A, B):
    """여느 알고리즘으로 행렬 A과 B을 곱한다.

    인수:
        A: n x n matrix (list of lists)
        B: n x n matrix (list of lists)

    반환값:
        C: n x n 결과 행렬
    """
    n = len(A)
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

# ===================================================================
# 되돌이 나란한 행렬 곱하기
# ===================================================================

def matmul_recursive(A, B):
    """되돌이 나누어 정복하기로 행렬을 곱한다.

    인수:
        A: n x n matrix (list of lists)
        B: n x n matrix (list of lists)

    반환값:
        C: n x n 결과 행렬
    """
    n = len(A)
    if n == 1:
        return [[A[0][0] * B[0][0]]]

    mid = n // 2
    a11, a12, a21, a22 = _split(A, mid)
    b11, b12, b21, b22 = _split(B, mid)

    # 되돌이 곱하기 8번(나란히 할 수 있음)
    c11 = _add(_matmul_rec(a11, b11), _matmul_rec(a12, b21))
    c12 = _add(_matmul_rec(a11, b12), _matmul_rec(a12, b22))
    c21 = _add(_matmul_rec(a21, b11), _matmul_rec(a22, b21))
    c22 = _add(_matmul_rec(a21, b12), _matmul_rec(a22, b22))

    return _merge(c11, c12, c21, c22)


def _matmul_rec(A, B):
    """안쪽 되돌이 곱하기."""
    return matmul_recursive(A, B)


def _split(M, mid):
    """행렬 M을 사분면 넷으로 가른다."""
    n = len(M)
    top_left = [row[:mid] for row in M[:mid]]
    top_right = [row[mid:] for row in M[:mid]]
    bot_left = [row[:mid] for row in M[mid:]]
    bot_right = [row[mid:] for row in M[mid:]]
    return top_left, top_right, bot_left, bot_right


def _add(A, B):
    """원소마다의 행렬 더하기."""
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))]
            for i in range(n)]


def _merge(c11, c12, c21, c22):
    """사분면 넷을 행렬 하나로 합친다."""
    top = [c11[i] + c12[i] for i in range(len(c11))]
    bot = [c21[i] + c22[i] for i in range(len(c21))]
    return top + bot

# ===================================================================
# 메인
# ===================================================================

if __name__ == "__main__":
    import math

    A = [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12],
         [13, 14, 15, 16]]

    B = [[1, 0, 0, 1],
         [0, 1, 1, 0],
         [1, 1, 0, 0],
         [0, 0, 1, 1]]

    C_naive = matmul_naive(A, B)
    C_recur = matmul_recursive(A, B)

    print("A x B (naive):")
    for row in C_naive:
        print(f"  {row}")

    print("\nA x B (recursive):")
    for row in C_recur:
        print(f"  {row}")

    print(f"\nResults match: {C_naive == C_recur}")

    # 일-뻗음 간추림
    n = len(A)
    work = n ** 3
    span_loop = math.ceil(math.log2(n))
    span_recur = math.ceil(math.log2(n)) ** 2
    print(f"\nn = {n}")
    print(f"Work T_1 = O(n^3) = {work}")
    print(f"Span (loop parallel):      O(log n) = {span_loop}")
    print(f"Span (recursive parallel):  O(log^2 n) = {span_recur}")
```

**출력:**
```
A x B (naive):
  [4, 6, 6, 5]
  [12, 14, 14, 13]
  [20, 22, 22, 21]
  [28, 30, 30, 29]

A x B (recursive):
  [4, 6, 6, 5]
  [12, 14, 14, 13]
  [20, 22, 22, 21]
  [28, 30, 30, 29]

결과가 맞는가: True

n = 4
Work T_1 = O(n^3) = 64
Span (loop parallel):      O(log n) = 2
Span (recursive parallel):  O(log^2 n) = 4
```

## 복잡도 요약

| 길 | 일 $T_1$ | 뻗음 $T_\infty$ | 나란함 |
|---|---|---|---|
| 되돌이 나란히 | $O(n^3)$ | $O(\log n)$ | $O(n^3 / \log n)$ |
| 되돌이 나란히 | $O(n^3)$ | $O(\log^2 n)$ | $O(n^3 / \log^2 n)$ |
| 슈트라센 + 나란히 | $O(n^{2.807})$ | $O(\log^2 n)$ | $O(n^{2.807} / \log^2 n)$ |

!!! note "슈트라센 알고리즘"
    슈트라센 알고리즘은 되돌이 곱하기를 8번 대신 7번 써서 일을 $O(n^{\log_2 7}) \approx O(n^{2.807})$으로 줄인다. 부분 문제 7개가 여전히 나란히 돌고 핵심 길에 가지 하나만 있으므로 뻗음은 $O(\log^2 n)$ 그대로다.

## 참고 문헌

- Cormen, T. H. et al. *Introduction to Algorithms*, 27장(여러 실 알고리즘).
- Grama, A. et al. *Introduction to Parallel Computing*.


## 연습문제

**연습문제 1.**
나란한 행렬 곱하기 알고리즘과 그 일-뻗음 살피기를 밝혀라.

??? success "연습문제 1 풀이"
    어수룩한 나란한 판: $C_{ij} = \sum_k A_{ik} B_{kj}$. $(i,j)$마다 곱 $n$개를 나란히 셈한 뒤 나란한 줄임으로 더한다. 일: $O(n^3)$. 뻗음: 줄임에 $O(\log n)$. 나란함: $O(n^3/\log n)$. 셈틀 $n^3$개면 때가 $O(\log n)$이다. 나란한 슈트라센: 일 $O(n^{2.807})$, 뻗음 $O(\log^2 n)$. 행렬 곱하기의 나란함이 커서 GPU이 이를 잘한다.

---

**연습문제 2.**
자료 나란히 하는 깊은 배움 익히기에서 행렬 곱하기가 어떻게 쪼개지는지 밝혀라.

??? success "연습문제 2 풀이"
    자료 나란히 익히기에서 GPU마다 다른 작은 묶음 조각으로 앞으로 가기와 거꿀 가기를 셈한다. 행렬 곱하기(선형 층)가 셈의 대부분이다. GPU마다 같은 무게 행렬 $W$에 다른 들임 행렬 $X_i$을 곱한다: $Y_i = W X_i$. 이는 GPU끼리 민망할 만큼 나란하다. 기울기 모으기(GPU를 가로지르는 $\partial L/\partial W$의 all-reduce)가 주고받기 덧짐이다. 셈과 주고받기의 비는 $O(n^2)$(행렬 곱하기) 대 $O(n^2)$(기울기 크기)이므로 묶음이 클수록 효율에 이롭다.

---

**연습문제 3.**
나눠 하는 행렬 곱하기의 캐넌 알고리즘을 밝혀라.

??? success "연습문제 3 풀이"
    캐넌 알고리즘은 $n \times n$ 행렬 $A$과 $B$을 $\sqrt{p} \times \sqrt{p}$ 셈틀 격자에 흩는다. 셈틀마다 행렬마다 $n/\sqrt{p} \times n/\sqrt{p}$ 덩이를 담는다. $\sqrt{p}$걸음 동안 (1) 셈틀마다 제 자리 $A$과 $B$ 덩이를 곱해 $C$에 쌓고, (2) $A$ 덩이를 왼쪽으로 한 열, $B$ 덩이를 위로 한 줄 옮긴다. $\sqrt{p}$걸음 뒤 셈틀마다 제 $C$ 덩이를 셈해 두었다. 주고받기: 걸음마다 $O(n^2/\sqrt{p})$, 모두 $O(n^2)$. 때: $O(n^3/p + n^2/\sqrt{p})$.

---

**연습문제 4.**
나눠 하는 행렬 곱하기의 주고받기 아래 가둠은 무엇인가?

??? success "연습문제 4 풀이"
    저마다 그 자리 기억이 $M = n^2/p$인 셈틀 $p$개에서 $n \times n$ 행렬을 곱할 때 주고받기 아래 가둠(홍-쿵, 아이러니-톨레도-티스킨)은 $\Omega(n^3/(p\sqrt{M})) = \Omega(n^2/\sqrt{p})$낱말이다. 곧 주고받기가 $\sqrt{p}$으로 늘어난다. 셈틀을 두 배로 하면 주고받기가 $\sqrt{2}$배가 된다. 캐넌 알고리즘이 이 가둠에 맞닿는다. 2.5차원 행렬 곱하기는 (기억에 사본을 더 두어) 주고받기 $O(n^2/p^{2/3})$을 이루며 기억과 주고받기를 맞바꾼다.