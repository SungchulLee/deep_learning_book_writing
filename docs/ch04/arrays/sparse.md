# 희소 배열

현실의 행렬 중에는 대부분이 0인 것이 많다. 자연어 처리의 단어-문서 빈도 행렬은 행이 100,000개, 열이 50,000개(항목 50억 개)일 수 있지만 0이 아닌 단어-문서 쌍은 아주 일부에 지나지 않는다. 그런 행렬을 조밀한 2차원 배열로 저장하면 수십억 개의 0에 메모리를 낭비한다. **희소 배열**은 0이 아닌 원소와 그 위치만 저장하여 이 구조를 활용하며, 메모리 사용량과 계산 시간을 $O(mn)$에서 $O(\text{nnz})$으로 줄인다. 여기서 $\text{nnz}$은 0이 아닌 항목의 개수이다.

## 희소성과 그것이 중요해지는 때

행렬 $A \in \mathbb{R}^{m \times n}$의 **희소도**는 항목 중 0인 것의 비율이다.

$$
\text{sparsity} = 1 - \frac{\text{nnz}}{m \cdot n}
$$

$\text{nnz} \ll m \cdot n$이면 그 행렬을 희소하다고 본다. 희소성이 생기는 흔한 원천은 다음과 같다.

- **그래프 인접 행렬**: 꼭짓점이 $n$개, 변이 $e$개인 그래프는 $n \times n$ 행렬에서 (무향일 때) $\text{nnz} = 2e$이다. 실제 네트워크는 대부분 $e = O(n)$이어서 $n$이 크면 행렬의 희소도가 99.9%를 넘는다.
- **원-핫 부호화**: 토큰이 50,000개인 어휘는 50,000개 중 정확히 1개만 0이 아닌 벡터를 만든다.
- **자연어 처리의 특징 행렬**: 각 문서가 어휘의 아주 일부만 쓰므로 단어 주머니나 TF-IDF 표현은 극도로 희소하다.

## 좌표 형식 (COO)

가장 단순한 희소 형식은 0이 아닌 각 항목을 세 쌍 $(i, j, v)$으로 저장한다. 여기서 $i$은 행 인덱스, $j$은 열 인덱스, $v$은 값이다.

**저장:** 길이가 $\text{nnz}$인 배열 세 개.

- `row_indices`: $[i_1, i_2, \ldots, i_{\text{nnz}}]$
- `col_indices`: $[j_1, j_2, \ldots, j_{\text{nnz}}]$
- `values`: $[v_1, v_2, \ldots, v_{\text{nnz}}]$

**공간 복잡도:** $O(\text{nnz})$이며, 구체적으로는 수 $3 \cdot \text{nnz}$개이다.

??? example "COO 표현"

    다음 행렬은

    $$
    A = \begin{pmatrix} 0 & 0 & 3 \\ 4 & 0 & 0 \\ 0 & 5 & 6 \end{pmatrix}
    $$

    $\text{nnz} = 4$이며 다음과 같이 저장된다.

    | 행 | 열 | 값 |
    |-----|-----|-------|
    | 0   | 2   | 3     |
    | 1   | 0   | 4     |
    | 2   | 1   | 5     |
    | 2   | 2   | 6     |

**장점:** 만들기 간단하고 새 항목을 추가하기 쉽다. **단점:** 특정 원소 $(i, j)$에 접근하려면 모든 항목을 훑어야 하므로 조회에 $O(\text{nnz})$ 시간이 든다.

## 압축 희소 행 (CSR)

CSR은 희소 행렬 연산에 가장 널리 쓰이는 형식이다. 행 인덱스를 각 행의 항목이 시작하는 곳을 가리키는 포인터 배열로 대체하여 압축한다.

**저장:** 배열 세 개.

- `values`: 길이 $\text{nnz}$ — 0이 아닌 값들을 행 단위로 저장한다.
- `col_indices`: 길이 $\text{nnz}$ — 0이 아닌 각 값의 열 인덱스.
- `row_ptr`: 길이 $m + 1$ — `row_ptr[i]`은 행 $i$이 시작하는 `values`의 인덱스이다. 행 $i$의 항목은 `values[row_ptr[i] : row_ptr[i+1]]`이다.

**공간 복잡도:** $O(\text{nnz} + m)$이며, 구체적으로는 수 $2 \cdot \text{nnz} + (m + 1)$개이다.

??? example "CSR 표현"

    같은 행렬 $A$에 대해 다음과 같다.

    ```
    values     = [3, 4, 5, 6]
    col_indices = [2, 0, 1, 2]
    row_ptr    = [0, 1, 2, 4]
    ```

    - 행 0: `values[0:1]` = `[3]`, 열은 `[2]`
    - 행 1: `values[1:2]` = `[4]`, 열은 `[0]`
    - 행 2: `values[2:4]` = `[5, 6]`, 열은 `[1, 2]`

**장점:** $O(1)$의 효율적인 행 자르기, 빠른 행렬-벡터 곱, 간결한 저장. **단점:** 0이 아닌 항목을 새로 넣으려면 배열을 밀어야 해서 비싸다.

## 압축 희소 열 (CSC)

CSC는 CSR의 열 중심 대응물이다. 열 포인터 배열과 함께 항목을 열 단위로 저장한다.

**저장:**

- `values`: 길이 $\text{nnz}$
- `row_indices`: 길이 $\text{nnz}$
- `col_ptr`: 길이 $n + 1$

**공간 복잡도:** $O(\text{nnz} + n)$.

특정 선형대수 해법처럼 열 자르기가 잦을 때는 CSC가 선호된다.

## 복잡도 비교

| 연산            | 조밀          | COO               | CSR               |
|----------------------|----------------|-------------------|-------------------|
| 저장              | $O(mn)$        | $O(\text{nnz})$   | $O(\text{nnz}+m)$ |
| 접근 $(i,j)$       | $O(1)$         | $O(\text{nnz})$   | $O(\log d_i)$     |
| 행 자르기            | $O(n)$         | $O(\text{nnz})$   | 위치 찾기에 $O(1)$  |
| 행렬-벡터 곱   | $O(mn)$        | $O(\text{nnz})$   | $O(\text{nnz})$   |
| 0 아닌 항목 삽입   | $O(1)$         | 상각 $O(1)$  | $O(\text{nnz})$   |

여기서 $d_i$은 행 $i$에서 0이 아닌 항목의 개수이다(그 행의 열 인덱스 안에서 이진 탐색을 한다).

!!! tip "희소 형식을 언제 쓸 것인가"

    희소 형식은 $\text{nnz} \ll mn$일 때에만 메모리와 계산을 아낀다. 어림잡은 경험칙으로, 항목의 10~20%를 넘는 부분이 0이 아니라면 조밀한 저장이 더 빠른 경우가 많다. 인덱스를 관리하는 부담이 없고 최적화된 BLAS 루틴과 캐시 지역성의 덕을 보기 때문이다.

## 파이썬 시연

```python
"""SciPy로 희소 행렬의 형식을 보인다."""

import numpy as np
from scipy import sparse

# === 희소 행렬 만들기 ===
dense = np.array([[0, 0, 3],
                  [4, 0, 0],
                  [0, 5, 6]])

# === COO 형식 ===
coo = sparse.coo_matrix(dense)
print("COO format:")
print(f"  row:  {coo.row}")
print(f"  col:  {coo.col}")
print(f"  data: {coo.data}")
print(f"  nnz:  {coo.nnz}")

# === CSR 형식 ===
csr = sparse.csr_matrix(dense)
print("\nCSR format:")
print(f"  data:    {csr.data}")
print(f"  indices: {csr.indices}")
print(f"  indptr:  {csr.indptr}")

# === 공간 절약 ===
n = 10000
big_sparse = sparse.random(n, n, density=0.001, format='csr')
dense_bytes = n * n * 8  # float64
sparse_bytes = big_sparse.data.nbytes + big_sparse.indices.nbytes + big_sparse.indptr.nbytes
print(f"\n{n}x{n} matrix with 0.1% density:")
print(f"  Dense:  {dense_bytes / 1e6:.1f} MB")
print(f"  Sparse: {sparse_bytes / 1e6:.2f} MB")
print(f"  Ratio:  {dense_bytes / sparse_bytes:.0f}x")
```

**출력:**
```
COO format:
  row:  [0 1 2 2]
  col:  [2 0 1 2]
  data: [3 4 5 6]
  nnz:  4

CSR format:
  data:    [3 4 5 6]
  indices: [2 0 1 2]
  indptr:  [0 1 2 4]

10000x10000 matrix with 0.1% density:
  Dense:  800.0 MB
  Sparse: 1.28 MB
  Ratio:  625x
```

## 참고 문헌

- [Introduction to Algorithms (CLRS), Chapter 10](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)


## 연습문제

**연습문제 1.**
희소 배열에 대해 삽입, 삭제, 탐색, 접근 연산의 시간 복잡도를 진술하라.

??? success "연습문제 1 풀이"
    복잡도는 구체적인 구현(배열 기반이냐 연결 기반이냐)에 달려 있다. 배열 기반은 접근이 $O(1)$이고 임의의 위치에서의 삽입·삭제가 $O(n)$이다. 연결 기반은 이미 아는 위치에서의 삽입·삭제가 $O(1)$이고 탐색·접근이 $O(n)$이다. 어떤 연산이 주를 이루느냐에 따라 선택이 갈린다.

---

**연습문제 2.**
원소 6개로 희소 배열을(를) 따라가며 각 연산 후의 자료구조 상태를 보여라.

??? success "연습문제 2 풀이"
    구조에 삽입, 접근, 삭제를 차례로 수행하라. 각 단계마다 (연결 구조라면) 포인터를, (배열 기반이라면) 배열의 내용을 보이며 구조가 불변식을 어떻게 유지하는지 나타내라.

---

**연습문제 3.**
희소 배열이(가) PyTorch의 텐서 저장과 어떻게 관련되는지 설명하라. 자료구조의 선택이 메모리 배치와 캐시 성능에 어떤 영향을 주는가?

??? success "연습문제 3 풀이"
    PyTorch 텐서는 캐시에 효율적으로 접근할 수 있도록 연속된 배열로 저장된다. 연결 구조는 autograd 그래프를 훑는 데 내부적으로 쓰인다. 이 선택은 메모리 사용량(배열에는 포인터 부담이 없다)과 접근 양상(캐시 지역성 덕분에 순차적인 배열 접근이 연결 리스트 순회보다 10~100배 빠르다)에 모두 영향을 준다.

---

**연습문제 4.**
반복문 불변식을 사용하여 희소 배열의 주요 연산의 시간 복잡도를 증명하라.

??? success "연습문제 4 풀이"
    알고리즘의 반복문이 유지하는 불변식을 진술하라. 초기화, 유지, 종료를 증명하라. 이 불변식으로부터 반복문이 명시된 횟수 안에 끝남이 따라 나오며, 이로써 복잡도의 상계가 확립된다. $\square$