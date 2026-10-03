# 생성 방법

이 스크립트는 텐서를 만드는 여러 방법을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
텐서 만들기와 초기화 방법.

Covers:
- 기본 생성기: torch.tensor, torch.as_tensor, torch.from_numpy
- 상수 텐서: zeros, ones, full, empty
- 항등과 대각: eye, diag
- 마구잡이 텐서: rand, randn, randint, randperm
- 범위 텐서: arange, linspace, logspace
- 모양을 본뜨는 생성기: zeros_like, ones_like 따위
- 기기와 데이터 클래스 밝히기
"""

import torch
import numpy as np

# ========================================================================
# 메인
# ========================================================================

def header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def main():
    torch.manual_seed(42)

    # -------------------------------------------------------------------------
    header("torch.tensor: creates copy from data")
    data = [[1, 2], [3, 4]]
    t = torch.tensor(data)
    print("torch.tensor(data):\n", t)
    print("dtype:", t.dtype, "| device:", t.device)
    
    # dtype 지정
    t_float = torch.tensor(data, dtype=torch.float64)
    print("With dtype=float64:", t_float.dtype)

    # -------------------------------------------------------------------------
    header("torch.as_tensor: may share memory with input")
    np_array = np.array([[1, 2], [3, 4]])
    t = torch.as_tensor(np_array)  # Shares memory if possible
    print("torch.as_tensor(numpy):\n", t)
    
    # 수정이 양쪽에 영향을 준다
    np_array[0, 0] = 999
    print("After modifying numpy, torch tensor:", t[0, 0].item())

    # -------------------------------------------------------------------------
    header("torch.from_numpy: shares memory with numpy")
    np_array = np.array([1.0, 2.0, 3.0])
    t = torch.from_numpy(np_array)
    print("torch.from_numpy:", t)
    print("Shares memory:", np.shares_memory(np_array, t.numpy()))

    # -------------------------------------------------------------------------
    header("torch.zeros: all zeros")
    z = torch.zeros(3, 4)
    print("zeros(3, 4):\n", z)
    
    # dtype과 device 지정
    z_int = torch.zeros(2, 3, dtype=torch.int64)
    print("zeros with dtype=int64:\n", z_int)

    # -------------------------------------------------------------------------
    header("torch.ones: all ones")
    o = torch.ones(2, 3)
    print("ones(2, 3):\n", o)

    # -------------------------------------------------------------------------
    header("torch.full: fill with specific value")
    f = torch.full((3, 3), fill_value=7.5)
    print("full((3,3), 7.5):\n", f)

    # -------------------------------------------------------------------------
    header("torch.empty: uninitialized memory (fast but random values)")
    e = torch.empty(2, 3)
    print("empty(2, 3) (uninitialized):\n", e)
    print("⚠️  Values are random - don't rely on them!")

    # -------------------------------------------------------------------------
    header("torch.eye: identity matrix")
    eye = torch.eye(4)
    print("eye(4):\n", eye)
    
    # 정사각이 아닌 단위 행렬
    eye_rect = torch.eye(3, 5)
    print("eye(3, 5):\n", eye_rect)

    # -------------------------------------------------------------------------
    header("torch.diag: create diagonal matrix or extract diagonal")
    # 벡터로 대각 행렬 만들기
    v = torch.tensor([1., 2., 3., 4.])
    diag_mat = torch.diag(v)
    print("diag(vector):\n", diag_mat)
    
    # 행렬에서 대각 성분 뽑아내기
    mat = torch.randn(4, 4)
    diagonal = torch.diag(mat)
    print("diag(matrix):", diagonal)
    
    # 어긋난 대각선
    upper_diag = torch.diag(mat, diagonal=1)  # One above main diagonal
    print("Upper diagonal:", upper_diag)

    # -------------------------------------------------------------------------
    header("torch.arange: sequence like Python range")
    seq = torch.arange(10)
    print("arange(10):", seq)
    
    seq2 = torch.arange(2, 10, 2)  # start, end, step
    print("arange(2, 10, 2):", seq2)
    
    seq3 = torch.arange(0, 1, 0.1)  # Float steps
    print("arange(0, 1, 0.1):", seq3)

    # -------------------------------------------------------------------------
    header("torch.linspace: linearly spaced values")
    lin = torch.linspace(0, 10, steps=11)
    print("linspace(0, 10, 11):", lin)
    
    # 범위를 그릴 때 유용하다
    x = torch.linspace(-3.14, 3.14, steps=7)
    print("linspace(-π, π, 7):", x)

    # -------------------------------------------------------------------------
    header("torch.logspace: logarithmically spaced values")
    log = torch.logspace(0, 3, steps=4)  # 10^0 to 10^3
    print("logspace(0, 3, 4):", log)
    print("Exponentially increasing: 10^0, 10^1, 10^2, 10^3")

    # -------------------------------------------------------------------------
    header("torch.rand: uniform [0, 1)")
    r = torch.rand(3, 4)
    print("rand(3, 4):\n", r)
    print("Range: [0, 1)")

    # -------------------------------------------------------------------------
    header("torch.randn: standard normal N(0, 1)")
    rn = torch.randn(3, 4)
    print("randn(3, 4):\n", rn)
    print("Distribution: N(0, 1)")
    
    # 사용자 지정 평균과 표준편차
    custom = torch.randn(1000) * 2.5 + 10  # N(10, 2.5^2)
    print(f"Custom N(10, 2.5): mean={custom.mean():.2f}, std={custom.std():.2f}")

    # -------------------------------------------------------------------------
    header("torch.randint: random integers")
    ri = torch.randint(low=0, high=10, size=(3, 4))
    print("randint(0, 10, (3,4)):\n", ri)
    print("Range: [0, 10)")

    # -------------------------------------------------------------------------
    header("torch.randperm: random permutation")
    perm = torch.randperm(10)
    print("randperm(10):", perm)
    print("Useful for shuffling indices")

    # -------------------------------------------------------------------------
    header("torch.multinomial: sample from multinomial distribution")
    # 확률 가중치(합이 1일 필요는 없다)
    weights = torch.tensor([1., 2., 3., 4.])  # Higher numbers = more likely
    samples = torch.multinomial(weights, num_samples=10, replacement=True)
    print("Weights:", weights)
    print("Samples:", samples)
    print("Higher indices (3) should appear more often")

    # -------------------------------------------------------------------------
    header("_like constructors: same shape as another tensor")
    template = torch.randn(3, 4)
    print("Template shape:", template.shape)
    
    z = torch.zeros_like(template)
    print("zeros_like:", z.shape, z.dtype)
    
    o = torch.ones_like(template)
    print("ones_like:", o.shape, o.dtype)
    
    r = torch.rand_like(template)
    print("rand_like:", r.shape, r.dtype)
    
    # dtype을 덮어쓸 수 있다
    z_int = torch.zeros_like(template, dtype=torch.int32)
    print("zeros_like with dtype override:", z_int.dtype)

    # -------------------------------------------------------------------------
    header("Device specification")
    cpu_tensor = torch.randn(3, 3, device='cpu')
    print("CPU tensor device:", cpu_tensor.device)
    
    if torch.cuda.is_available():
        gpu_tensor = torch.randn(3, 3, device='cuda')
        print("GPU tensor device:", gpu_tensor.device)
        
        # 장치 간 전송
        moved = cpu_tensor.to('cuda')
        print("Moved to GPU:", moved.device)
    else:
        print("CUDA not available, skipping GPU examples")

    # -------------------------------------------------------------------------
    header("requires_grad specification")
    # autograd를 켠 텐서 만들기
    x = torch.randn(3, 4, requires_grad=True)
    print("requires_grad:", x.requires_grad)
    print("is_leaf:", x.is_leaf)
    
    # 생성 후에 설정할 수도 있다
    y = torch.randn(3, 4)
    y.requires_grad_(True)
    print("Set requires_grad after:", y.requires_grad)

    # -------------------------------------------------------------------------
    header("torch.empty_like vs torch.zeros_like performance")
    import time
    
    large_template = torch.randn(1000, 1000)
    
    # empty_like가 더 빠르다(초기화가 없다)
    start = time.time()
    for _ in range(1000):
        _ = torch.empty_like(large_template)
    empty_time = time.time() - start
    
    start = time.time()
    for _ in range(1000):
        _ = torch.zeros_like(large_template)
    zeros_time = time.time() - start
    
    print(f"empty_like: {empty_time:.4f}s")
    print(f"zeros_like: {zeros_time:.4f}s")
    print(f"Speedup: {zeros_time/empty_time:.2f}x")
    print("⚠️  Use empty only when you'll immediately overwrite values")

    # -------------------------------------------------------------------------
    header("Complex number tensors")
    # 복소수 텐서 만들기
    real = torch.tensor([1., 2., 3.])
    imag = torch.tensor([4., 5., 6.])
    c = torch.complex(real, imag)
    print("Complex tensor:", c)
    print("dtype:", c.dtype)
    
    # 직접 생성
    c2 = torch.tensor([1+2j, 3+4j, 5+6j])
    print("Direct complex:", c2)

    # -------------------------------------------------------------------------
    header("Sparse tensors (briefly)")
    # 희소 COO 텐서 만들기
    indices = torch.tensor([[0, 1, 2], [1, 0, 2]])  # (row, col) indices
    values = torch.tensor([3., 4., 5.])
    sparse = torch.sparse_coo_tensor(indices, values, (3, 3))
    print("Sparse tensor:\n", sparse)
    print("Dense representation:\n", sparse.to_dense())

    # -------------------------------------------------------------------------
    header("Cloning and copying")
    original = torch.randn(3, 3)
    
    # clone()은 복사본을 만든다
    copy = original.clone()
    print("Shares storage (clone):", original.untyped_storage().data_ptr() == copy.untyped_storage().data_ptr())
    
    # 경사가 없는 복사본을 만들려면 detach().clone()
    x = torch.randn(3, requires_grad=True)
    y = x.detach().clone()
    print("Detached clone requires_grad:", y.requires_grad)

    # -------------------------------------------------------------------------
    header("Quick reference: creation functions")
    print("\nZero-value tensors:")
    print("  torch.zeros(shape)           - All zeros")
    print("  torch.zeros_like(tensor)     - Zeros with same shape")
    
    print("\nOne-value tensors:")
    print("  torch.ones(shape)            - All ones")
    print("  torch.ones_like(tensor)      - Ones with same shape")
    print("  torch.full(shape, value)     - All same value")
    
    print("\nRandom tensors:")
    print("  torch.rand(shape)            - Uniform [0, 1)")
    print("  torch.randn(shape)           - Normal N(0, 1)")
    print("  torch.randint(low, high, sz) - Random integers")
    print("  torch.randperm(n)            - Random permutation")
    
    print("\nSequential tensors:")
    print("  torch.arange(start, end, step) - Like Python range")
    print("  torch.linspace(start, end, n)  - n evenly spaced")
    print("  torch.logspace(start, end, n)  - n log-spaced")
    
    print("\nStructured tensors:")
    print("  torch.eye(n)                 - Identity matrix")
    print("  torch.diag(vector)           - Diagonal matrix")
    
    print("\nFrom data:")
    print("  torch.tensor(data)           - Copy from list/array")
    print("  torch.from_numpy(array)      - Share memory with numpy")
    print("  torch.as_tensor(data)        - Share memory if possible")

if __name__ == "__main__":
    main()
```

??? note "전체 출력 (232줄)"

    ```

    ================================================================================
    torch.tensor: creates copy from data
    ================================================================================
    torch.tensor(data):
     tensor([[1, 2],
            [3, 4]])
    dtype: torch.int64 | device: cpu
    With dtype=float64: torch.float64

    ================================================================================
    torch.as_tensor: may share memory with input
    ================================================================================
    torch.as_tensor(numpy):
     tensor([[1, 2],
            [3, 4]])
    After modifying numpy, torch tensor: 999

    ================================================================================
    torch.from_numpy: shares memory with numpy
    ================================================================================
    torch.from_numpy: tensor([1., 2., 3.], dtype=torch.float64)
    Shares memory: True

    ================================================================================
    torch.zeros: all zeros
    ================================================================================
    zeros(3, 4):
     tensor([[0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]])
    zeros with dtype=int64:
     tensor([[0, 0, 0],
            [0, 0, 0]])

    ================================================================================
    torch.ones: all ones
    ================================================================================
    ones(2, 3):
     tensor([[1., 1., 1.],
            [1., 1., 1.]])

    ================================================================================
    torch.full: fill with specific value
    ================================================================================
    full((3,3), 7.5):
     tensor([[7.5000, 7.5000, 7.5000],
            [7.5000, 7.5000, 7.5000],
            [7.5000, 7.5000, 7.5000]])

    ================================================================================
    torch.empty: uninitialized memory (fast but random values)
    ================================================================================
    empty(2, 3) (uninitialized):
     tensor([[0., 0., 0.],
            [0., 0., 0.]])
    ⚠️  Values are random - don't rely on them!

    ================================================================================
    torch.eye: identity matrix
    ================================================================================
    eye(4):
     tensor([[1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]])
    eye(3, 5):
     tensor([[1., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 0.]])

    ================================================================================
    torch.diag: create diagonal matrix or extract diagonal
    ================================================================================
    diag(vector):
     tensor([[1., 0., 0., 0.],
            [0., 2., 0., 0.],
            [0., 0., 3., 0.],
            [0., 0., 0., 4.]])
    diag(matrix): tensor([ 1.9269, -1.2345, -0.3925,  0.7624])
    Upper diagonal: tensor([ 1.4873, -0.0431, -1.4036])

    ================================================================================
    torch.arange: sequence like Python range
    ================================================================================
    arange(10): tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    arange(2, 10, 2): tensor([2, 4, 6, 8])
    arange(0, 1, 0.1): tensor([0.0000, 0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000,
            0.9000])

    ================================================================================
    torch.linspace: linearly spaced values
    ================================================================================
    linspace(0, 10, 11): tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])
    linspace(-π, π, 7): tensor([-3.1400e+00, -2.0933e+00, -1.0467e+00, -1.1921e-07,  1.0467e+00,
             2.0933e+00,  3.1400e+00])

    ================================================================================
    torch.logspace: logarithmically spaced values
    ================================================================================
    logspace(0, 3, 4): tensor([   1.,   10.,  100., 1000.])
    Exponentially increasing: 10^0, 10^1, 10^2, 10^3

    ================================================================================
    torch.rand: uniform [0, 1)
    ================================================================================
    rand(3, 4):
     tensor([[0.8854, 0.5739, 0.2666, 0.6274],
            [0.2696, 0.4414, 0.2969, 0.8317],
            [0.1053, 0.2695, 0.3588, 0.1994]])
    Range: [0, 1)

    ================================================================================
    torch.randn: standard normal N(0, 1)
    ================================================================================
    randn(3, 4):
     tensor([[-0.9890,  0.9580,  1.3221,  0.8172],
            [-0.7658, -0.7506,  1.3525,  0.6863],
            [-0.3278,  0.7950,  0.2815,  0.0562]])
    Distribution: N(0, 1)
    Custom N(10, 2.5): mean=10.07, std=2.40

    ================================================================================
    torch.randint: random integers
    ================================================================================
    randint(0, 10, (3,4)):
     tensor([[0, 8, 2, 5],
            [3, 2, 6, 6],
            [3, 4, 9, 4]])
    Range: [0, 10)

    ================================================================================
    torch.randperm: random permutation
    ================================================================================
    randperm(10): tensor([0, 6, 1, 9, 7, 3, 2, 5, 4, 8])
    Useful for shuffling indices

    ================================================================================
    torch.multinomial: sample from multinomial distribution
    ================================================================================
    Weights: tensor([1., 2., 3., 4.])
    Samples: tensor([0, 1, 2, 1, 1, 3, 2, 3, 2, 0])
    Higher indices (3) should appear more often

    ================================================================================
    _like constructors: same shape as another tensor
    ================================================================================
    Template shape: torch.Size([3, 4])
    zeros_like: torch.Size([3, 4]) torch.float32
    ones_like: torch.Size([3, 4]) torch.float32
    rand_like: torch.Size([3, 4]) torch.float32
    zeros_like with dtype override: torch.int32

    ================================================================================
    Device specification
    ================================================================================
    CPU tensor device: cpu
    CUDA not available, skipping GPU examples

    ================================================================================
    requires_grad specification
    ================================================================================
    requires_grad: True
    is_leaf: True
    Set requires_grad after: True

    ================================================================================
    torch.empty_like vs torch.zeros_like performance
    ================================================================================
    empty_like: 0.0018s
    zeros_like: 0.1779s
    Speedup: 98.33x
    ⚠️  Use empty only when you'll immediately overwrite values

    ================================================================================
    Complex number tensors
    ================================================================================
    Complex tensor: tensor([1.+4.j, 2.+5.j, 3.+6.j])
    dtype: torch.complex64
    Direct complex: tensor([1.+2.j, 3.+4.j, 5.+6.j])

    ================================================================================
    Sparse tensors (briefly)
    ================================================================================
    Sparse tensor:
     tensor(indices=tensor([[0, 1, 2],
                           [1, 0, 2]]),
           values=tensor([3., 4., 5.]),
           size=(3, 3), nnz=3, layout=torch.sparse_coo)
    Dense representation:
     tensor([[0., 3., 0.],
            [4., 0., 0.],
            [0., 0., 5.]])

    ================================================================================
    Cloning and copying
    ================================================================================
    Shares storage (clone): False
    Detached clone requires_grad: False

    ================================================================================
    Quick reference: creation functions
    ================================================================================

    Zero-value tensors:
      torch.zeros(shape)           - All zeros
      torch.zeros_like(tensor)     - Zeros with same shape

    One-value tensors:
      torch.ones(shape)            - All ones
      torch.ones_like(tensor)      - Ones with same shape
      torch.full(shape, value)     - All same value

    Random tensors:
      torch.rand(shape)            - Uniform [0, 1)
      torch.randn(shape)           - Normal N(0, 1)
      torch.randint(low, high, sz) - Random integers
      torch.randperm(n)            - Random permutation

    Sequential tensors:
      torch.arange(start, end, step) - Like Python range
      torch.linspace(start, end, n)  - n evenly spaced
      torch.logspace(start, end, n)  - n log-spaced

    Structured tensors:
      torch.eye(n)                 - Identity matrix
      torch.diag(vector)           - Diagonal matrix

    From data:
      torch.tensor(data)           - Copy from list/array
      torch.from_numpy(array)      - Share memory with numpy
      torch.as_tensor(data)        - Share memory if possible
    ```
## 2. 논의

경사 추적을 제어하는 것은 정확성과 성능 모두에 필수적이다. `torch.no_grad()` 컨텍스트 관리자는 매개변수 갱신이나 추론처럼 계산 그래프에 포함되어서는 안 되는 연산에 대해 autograd를 끈다. `.detach()` 메서드는 저장소는 공유하지만 그래프와는 분리된 텐서를 만들며, 값을 기록하거나 NumPy로 변환할 때 유용하다.

CPU 텐서에서 PyTorch와 NumPy의 상호 운용은 매끄럽다. `torch.from_numpy()`는 배열과 메모리를 공유하는 텐서를 만들고, `torch.tensor()`는 항상 복사한다. 어떤 연산이 저장소를 공유하고 어떤 연산이 독립적인 복사본을 만드는지 이해하는 것이 미묘한 버그를 피하는 데 결정적이다.

GPU 가속은 텐서 연산, 특히 신경망 계산을 지배하는 행렬 곱에 대해 몇 자릿수의 속도 향상을 제공한다. `.to(device)`로 텐서와 모델을 GPU로 옮기는 것은 간단하지만, 성능을 유지하려면 CPU-GPU 사이의 데이터 전송을 최소화하는 것이 결정적이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
NumPy 배열을 만들고 `torch.from_numpy()`로 PyTorch 텐서로 변환한 뒤, 원래 배열을 수정하여 텐서도 함께 바뀌는지 확인하라.

</div>

??? success "연습문제 1 풀이"
    ```python
    import numpy as np
    arr = np.array([1.0, 2.0, 3.0])
    t = torch.from_numpy(arr)
    arr[0] = 99.0
    print(t)  # tensor([99.,  2.,  3.]) -- shared memory
    ```

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`torch.as_tensor()`가 언제 데이터를 복사하고 언제 메모리를 공유하는지 설명하라. 어떤 조건에서 복사가 일어나는가?

</div>

??? success "연습문제 2 풀이"
    `torch.as_tensor()`는 입력이 스트라이드가 호환되는 쓰기 가능한 NumPy 배열이고 요청한 dtype/device가 일치할 때 메모리를 공유한다. 배열이 읽기 전용이거나, 스트라이드가 음수이거나, dtype 또는 device 변환이 필요할 때는 복사한다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
`requires_grad=True`인 텐서에 `.numpy()`를 호출하면 오류가 나는 이유는 무엇인가? 올바른 변환 방법을 보여라.

</div>

??? success "연습문제 3 풀이"
    ```python
    x = torch.randn(3, requires_grad=True)
    # x.numpy()  # 오류: 경사가 필요한 텐서에는 numpy()를 호출할 수 없다
    x_np = x.detach().cpu().numpy()  # Correct: detach from graph first
    ```

    NumPy에는 autograd 체계가 없으므로, 추적 중인 텐서의 뷰를 노출하면 경사 계산을 망가뜨리는 변경이 일어날 수 있다. `.detach()`는 텐서를 계산 그래프에서 떼어낸다.

## 정리하며

**다룬 것** — 생성 방법

경사 추적을 제어하는 것은 정확성과 성능 모두에 필수적이다.

앞의 연습문제 3개로 직접 확인할 수 있다.
