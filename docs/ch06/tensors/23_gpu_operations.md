# GPU 연산 - CUDA와 장치 관리

CUDA 사용 가능 여부를 확인하고 텐서와 모델을 같은 장치로 옮기는 과정을 살펴본다. 실행 시간과 메모리 지표를 읽을 때 필요한 조건도 함께 설명한다.

## 1. 코드

```python
"""튜토리얼 23: GPU 셈 - CUDA과 기기 다루기"""
import torch
# 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
# 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
torch.manual_seed(0)

# ========================================================================
# 메인
# ========================================================================

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

def main():
    header("1. Checking CUDA Availability")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU")
    
    header("2. Device Selection")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # Specific GPU
        print(f"Specific GPU: {device}")
    
    header("3. Moving Tensors to GPU")
    x = torch.randn(3, 3)
    print(f"CPU tensor device: {x.device}")
    if torch.cuda.is_available():
        x_gpu = x.to(device)
        print(f"GPU tensor device: {x_gpu.device}")
        x_gpu_alt = x.cuda()  # Alternative method
        print(f"Using .cuda(): {x_gpu_alt.device}")
    
    header("4. Creating Tensors Directly on GPU")
    if torch.cuda.is_available():
        x_gpu = torch.randn(3, 3, device=device)
        print(f"Created on GPU: {x_gpu.device}")
        print("This is more efficient than creating on CPU then moving!")
    else:
        print("Create with device parameter: torch.randn(3, 3, device=device)")
    
    header("5. Moving Models to GPU")
    import torch.nn as nn
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    print(f"Model on CPU")
    if torch.cuda.is_available():
        model = model.to(device)
        print(f"Model moved to GPU")
        for name, param in model.named_parameters():
            print(f"  {name}: {param.device}")
    
    header("6. GPU Memory Management")
    if torch.cuda.is_available():
        print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
        x = torch.randn(1000, 1000, device=device)
        print(f"After allocation: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        del x
        torch.cuda.empty_cache()
        print(f"After clearing cache: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    header("7. Performance Comparison")
    print("CPU vs GPU performance (if GPU available):")
    size = 1000
    x = torch.randn(size, size)
    y = torch.randn(size, size)
    
    import time
    start = time.time()
    z_cpu = x @ y
    cpu_time = time.time() - start
    print(f"CPU matrix multiplication: {cpu_time:.4f}s")
    
    if torch.cuda.is_available():
        x_gpu = x.to(device)
        y_gpu = y.to(device)
        torch.cuda.synchronize()  # Wait for GPU
        start = time.time()
        z_gpu = x_gpu @ y_gpu
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        print(f"GPU matrix multiplication: {gpu_time:.4f}s")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
    
    header("8. Best Practices")
    print("""
    GPU을 잘 쓰는 버릇:
    
    1. 될 수 있으면 텐서를 GPU에서 곧바로 만들어라
    2. CPU ↔ GPU 데이터 옮김을 가장 적게 하라
    3. GPU을 가득 채우도록 배치 크기를 키워라
    4. 기억 자리가 모자라면 저장을 비워라
    5. 때를 정확히 재려면 torch.cuda.synchronize()을 써라
    6. 섞인 촘촘함을 써라(튜토리얼 24을 보아라)
    7. torch.cuda.memory_allocated()으로 기억 자리를 지켜보아라
    """)

if __name__ == "__main__":
    main()
```

**출력:**

```

======================================================================
1. Checking CUDA Availability
======================================================================
CUDA available: False
Running on CPU

======================================================================
2. Device Selection
======================================================================
Selected device: cpu

======================================================================
3. Moving Tensors to GPU
======================================================================
CPU tensor device: cpu

======================================================================
4. Creating Tensors Directly on GPU
======================================================================
Create with device parameter: torch.randn(3, 3, device=device)

======================================================================
5. Moving Models to GPU
======================================================================
Model on CPU

======================================================================
6. GPU Memory Management
======================================================================

======================================================================
7. Performance Comparison
======================================================================
CPU vs GPU performance (if GPU available):
CPU matrix multiplication: 0.0038s

======================================================================
8. Best Practices
======================================================================

    GPU을 잘 쓰는 버릇:
    
    1. 될 수 있으면 텐서를 GPU에서 곧바로 만들어라
    2. CPU ↔ GPU 데이터 옮김을 가장 적게 하라
    3. GPU을 가득 채우도록 배치 크기를 키워라
    4. 기억 자리가 모자라면 저장을 비워라
    5. 때를 정확히 재려면 torch.cuda.synchronize()을 써라
    6. 섞인 촘촘함을 써라(튜토리얼 24을 보아라)
    7. torch.cuda.memory_allocated()으로 기억 자리를 지켜보아라
    
```

## 2. 논의

예제는 CUDA를 사용할 수 있으면 GPU를, 그렇지 않으면 CPU를 선택한다. 모델의 파라미터와 계산에 사용하는 텐서는 해당 연산이 요구하는 같은 장치에 있어야 한다. `tensor.to(device)`는 반환값을 사용해야 하며, `model.to(device)`는 등록된 파라미터와 버퍼를 옮긴다.

CUDA 연산은 일반적으로 비동기로 제출된다. CPU 시계로 경과 시간을 잴 때는 측정 경계에서 GPU 작업 완료를 기다려야 한다. 예제의 `torch.cuda.synchronize()`가 이 역할을 한다. 초기화와 워밍업, 데이터 전송을 포함하는지에 따라서도 시간 비교 결과가 달라진다.

GPU는 큰 행렬 연산처럼 병렬성이 높은 작업에서 이득을 줄 수 있다. 작은 연산은 커널 호출과 데이터 전송 비용 때문에 CPU보다 느릴 수도 있다. CPU와 GPU 사이에 반복해서 데이터를 옮기지 않도록 계산 흐름을 구성한다.

### 연산이 실제로 일어나는 곳

왜 GPU를 쓰는지 이해하려면 데이터가 지나는 길을 먼저 보아야 한다.

| 구성 요소 | 역할 |
|---|---|
| CPU | 범용 연산과 프로그램 제어 |
| RAM | CPU가 사용하는 주 메모리 |
| GPU | 병렬성이 높은 텐서 연산 |
| GPU 메모리 | GPU 연산의 입력과 중간 결과 저장 |
| SSD | 데이터와 모델 파일 보관 |

일반적인 외장 GPU 환경에서는 저장 장치의 데이터를 CPU 메모리로 읽은 뒤 GPU 메모리로 전송한다. 계산 성능이 높아도 데이터 읽기나 장치 간 전송이 병목이 될 수 있다. 통합 메모리를 사용하는 장치 등에서는 물리적 메모리 구성이 다르므로 이 경로는 개략적인 설명으로 이해한다.

$2 \times 2$ 행렬은 메모리에 한 줄로 눕는다.

```
A = | 1 | 2 | 3 | 4 |
B = | 1 | 1 | 2 | 3 |
```

이 둘을 곱하면 첫 원소가 $1 \times 1 + 2 \times 2 = 5$이다.

$$
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
\begin{bmatrix} 1 & 1 \\ 2 & 3 \end{bmatrix}
=
\begin{bmatrix} 5 & 7 \\ 11 & 15 \end{bmatrix}
$$

행렬 곱의 결과 원소들은 각각 행과 열의 내적으로 계산할 수 있다. GPU는 이런 계산을 타일과 여러 실행 단위로 나누어 병렬 처리한다. 실제 성능은 병렬성뿐 아니라 메모리 대역폭, 데이터 재사용과 커널 호출 비용에 달려 있다.

- **NumPy**도 BLAS 구현과 설정에 따라 여러 CPU 스레드를 사용한다.
- **GPU**의 활용률은 연산 크기와 커널 구성에 달려 있다.

CUDA는 NVIDIA GPU를 위한 병렬 계산 플랫폼이다. CPU도 여러 코어와 최적화된 라이브러리로 병렬 계산할 수 있으므로, 작은 작업이나 전송 비용이 큰 작업에서 GPU가 항상 빠른 것은 아니다. 이 예제의 CUDA 사용 가능 여부가 MPS 같은 다른 가속기의 사용 가능 여부까지 판정하는 것은 아니다.

```
CPU  --PyTorch-->  CUDA  -->  GPU
```

### 배치 크기와 GPU 활용률

GPU 코어 하나에 이미지 한 장을 대응시키는 구조는 아니다. 커널이 텐서 원소의 계산을 여러 실행 단위에 나누며, 배치 크기·행렬 차원의 정렬·메모리 용량이 효율에 영향을 준다. 2의 거듭제곱 배치는 흔한 선택이지만 필수 조건이나 최적 성능의 보장은 아니다.

!!! tip "가장 중요한 습관"
    `tensor.device`와 모델 파라미터의 장치를 확인한다. GPU가 설치되어 있어도 데이터를 CPU에 둔 채 계산하면 자동으로 CUDA를 사용하는 것은 아니다. 기본 장치는 설정으로 바꿀 수 있으므로 실제 속성을 기준으로 점검한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
$(2, 3)$ 행렬과 $(3, 4)$ 행렬의 곱을 `@`, `torch.mm()`, `torch.matmul()` 세 가지 방법으로 계산하라. 결과가 동일함을 확인하라.

</div>

??? success "연습문제 1 풀이"
    ```python
    A = torch.randn(2, 3)
    B = torch.randn(3, 4)
    r1 = A @ B
    r2 = torch.mm(A, B)
    r3 = torch.matmul(A, B)
    print(torch.allclose(r1, r2) and torch.allclose(r2, r3))  # True
    ```

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
원소별 곱(`*`)과 행렬 곱(`@`)의 차이를 설명하라. 같은 입력에 대해 서로 다른 결과를 내는 예를 들라.

</div>

??? success "연습문제 2 풀이"
    원소별 곱은 대응하는 원소끼리 곱한다. 같은 모양의 행렬 $A, B$에 대해 $(A * B)_{ij} = A_{ij} B_{ij}$이다. 행렬 곱은 안쪽 차원에 대해 더한다. $(A @ B)_{ij} = \sum_k A_{ik} B_{kj}$이다. $A = B = [[1,2],[3,4]]$일 때 원소별 곱은 $[[1,4],[9,16]]$을, 행렬 곱은 $[[7,10],[15,22]]$을 준다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
`torch.einsum`을 사용하여 모양 $(10, 4, 4)$인 텐서의 배치 대각합을 계산하고 모양 $(10,)$인 텐서를 반환하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    A = torch.randn(10, 4, 4)
    traces = torch.einsum('bii->b', A)
    print(traces.shape)  # torch.Size([10])
    # 직접 계산한 값과 대조 확인:
    manual = torch.stack([torch.trace(A[i]) for i in range(10)])
    print(torch.allclose(traces, manual))  # True
    ```

## 정리하며

GPU 활용에서는 장치 일치, 전송 비용, 비동기 실행을 함께 관리한다. 성능 비교는 같은 작업과 측정 범위에서 수행하고 할당 메모리와 예약 메모리를 구분한다.
