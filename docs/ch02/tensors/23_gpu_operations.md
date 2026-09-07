# GPU 연산 - CUDA와 장치 관리

이 스크립트는 GPU 연산, 즉 CUDA와 장치 관리을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
"""Tutorial 23: GPU Operations - CUDA and device management"""
import torch

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
    GPU Best Practices:
    
    1. Create tensors directly on GPU when possible
    2. Minimize CPU ↔ GPU data transfers
    3. Use larger batch sizes to saturate GPU
    4. Clear cache when running out of memory
    5. Use torch.cuda.synchronize() for accurate timing
    6. Use mixed precision (see tutorial 24)
    7. Monitor memory with torch.cuda.memory_allocated()
    """)

if __name__ == "__main__":
    main()```

## 논의

PyTorch의 `nn.Module`은 신경망 구조를 정의하는 체계적인 방법을 제공한다. 각 모듈이 자신의 매개변수와 하위 모듈을 관리하므로 모델을 살펴보고, 저장하고, 장치 사이에 옮기기가 간편하다.

행렬 연산은 신경망 계산의 핵심을 이룬다. `@` 연산자와 `torch.matmul()`은 배치 차원에 대한 자동 브로드캐스팅과 함께 행렬 곱을 처리하고, `torch.mm()`이나 `torch.bmm()` 같은 특화된 함수는 명료함과 성능을 위해 텐서의 계수를 특정 값으로 강제한다.

GPU 가속은 텐서 연산, 특히 신경망 계산을 지배하는 행렬 곱에 대해 몇 자릿수의 속도 향상을 제공한다. `.to(device)`로 텐서와 모델을 GPU로 옮기는 것은 간단하지만, 성능을 유지하려면 CPU-GPU 사이의 데이터 전송을 최소화하는 것이 결정적이다.

### 연산이 실제로 일어나는 곳

왜 GPU를 쓰는지 이해하려면 데이터가 지나는 길을 먼저 보아야 한다.

| 층 | 성격 |
|---|---|
| CPU (코어) | 실제로 셈하는 곳 |
| RAM | 주기억장치 |
| ROM | 읽기 전용 |
| SSD | 데이터 저장소 |

데이터는 SSD에 있다가 RAM으로 올라오고, 거기서 CPU가 가져다 쓴다. GPU, CPU, RAM, SSD는 모두 **마더보드**에 꽂히며, 이들 사이를 오가는 길이 **버스(bus)** 이다. 버스는 유닛끼리 통신하는 통로이므로, 버스가 느리면 코어가 아무리 많아도 앞에서 막힌다. 위에서 말한 "데이터 전송 최소화"가 바로 이 지점이다.

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

여기서 중요한 것은 값이 아니라 **결과 원소들이 서로를 기다릴 까닭이 없다**는 점이다. 각 원소는 자기 행과 열만 있으면 되므로 네 개를 동시에 셈할 수 있다. GPU를 쓰는 까닭은 이것이 전부다.

- **NumPy**는 CPU 코어를 **하나만** 쓴다.
- **GPU**는 코어를 **모두** 쓴다.

딥러닝에서 CPU를 쓰면 행렬 크기가 커지면서 문제가 생긴다. 그래서 슈퍼컴퓨터가 병렬 처리하듯 코어를 병렬로 쓰자는 생각이 나왔고, 그것을 위해 **CUDA**가 만들어졌다.

```
CPU  --PyTorch-->  CUDA  -->  GPU
```

### 배치 크기를 $2^n$으로 잡는 까닭

GPU는 코어마다 한 장씩 받아 갈 수 있다. 코어가 2의 거듭제곱 꼴로 묶여 있으므로 넘겨 주는 장 수도 $2^n$으로 맞추면 코어가 남거나 모자라지 않는다. 학습 코드에서 배치 크기로 512, 1024, 2048 같은 값이 관행처럼 쓰이는 이유가 여기에 있다.

!!! tip "가장 중요한 습관"
    PyTorch를 쓸 때에는 **이 코드가 CPU에서 도는지 GPU에서 도는지 명확히 구분하라.** 기본값이 CPU이므로 아무 생각 없이 짜면 GPU를 두고도 쓰지 않게 된다. 로컬에 GPU가 없다면 Google Colab에서 런타임을 GPU로 바꿔 같은 코드가 얼마나 달라지는지 확인해 보는 것이 가장 빠른 감각 훈련이다.

## 연습문제

**연습문제 1.**
$(2, 3)$ 행렬과 $(3, 4)$ 행렬의 곱을 `@`, `torch.mm()`, `torch.matmul()` 세 가지 방법으로 계산하라. 결과가 동일함을 확인하라.

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


**연습문제 2.**
원소별 곱(`*`)과 행렬 곱(`@`)의 차이를 설명하라. 같은 입력에 대해 서로 다른 결과를 내는 예를 들라.

??? success "연습문제 2 풀이"
    원소별 곱은 대응하는 원소끼리 곱한다. 같은 모양의 행렬 $A, B$에 대해 $(A * B)_{ij} = A_{ij} B_{ij}$이다. 행렬 곱은 안쪽 차원에 대해 더한다. $(A @ B)_{ij} = \sum_k A_{ik} B_{kj}$이다. $A = B = [[1,2],[3,4]]$일 때 원소별 곱은 $[[1,4],[9,16]]$을, 행렬 곱은 $[[7,10],[15,22]]$을 준다.

---


**연습문제 3.**
`torch.einsum`을 사용하여 모양 $(10, 4, 4)$인 텐서의 배치 대각합을 계산하고 모양 $(10,)$인 텐서를 반환하라.

??? success "연습문제 3 풀이"
    ```python
    A = torch.randn(10, 4, 4)
    traces = torch.einsum('bii->b', A)
    print(traces.shape)  # torch.Size([10])
    # 직접 계산한 값과 대조 확인:
    manual = torch.stack([torch.trace(A[i]) for i in range(10)])
    print(torch.allclose(traces, manual))  # True
    ```
