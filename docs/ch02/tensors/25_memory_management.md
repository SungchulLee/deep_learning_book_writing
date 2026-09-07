# 메모리 관리 - 메모리 사용 최적화

이 스크립트는 메모리 관리와 메모리 사용 최적화을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
"""익힘 25: 기억 자리 다루기 - 기억 자리 씀씀이 다듬기"""
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

def main():
    header("1. Understanding Memory Usage")
    x = torch.randn(1000, 1000)
    memory_bytes = x.element_size() * x.nelement()
    memory_mb = memory_bytes / (1024 ** 2)
    print(f"Tensor shape: {x.shape}")
    print(f"Element size: {x.element_size()} bytes")
    print(f"Number of elements: {x.nelement()}")
    print(f"Total memory: {memory_mb:.2f} MB")
    
    header("2. In-place Operations Save Memory")
    x = torch.randn(1000, 1000)
    print(f"Initial memory ID: {id(x)}")
    y = x + 1  # Creates new tensor
    print(f"After x + 1 (new tensor): {id(y)}")
    x.add_(1)  # In-place operation
    print(f"After x.add_(1) (same tensor): {id(x)}")
    print("In-place operations modify tensor without creating a copy!")
    
    header("3. Detaching from Computation Graph")
    x = torch.randn(100, 100, requires_grad=True)
    y = x ** 2
    print(f"y requires_grad: {y.requires_grad}")
    print(f"y has grad_fn: {y.grad_fn is not None}")
    
    z = y.detach()
    print(f"\nAfter detach:")
    print(f"z requires_grad: {z.requires_grad}")
    print(f"z has grad_fn: {z.grad_fn is not None}")
    print("Detach removes from computation graph, saves memory!")
    
    header("4. Using torch.no_grad()")
    model = nn.Linear(1000, 1000)
    x = torch.randn(100, 1000)
    
    print("During inference, use no_grad to save memory:")
    with torch.no_grad():
        output = model(x)
    print(f"Output requires_grad: {output.requires_grad}")
    print("No gradients computed or stored!")
    
    header("5. Gradient Accumulation")
    print("""
    Instead of:
        batch_size = 128  # 기억 자리가 모자랄 수 있다!
        
    기울기 쌓기를 쓴다.
        batch_size = 32
        accumulation_steps = 4  # 실제 묶음 크기 = 128
        
    for i, (x, y) in enumerate(dataloader):
        output = model(x)
        loss = criterion(output, y) / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    """)
    
    header("6. Checkpoint Activations")
    print("""
    아주 깊은 그물에는 기울기 되짚음 저장을 쓴다.
    
    from torch.utils.checkpoint import checkpoint
    
    class DeepModel(nn.Module):
        def forward(self, x):
            # 메모리를 위해 계산을 희생한다
            x = checkpoint(self.layer1, x)
            x = checkpoint(self.layer2, x)
            x = checkpoint(self.layer3, x)
            return x
    
    셈을 30% 더 하는 대신 기억 자리를 10분의 1로 줄인다!
    """)
    
    header("7. Empty Cache (GPU)")
    if torch.cuda.is_available():
        print(f"Allocated: {torch.cuda.memory_allocated()/1e6:.2f} MB")
        x = torch.randn(1000, 1000, device='cuda')
        print(f"After allocation: {torch.cuda.memory_allocated()/1e6:.2f} MB")
        del x
        torch.cuda.empty_cache()
        print(f"After cache clear: {torch.cuda.memory_allocated()/1e6:.2f} MB")
    else:
        print("torch.cuda.empty_cache() - Releases cached GPU memory")
    
    header("8. Memory Profiling")
    print("""
    PyTorch의 기억 자리 살피개를 쓴다.
    
    from torch.profiler import profile, ProfilerActivity
    
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 profile_memory=True) as prof:
        model(input)
    
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))
    """)
    
    header("9. Best Practices Summary")
    print("""
    기억 자리 다듬기 요령:
    
    1. 미룸 때는 torch.no_grad()을 써라
    2. 기울기가 필요 없으면 .detach()을 불러라
    3. 안전할 때는 제자리 셈(_)을 써라
    4. 실제 묶음 크기를 키우려면 기울기를 쌓아라
    5. 섞인 촘촘함 익히기를 써라(익힘 24을 보아라)
    6. 다 쓴 큰 텐서는 지워라: del x
    7. GPU 갈무리를 비워라: torch.cuda.empty_cache()
    8. 깊은 그물에는 기울기 되짚음 저장을 써라
    9. 기억 자리 씀씀이를 살펴 목을 찾아라
    10. 묶음 크기나 모형 크기를 줄이는 것도 생각해 보아라
    """)

if __name__ == "__main__":
    main()```

## 논의

이 코드는 `requires_grad=True`인 텐서에 대한 연산을 자동으로 추적하는 PyTorch의 autograd 체계를 보여준다. 스칼라 손실에 `.backward()`를 호출하면 autograd가 계산 그래프를 역방향으로 훑으며 연쇄 법칙을 적용해 모든 잎 텐서의 경사를 계산한다. 이 구조가 PyTorch의 모든 신경망 학습을 떠받친다.

PyTorch의 `nn.Module`은 신경망 구조를 정의하는 체계적인 방법을 제공한다. 각 모듈이 자신의 매개변수와 하위 모듈을 관리하므로 모델을 살펴보고, 저장하고, 장치 사이에 옮기기가 간편하다.

경사 추적을 제어하는 것은 정확성과 성능 모두에 필수적이다. `torch.no_grad()` 컨텍스트 관리자는 매개변수 갱신이나 추론처럼 계산 그래프에 포함되어서는 안 되는 연산에 대해 autograd를 끈다. `.detach()` 메서드는 저장소는 공유하지만 그래프와는 분리된 텐서를 만들며, 값을 기록하거나 NumPy로 변환할 때 유용하다.

## 연습문제

**연습문제 1.**
함수 $f(x) = x^3 - 2x^2 + x$를 생각하자. PyTorch autograd를 사용하여 $f'(3)$을 계산하라.

??? success "연습문제 1 풀이"
    ```python
    import torch

    x = torch.tensor(3.0, requires_grad=True)
    f = x**3 - 2*x**2 + x
    f.backward()
    print(x.grad)  # f'(x) = 3x^2 - 4x + 1 = 27 - 12 + 1 = 16.0
    ```

---


**연습문제 2.**
`retain_graph=True` 없이 같은 계산 그래프에 `.backward()`를 두 번 호출하면 오류가 나는 이유를 설명하라. `retain_graph=True`는 메모리 사용량에 어떤 영향을 주는가?

??? success "연습문제 2 풀이"
    기본적으로 PyTorch는 메모리를 아끼기 위해 `.backward()` 후에 계산 그래프를 해제한다. `.backward()`를 두 번째로 호출하면 더 이상 존재하지 않는 그래프를 훑으려 하므로 `RuntimeError`가 발생한다. `retain_graph=True`로 두면 그래프가 메모리에 남아 재사용할 수 있지만, 모든 중간 텐서가 할당된 채로 남으므로 메모리 소비가 늘어난다.

---


**연습문제 3.**
잎 텐서 `w`를 만들고 손실을 계산한 뒤, 경사를 초기화하지 않고 `.backward()`를 세 번 호출하며 매번 `w.grad`를 출력하는 코드를 작성하라. 관찰된 값을 설명하라.

??? success "연습문제 3 풀이"
    ```python
    import torch

    w = torch.tensor(2.0, requires_grad=True)
    for i in range(3):
        loss = (w ** 2).sum()
        loss.backward()
        print(f'After backward {i+1}: w.grad = {w.grad}')
    # 출력: 4.0, 8.0, 12.0
    # 경사가 누적된다. 매 backward가 기존 경사에 2*w = 4.0을 더한다.
    ```
