# 메모리 관리 - 메모리 사용 최적화

학습 메모리는 파라미터뿐 아니라 경사, 옵티마이저 상태, 역전파용 활성값에도 사용된다. 뷰와 사본, 추적 제어, 활성값 체크포인팅이 각각 어떤 비용을 바꾸는지 살펴본다.

## 1. 코드

```python
"""튜토리얼 25: 기억 자리 다루기 - 기억 자리 씀씀이 다듬기"""
import torch
import torch.nn as nn
# 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
# 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
torch.manual_seed(0)

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
    # data_ptr()가 돌려주는 번지 값은 돌릴 때마다, 기계마다 다르다. 그래서 번지
    # 자체는 싣지 않고, 뜻이 있는 것 — 같은 저장소를 쓰는가 — 만 견주어 찍는다
    x_storage = x.data_ptr()
    first = x[0, 0].item()

    y = x + 1  # Creates new tensor
    print(f"x + 1 -> same object as x?    {y is x}")
    print(f"x + 1 -> same storage as x?   {y.data_ptr() == x_storage}")
    print(f"x left unchanged by x + 1?    {x[0, 0].item() == first}")

    x.add_(1)  # In-place operation
    print(f"x.add_(1) -> same storage?    {x.data_ptr() == x_storage}")
    print(f"x changed in place?           {abs(x[0, 0].item() - (first + 1)) < 1e-6}")
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
        accumulation_steps = 4  # 실제 배치 크기 = 128
        
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
    아주 깊은 망에는 기울기 되짚음 저장을 쓴다.
    
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
    
    1. 추론 때는 torch.no_grad()을 써라
    2. 기울기가 필요 없으면 .detach()을 불러라
    3. 안전할 때는 제자리 셈(_)을 써라
    4. 실제 배치 크기를 키우려면 기울기를 쌓아라
    5. 섞인 촘촘함 익히기를 써라(튜토리얼 24을 보아라)
    6. 다 쓴 큰 텐서는 지워라: del x
    7. GPU 저장을 비워라: torch.cuda.empty_cache()
    8. 깊은 망에는 기울기 되짚음 저장을 써라
    9. 기억 자리 씀씀이를 살펴 목을 찾아라
    10. 배치 크기나 모델 크기를 줄이는 것도 생각해 보아라
    """)

if __name__ == "__main__":
    main()
```

??? note "전체 출력 (114줄)"

    ```

    ======================================================================
    1. Understanding Memory Usage
    ======================================================================
    Tensor shape: torch.Size([1000, 1000])
    Element size: 4 bytes
    Number of elements: 1000000
    Total memory: 3.81 MB

    ======================================================================
    2. In-place Operations Save Memory
    ======================================================================
    x + 1 -> same object as x?    False
    x + 1 -> same storage as x?   False
    x left unchanged by x + 1?    True
    x.add_(1) -> same storage?    True
    x changed in place?           True
    In-place operations modify tensor without creating a copy!

    ======================================================================
    3. Detaching from Computation Graph
    ======================================================================
    y requires_grad: True
    y has grad_fn: True

    After detach:
    z requires_grad: False
    z has grad_fn: False
    Detach removes from computation graph, saves memory!

    ======================================================================
    4. Using torch.no_grad()
    ======================================================================
    During inference, use no_grad to save memory:
    Output requires_grad: False
    No gradients computed or stored!

    ======================================================================
    5. Gradient Accumulation
    ======================================================================

        Instead of:
            batch_size = 128  # 기억 자리가 모자랄 수 있다!

        기울기 쌓기를 쓴다.
            batch_size = 32
            accumulation_steps = 4  # 실제 배치 크기 = 128

        for i, (x, y) in enumerate(dataloader):
            output = model(x)
            loss = criterion(output, y) / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()


    ======================================================================
    6. Checkpoint Activations
    ======================================================================

        아주 깊은 망에는 기울기 되짚음 저장을 쓴다.

        from torch.utils.checkpoint import checkpoint

        class DeepModel(nn.Module):
            def forward(self, x):
                # 메모리를 위해 계산을 희생한다
                x = checkpoint(self.layer1, x)
                x = checkpoint(self.layer2, x)
                x = checkpoint(self.layer3, x)
                return x

        셈을 30% 더 하는 대신 기억 자리를 10분의 1로 줄인다!


    ======================================================================
    7. Empty Cache (GPU)
    ======================================================================
    torch.cuda.empty_cache() - Releases cached GPU memory

    ======================================================================
    8. Memory Profiling
    ======================================================================

        PyTorch의 기억 자리 살피개를 쓴다.

        from torch.profiler import profile, ProfilerActivity

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     profile_memory=True) as prof:
            model(input)

        print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))


    ======================================================================
    9. Best Practices Summary
    ======================================================================

        기억 자리 다듬기 요령:

        1. 추론 때는 torch.no_grad()을 써라
        2. 기울기가 필요 없으면 .detach()을 불러라
        3. 안전할 때는 제자리 셈(_)을 써라
        4. 실제 배치 크기를 키우려면 기울기를 쌓아라
        5. 섞인 촘촘함 익히기를 써라(튜토리얼 24을 보아라)
        6. 다 쓴 큰 텐서는 지워라: del x
        7. GPU 저장을 비워라: torch.cuda.empty_cache()
        8. 깊은 망에는 기울기 되짚음 저장을 써라
        9. 기억 자리 씀씀이를 살펴 목을 찾아라
        10. 배치 크기나 모델 크기를 줄이는 것도 생각해 보아라

    ```


## 2. 논의

### 텐서 크기와 실제 사용량

`element_size() * nelement()`는 텐서 원소가 차지하는 논리적 바이트 수다. 실제 프로세스 메모리에는 저장 공간 공유, 할당기 관리와 다른 객체의 비용도 포함된다. 뷰를 여러 개 만들었다고 이 논리적 크기들을 모두 더해 실제 사용량으로 볼 수는 없다.

예제는 `x + 1`과 `x.add_(1)`의 주소를 비교한다. 전자는 새 결과를 만들고, 후자는 기존 값을 수정한다. 다만 제자리 연산은 원래 값이 필요한 역전파와 충돌할 수 있으며, 항상 큰 메모리 절약을 주는 최적화는 아니다.

### 그래프와 저장 공간의 수명

`detach()`는 분리된 텐서를 반환하지만 원래 텐서와 저장 공간을 공유한다. 원래 결과나 그래프를 다른 곳에서 계속 참조하면 그것만으로 그래프 메모리가 해제되지는 않는다. 기록용 손실은 `loss.item()`처럼 그래프와 독립적인 값으로 저장하고, 추론은 `no_grad()`로 불필요한 기록을 줄인다.

### 재계산과 CUDA 캐시

활성값 체크포인팅은 일부 중간값을 저장하는 대신 역전파 때 다시 계산해 메모리와 계산량을 교환한다. 코드에 적힌 절약 비율이나 추가 시간은 일반적인 보장값이 아니며 모델별로 측정해야 한다. 현재 API에서는 `use_reentrant` 선택에 따라 동작 조건도 달라지므로 사용 버전의 규칙을 확인한다.

CUDA의 `memory_allocated()`는 살아 있는 텐서의 할당량, `memory_reserved()`는 캐시 할당기가 확보한 메모리를 나타낸다. `empty_cache()`는 사용하지 않는 캐시를 반환하며 참조 중인 텐서를 지우지 않는다. 따라서 호출 직후 할당량이 줄어들지 않아도 이상한 것은 아니다. 프로파일러는 비용이 큰 연산을 찾는 데 사용하되 CPU와 CUDA 실행 환경의 차이를 고려한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
함수 $f(x) = x^3 - 2x^2 + x$를 생각하자. PyTorch autograd를 사용하여 $f'(3)$을 계산하라.

</div>

??? success "연습문제 1 풀이"
    ```python
    import torch

    x = torch.tensor(3.0, requires_grad=True)
    f = x**3 - 2*x**2 + x
    f.backward()
    print(x.grad)  # f'(x) = 3x^2 - 4x + 1 = 27 - 12 + 1 = 16.0
    ```

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
제곱합처럼 저장된 중간값이 필요한 손실에서 `retain_graph=True` 없이 같은 그래프에 `.backward()`를 두 번 호출하면 오류가 나는 이유를 설명하라. `retain_graph=True`는 메모리 사용량에 어떤 영향을 주는가?

</div>

??? success "연습문제 2 풀이"
    기본 역전파는 사용을 마친 저장 텐서를 해제한다. 같은 그래프의 두 번째 역전파에 이 값들이 필요하면 `RuntimeError`가 발생한다. 저장 텐서가 필요 없는 단순 연산은 반복 호출이 가능할 수도 있으므로 모든 그래프에서 반드시 오류가 나는 것은 아니다. 첫 호출의 `retain_graph=True`는 필요한 중간 결과를 보존해 재사용을 가능하게 하지만 그만큼 메모리를 더 오래 사용한다. `.grad` 초기화는 이 보존 여부와 별개의 작업이다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
잎 텐서 `w`를 만들고, 반복마다 같은 식으로 손실을 새로 계산하라. 경사를 초기화하지 않고 `.backward()`를 세 번 호출하며 매번 `w.grad`를 출력하라. 관찰된 값을 설명하라.

</div>

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

## 정리하며

메모리를 줄이려면 살아 있는 텐서와 그래프 참조부터 확인한다. 복사 감소, 추적 제어, 활성값 재계산과 캐시 정리는 서로 다른 역할을 한다.
