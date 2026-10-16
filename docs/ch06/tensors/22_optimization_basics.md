# 최적화 기초 - 옵티마이저와 학습

그래디언트를 계산한 뒤 실제로 파라메터를 움직이는 것은 옵티마이저다. 이 페이지는 SGD·Adam과 러닝레이트 스케줄러를 통해 그래디언트 계산, 갱신, 러닝레이트 조절을 구별한다.

## 1. 코드

```python
"""튜토리얼 22: 최적화 기초 - 최적화기와 학습"""
import torch
import torch.nn as nn
import torch.optim as optim

# ========================================================================
# 메인
# ========================================================================

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

def main():
    header("1. SGD Optimizer")
    params = [torch.tensor([2.0], requires_grad=True)]
    optimizer = optim.SGD(params, lr=0.1)
    print(f"Initial param: {params[0]}")
    for i in range(3):
        optimizer.zero_grad()
        loss = (params[0] - 1.0) ** 2
        loss.backward()
        print(f"Step {i+1}: grad={params[0].grad.item():.4f}, loss={loss.item():.4f}")
        optimizer.step()
        print(f"  Updated param: {params[0].item():.4f}")
    
    header("2. Adam Optimizer")
    model = nn.Linear(10, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("Adam uses adaptive learning rates")
    print(f"Learning rate: {optimizer.defaults['lr']}")
    print(f"Betas: {optimizer.defaults['betas']}")
    
    header("3. Learning Rate Scheduler")
    optimizer = optim.SGD([torch.randn(1, requires_grad=True)], lr=1.0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    for epoch in range(10):
        optimizer.step()
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: LR = {current_lr:.6f}")
    
    header("4. Gradient Clipping")
    model = nn.Linear(5, 1)
    x = torch.randn(10, 5)
    y = torch.randn(10, 1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    
    total_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
    print(f"Gradient norm before clipping: {total_norm_before:.4f}")
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    total_norm_after = sum(p.grad.norm(2).item() ** 2 for p in model.parameters()) ** 0.5
    print(f"Gradient norm after clipping: {total_norm_after:.4f}")
    
    header("5. Complete Training Example")
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    X_train = torch.randn(100, 10)
    y_train = torch.randn(100, 1)
    
    print("Training for 5 epochs...")
    for epoch in range(5):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/5, Loss: {loss.item():.4f}")
    
    header("6. Comparing Optimizers")
    print("""
    흔한 최적화기:
    
    SGD: 단순하고 손봐야 하지만 두루 잘 미친다
    Adam: 맞추어 가며 그대로 써도 잘 듣지만 지나치게 맞춰질 수 있다
    RMSprop: 되도는 뉴럴 넷트웍에 좋다
    AdaGrad: 성긴 그래디언트에 좋다
    
    어림 규칙: Adam으로 비롯하고 마지막 손질에는 SGD으로 갈아탄다
    """)

if __name__ == "__main__":
    main()
```

**출력:**

```

======================================================================
1. SGD Optimizer
======================================================================
Initial param: tensor([2.], requires_grad=True)
Step 1: grad=2.0000, loss=1.0000
  Updated param: 1.8000
Step 2: grad=1.6000, loss=0.6400
  Updated param: 1.6400
Step 3: grad=1.2800, loss=0.4096
  Updated param: 1.5120

======================================================================
2. Adam Optimizer
======================================================================
Adam uses adaptive learning rates
Learning rate: 0.001
Betas: (0.9, 0.999)

======================================================================
3. Learning Rate Scheduler
======================================================================
Epoch 1: LR = 1.000000
Epoch 2: LR = 1.000000
Epoch 3: LR = 0.100000
Epoch 4: LR = 0.100000
Epoch 5: LR = 0.100000
Epoch 6: LR = 0.010000
Epoch 7: LR = 0.010000
Epoch 8: LR = 0.010000
Epoch 9: LR = 0.001000
Epoch 10: LR = 0.001000

======================================================================
4. Gradient Clipping
======================================================================
Gradient norm before clipping: 3.9841
Gradient norm after clipping: 1.0000

======================================================================
5. Complete Training Example
======================================================================
Training for 5 epochs...
Epoch 1/5, Loss: 1.1767
Epoch 2/5, Loss: 1.1320
Epoch 3/5, Loss: 1.0959
Epoch 4/5, Loss: 1.0668
Epoch 5/5, Loss: 1.0411

======================================================================
6. Comparing Optimizers
======================================================================

    흔한 최적화기:
    
    SGD: 단순하고 손봐야 하지만 두루 잘 미친다
    Adam: 맞추어 가며 그대로 써도 잘 듣지만 지나치게 맞춰질 수 있다
    RMSprop: 되도는 뉴럴 넷트웍에 좋다
    AdaGrad: 성긴 그래디언트에 좋다
    
    어림 규칙: Adam으로 비롯하고 마지막 손질에는 SGD으로 갈아탄다
    
```

## 2. 논의

`backward()`는 그래디언트를 계산하고 `optimizer.step()`은 현재 `.grad`로 등록된 파라메터를 갱신한다. `step()`은 그래디언트를 자동으로 지우지 않으므로 누적이 목적이 아니면 다음 역전파 전에 초기화한다.

SGD는 러닝레이트에 따라 이동하고 Adam은 그래디언트의 이동평균과 제곱의 이동평균을 이용한다. 같은 러닝레이트가 두 옵티마이저에서 같은 이동 폭을 뜻하지는 않는다. 스케줄러는 갱신에 사용할 러닝레이트를 바꾸며, 호출 주기가 배치인지 에포크인지와 `step()` 순서를 명확히 해야 한다.

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
제곱합처럼 저장된 중간값이 필요한 로스에서 `retain_graph=True` 없이 같은 그래프에 `.backward()`를 두 번 호출하면 오류가 나는 이유를 설명하라. `retain_graph=True`는 메모리 사용량에 어떤 영향을 주는가?

</div>

??? success "연습문제 2 풀이"
    기본 역전파는 사용을 마친 저장 텐서를 해제한다. 같은 그래프의 두 번째 역전파에 이 값들이 필요하면 `RuntimeError`가 발생한다. 저장 텐서가 필요 없는 단순 연산은 반복 호출이 가능할 수도 있으므로 모든 그래프에서 반드시 오류가 나는 것은 아니다. 첫 호출의 `retain_graph=True`는 필요한 중간 결과를 보존해 재사용을 가능하게 하지만 그만큼 메모리를 더 오래 사용한다. `.grad` 초기화는 이 보존 여부와 별개의 작업이다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
잎 텐서 `w`를 만들고, 반복마다 같은 식으로 로스를 새로 계산하라. 경사를 초기화하지 않고 `.backward()`를 세 번 호출하며 매번 `w.grad`를 출력하라. 관찰된 값을 설명하라.

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

그래디언트를 계산한 뒤 실제로 파라메터를 움직이는 것은 옵티마이저다. 이 페이지는 SGD·Adam과 러닝레이트 스케줄러를 통해 그래디언트 계산, 갱신, 러닝레이트 조절을 구별한다.
