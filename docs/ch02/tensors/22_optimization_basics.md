# 최적화 기초 - 최적화기와 학습

이 스크립트는 최적화의 기초, 즉 최적화기와 학습을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

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
    RMSprop: 되도는 신경망에 좋다
    AdaGrad: 성긴 기울기에 좋다
    
    어림 규칙: Adam으로 비롯하고 마지막 손질에는 SGD으로 갈아탄다
    """)

if __name__ == "__main__":
    main()```

## 논의

이 코드는 `requires_grad=True`인 텐서에 대한 연산을 자동으로 추적하는 PyTorch의 autograd 체계를 보여준다. 스칼라 손실에 `.backward()`를 호출하면 autograd가 계산 그래프를 역방향으로 훑으며 연쇄 법칙을 적용해 모든 잎 텐서의 경사를 계산한다. 이 구조가 PyTorch의 모든 신경망 학습을 떠받친다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사 초기화, 역전파, 매개변수 갱신이다. 각 구성 요소가 결정적인 역할을 한다. 최적화기는 갱신 규칙(SGD, Adam 등)을 캡슐화하고 학습률과 모멘텀 상태를 내부에서 관리한다.

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
