# 경사 누적

이 스크립트는 경사 누적을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
#!/usr/bin/env python3
"""
==============================================================================
첫걸음 튜토리얼 02: 기울기 쌓임
==============================================================================

학습 목표:
-------------------
1. PyTorch에서 기울기가 기본으로 쌓인다는 것을 이해한다
2. 학습 걸음 사이에 기울기를 0으로 만들어야 하는 까닭을 배운다
3. 기울기가 쌓이는 모습을 손에 잡히는 보기로 본다
4. 기울기 쌓임이 언제 쓸모 있고 언제 골칫거리인지 이해한다

핵심 개념:
------------
- 기울기 쌓임: .backward()을 부를 때마다 기울기가 더해진다
- 기울기 0으로 만들기: .zero_()이나 .grad = None으로 드러내 놓고 지워야 한다
- 쓰임새: 큰 배치 흉내내기, 여러 손실 마디

흔한 함정:
--------------
기울기를 0으로 만들기를 잊으면 잘못 고치게 된다!

==============================================================================
"""

import torch

# ========================================================================
# 메인
# ========================================================================


def main():
    """
    여러 번 뒤로 걸으며 기울기가 어떻게 쌓이는지,
    그리고 옳게 익히려면 왜 기울기를 0으로 만들어야 하는지 보인다.
    """
    
    torch.manual_seed(0)
    
    print("="*70)
    print("PART 1: Default Behavior - Gradients Accumulate!")
    print("="*70)
    
    # 간단한 매개변수를 만든다
    x = torch.tensor([2.0], requires_grad=True)
    
    print(f"Initial x: {x}")
    print(f"Initial x.grad: {x.grad}")
    print()
    
    # 첫 번째 역전파
    print("First backward pass: loss1 = x^2")
    loss1 = x ** 2  # loss1 = 4.0, d(loss1)/dx = 2x = 4.0
    loss1.backward()
    print(f"After 1st backward: x.grad = {x.grad}")
    print(f"  (Expected: 2*x = 2*2.0 = 4.0)")
    print()
    
    # 경사를 초기화하지 않은 두 번째 역전파
    print("Second backward pass: loss2 = 3*x")
    loss2 = 3 * x  # d(loss2)/dx = 3.0
    loss2.backward()
    print(f"After 2nd backward: x.grad = {x.grad}")
    print(f"  (NOT what we want! 4.0 + 3.0 = 7.0)")
    print(f"  (Gradients accumulated: 1st gradient + 2nd gradient)")
    print()
    
    print("="*70)
    print("PART 2: Correct Behavior - Zero Gradients Between Steps")
    print("="*70)
    
    # 초기화
    x2 = torch.tensor([2.0], requires_grad=True)
    
    # 첫 단계
    loss1 = x2 ** 2
    loss1.backward()
    print(f"After 1st backward: x2.grad = {x2.grad}")
    
    # 다음 단계 전에 경사를 0으로
    x2.grad.zero_()
    print(f"After zeroing: x2.grad = {x2.grad}")
    
    # 두 번째 단계
    loss2 = 3 * x2
    loss2.backward()
    print(f"After 2nd backward: x2.grad = {x2.grad}")
    print(f"  (Correct! Only the gradient from loss2 = 3.0)")
    print()
    
    print("="*70)
    print("PART 3: Two Ways to Zero Gradients")
    print("="*70)
    
    x3 = torch.tensor([5.0], requires_grad=True)
    loss = (x3 ** 2).sum()
    loss.backward()
    print(f"Before zeroing: x3.grad = {x3.grad}")
    
    print("\nMethod 1: .zero_() - Sets values to 0")
    x3.grad.zero_()
    print(f"After .zero_(): x3.grad = {x3.grad}")
    print(f"  Type: {type(x3.grad)}")
    
    # 다시 누적
    loss.backward()
    print(f"After another backward: x3.grad = {x3.grad}")
    
    print("\nMethod 2: Set to None - Frees memory")
    x3.grad = None
    print(f"After setting to None: x3.grad = {x3.grad}")
    
    # 다음 역전파가 새 경사 텐서를 만든다
    loss.backward()
    print(f"After backward: x3.grad = {x3.grad}")
    print()
    
    print("="*70)
    print("PART 4: Intentional Gradient Accumulation (Valid Use Case)")
    print("="*70)
    print("Use case: Simulating large batch when GPU memory is limited")
    print()
    
    torch.manual_seed(42)
    w = torch.randn(1, requires_grad=True)
    lr = 0.01
    
    # 큰 배치를 작은 덩어리로 나누어 처리하는 상황을 흉내 낸다
    micro_batches = 4
    print(f"Simulating batch_size={micro_batches * 10} by accumulating {micro_batches} micro-batches")
    print()
    
    # 시작할 때 경사를 0으로
    if w.grad is not None:
        w.grad.zero_()
    
    # 여러 마이크로배치의 경사를 누적한다
    for i in range(micro_batches):
        # 각 마이크로배치가 자기 손실을 계산한다
        x_batch = torch.randn(10, 1)
        y_batch = 2 * x_batch + 1 + 0.1 * torch.randn(10, 1)
        
        pred = x_batch * w
        loss = ((pred - y_batch) ** 2).mean()
        
        # 초기화 없이 역전파 - 경사가 누적된다
        loss.backward()
        
        print(f"  Micro-batch {i+1}: loss={loss.item():.4f}, w.grad={w.grad.item():.4f}")
    
    # 누적된 경사를 평균 낸다
    w.grad /= micro_batches
    print(f"\nAfter averaging: w.grad = {w.grad.item():.4f}")
    
    # 이제 매개변수를 갱신한다
    with torch.no_grad():
        w -= lr * w.grad
    
    print(f"Updated w: {w.item():.4f}")
    print()
    
    print("="*70)
    print("PART 5: Multiple Loss Terms")
    print("="*70)
    
    x4 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    
    # 합치고 싶은 여러 손실 항
    loss_mse = ((x4 - 2) ** 2).mean()  # MSE loss
    loss_l1 = x4.abs().mean()           # L1 regularization
    
    # 방법 1: 역전파 전에 합친다(권장)
    total_loss = loss_mse + 0.1 * loss_l1
    total_loss.backward()
    print(f"Method 1 - Combined loss backward:")
    print(f"  x4.grad = {x4.grad}")
    
    # 방법 2: 별도의 역전파로부터 누적한다(유효하지만 덜 흔하다)
    x5 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    loss_mse = ((x5 - 2) ** 2).mean()
    loss_l1 = x5.abs().mean()
    
    # 먼저 0으로
    if x5.grad is not None:
        x5.grad.zero_()
    
    # 첫 번째 항에 대한 역전파
    loss_mse.backward(retain_graph=True)  # Need retain_graph for second backward
    grad_after_mse = x5.grad.clone()
    
    # 두 번째 항에 대한 역전파(누적된다)
    scaled_l1 = 0.1 * loss_l1
    scaled_l1.backward()
    
    print(f"\nMethod 2 - Separate backwards with accumulation:")
    print(f"  After MSE: {grad_after_mse}")
    print(f"  After both: {x5.grad}")
    print(f"  Match Method 1? {torch.allclose(x4.grad, x5.grad)}")
    print()
    
    print("="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("""
    ✓ 기울기는 기본으로 쌓인다. 이는 버릇이지 벌레가 아니다!
    ✓ 학습 걸음 사이에는 늘 기울기를 0으로 만들어라: .zero_()이나 = None
    ✓ .zero_()은 텐서를 남기고 = None은 기억 자리를 놓아준다(조금 더 빠르다)
    ✓ 일부러 쌓는 것은 다음에 쓸모 있다.
        • 기억 자리가 적을 때 큰 배치을 흉내내기
        • 여러 손실 마디를 아우르기
        • 흩어 익힐 때의 기울기 쌓기
    ✓ 학습 루프에서 기울기를 0으로 만들기를 절대 잊지 마라!
    """)


if __name__ == "__main__":
    main()```

## 논의

이 코드는 `requires_grad=True`인 텐서에 대한 연산을 자동으로 추적하는 PyTorch의 autograd 체계를 보여준다. 스칼라 손실에 `.backward()`를 호출하면 autograd가 계산 그래프를 역방향으로 훑으며 연쇄 법칙을 적용해 모든 잎 텐서의 경사를 계산한다. 이 구조가 PyTorch의 모든 신경망 학습을 떠받친다.

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
