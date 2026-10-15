# 경사 누적

서로 다른 손실에서 계산한 경사가 같은 텐서의 `.grad`에 더해지는 과정을 살펴본다. 경사 초기화와 의도적인 누적이 각각 언제 필요한지 설명한다.

## 1. 코드

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
    
    # 다시 누적.
    # 손실을 다시 계산해야 한다. backward()는 계산 그래프를 쓰고 나면
    # 풀어 버리므로, 같은 loss로 두 번 부르면 "Trying to backward through
    # the graph a second time" 오류가 난다.
    # retain_graph=True로 그래프를 붙들어 둘 수도 있지만, 실제 학습에서는
    # 미니배치마다 손실을 새로 계산하므로 이쪽이 실전에 가깝다
    loss = (x3 ** 2).sum()
    loss.backward()
    print(f"After another backward: x3.grad = {x3.grad}")
    
    print("\nMethod 2: Set to None - Frees memory")
    x3.grad = None
    print(f"After setting to None: x3.grad = {x3.grad}")
    
    # 다음 역전파가 새 경사 텐서를 만든다
    loss = (x3 ** 2).sum()
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
        • 기억 자리가 적을 때 큰 배치를 흉내내기
        • 여러 손실 마디를 아우르기
        • 흩어 익힐 때의 기울기 쌓기
    ✓ 학습 루프에서 기울기를 0으로 만들기를 절대 잊지 마라!
    """)


if __name__ == "__main__":
    main()
```

**출력:**

```
======================================================================
PART 1: Default Behavior - Gradients Accumulate!
======================================================================
Initial x: tensor([2.], requires_grad=True)
Initial x.grad: None

First backward pass: loss1 = x^2
After 1st backward: x.grad = tensor([4.])
  (Expected: 2*x = 2*2.0 = 4.0)

Second backward pass: loss2 = 3*x
After 2nd backward: x.grad = tensor([7.])
  (NOT what we want! 4.0 + 3.0 = 7.0)
  (Gradients accumulated: 1st gradient + 2nd gradient)

======================================================================
PART 2: Correct Behavior - Zero Gradients Between Steps
======================================================================
After 1st backward: x2.grad = tensor([4.])
...
```

## 2. 논의

### 새 경사는 기존 경사에 더해진다

첫 손실 $x^2$을 $x=2$에서 미분하면 4이다. 초기화 없이 다음 손실 $3x$를 역전파하면 `.grad`에는 새 경사 3이 더해져 7이 저장된다. 두 손실은 각각 새로 계산했으므로, 이 누적에는 같은 계산 그래프를 재사용하는 기능이 필요하지 않다.

### 초기화와 누적의 경계

각 손실로 독립적인 파라미터 업데이트를 하려면 업데이트 단위마다 이전 경사를 지운다. `.grad.zero_()`는 기존 경사 텐서를 0으로 채우며, `.grad = None`은 경사 참조를 없애 다음 역전파에서 새 경사가 저장되게 한다. 아직 경사가 없는 텐서에는 `.grad.zero_()`를 바로 호출할 수 없다.

여러 마이크로배치를 하나의 배치처럼 처리할 때는 경사를 지우지 않고 모은 뒤 한 번 업데이트한다. 예제처럼 크기가 같은 마이크로배치의 평균 손실을 사용한다면, 누적된 경사를 마이크로배치 수로 나누어 전체 평균에 맞춘다. 크기가 다르면 각 배치의 표본 수에 맞춰 가중해야 한다.

여러 손실 항을 합쳐 한 번 역전파하는 방법과 각각 역전파해 경사를 더하는 방법도 비교한다. 별도 역전파에서 `retain_graph=True`가 필요한지는 두 손실이 역전파용 중간 결과를 공유하는지에 달려 있다. 예제의 서로 독립적인 두 경로에는 이를 반드시 지정할 필요가 없다.

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
서로 새로 계산한 두 손실을 같은 잎 텐서로 역전파하면 경사 누적을 위해 `retain_graph=True`가 필요한가?

</div>

??? success "연습문제 2 풀이"
    필요하지 않다. `.grad`의 누적과 그래프 보존은 별개다. 각 손실을 새로 계산했다면 각각의 그래프로 역전파할 수 있다. 다만 두 손실이 역전파에 필요한 중간 결과를 공유한다면 별도의 보존 여부를 검토해야 한다.

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

경사 누적은 기본 동작이다. 업데이트 단위와 손실의 평균 기준을 먼저 정한 뒤, 그 경계에 맞춰 경사 초기화와 스케일 조정을 배치한다.
