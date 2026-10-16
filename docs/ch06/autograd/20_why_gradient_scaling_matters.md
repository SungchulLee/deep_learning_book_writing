# 경사 스케일 조정이 필요한 이유

동일한 초기 웨이트에서 전체 배치의 경사와 두 마이크로배치의 누적 경사를 비교한다. 평균 로스를 그대로 더했을 때 생기는 배율 차이를 확인한다.

## 1. 코드

```python
"""그래디언트 척도 잡기가 왜 종요로운가."""
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

def main():

    torch.manual_seed(3)

    modelA = nn.Linear(3, 1, bias=False)
    modelB = nn.Linear(3, 1, bias=False)

    # 두 모델이 같은 지점에서 출발하도록 동일한 웨이트를 복사한다
    with torch.no_grad():
        modelB.weight.copy_(modelA.weight)

    X = torch.randn(4, 3)
    y = torch.randn(4, 1)

    # -- 기준선: 전체 배치 평균 로스 → 참조 경사
    # 전체 배치에 대해 loss = MSE(X→y, reduction="mean")의 경사를 계산한다.
    # 마이크로배치 누적을 이 "정답"과 비교할 것이다.
    pred_full = modelA(X)
    loss_full = nn.functional.mse_loss(pred_full, y, reduction="mean")
    loss_full.backward()

    # param.grad에 관하여:
    # - modelA.weight.grad는 autograd가 할당한 *경사 버퍼 텐서* 이다.
    # - 순전파 그래프의 일부가 아니다(requires_grad=False, grad_fn=None).
    # - 따라서 추적을 피하기 위해 .detach()가 필요하지는 않다. 이미 추적되지 않는다.
    # - 다만 이후에 경사를 바꾸거나 지우기 전에 안정적인 스냅숏을 남기려면 복제가 유용하다.
    #   (예: opt.zero_grad(set_to_none=True)는 grad를 None으로 만들고, 이후 backward()가 덮어쓴다.)
    #
    # 선택지:
    #   grad_full = modelA.weight.grad.clone()                # 충분하다(어차피 추적되지 않는다)
    #   grad_full = modelA.weight.grad.detach().clone()       # 이것도 괜찮다. 추가 .detach()는 불필요하다(방어적 코딩)
    #   grad_full = modelA.weight.grad.detach()               # 저장소를 공유한다. 이후 grad가 바뀌면 위험하다
    grad_full = modelA.weight.grad.detach().clone()

    # -- modelB에 대한 마이크로배치 누적
    modelB.zero_grad(set_to_none=True)
    mb1 = (X[:2], y[:2])
    mb2 = (X[2:], y[2:])

    # (i) 잘못된 방법: 마이크로배치마다 **평균** 로스를 역전파하고 그냥 누적한다.
    # 직관: 각 마이크로배치의 평균은 이미 자기 크기로 나뉘어 있다.
    # 이런 경사 둘을 누적하면 전체 배치 평균 경사의 약 두 배가 된다(절반씩 같을 때).
    # 그래서 결과가 너무 커진다(여기서는 약 2배).
    pred1 = modelB(mb1[0])
    loss1 = nn.functional.mse_loss(pred1, mb1[1], reduction="mean")
    loss1.backward()  # grads accumulate → already too big
    pred2 = modelB(mb2[0])
    loss2 = nn.functional.mse_loss(pred2, mb2[1], reduction="mean")
    loss2.backward()  # accumulates again → now ≈ 2× full-batch mean grad
    wrong_grad = modelB.weight.grad.detach().clone()  # snapshot for fair comparison

    # (ii) 올바른 방법: 각 마이크로배치의 평균 로스에 1 / (마이크로배치 개수)를 곱한다.
    # 마이크로배치가 K개일 때 마이크로배치마다 (loss_mean / K)를 역전파한다.
    # 그러면 누적된 경사가 전체 배치 평균 로스의 경사와 일치한다.
    modelB.zero_grad(set_to_none=True)
    pred1 = modelB(mb1[0])
    (nn.functional.mse_loss(pred1, mb1[1], reduction="mean") / 2).backward()
    pred2 = modelB(mb2[0])
    (nn.functional.mse_loss(pred2, mb2[1], reduction="mean") / 2).backward()
    right_grad = modelB.weight.grad.detach().clone()  # safe snapshot again

    print("Full-batch grad:\n", grad_full)
    print("Accumulated grad (WRONG, unscaled):\n", wrong_grad)
    print("Accumulated grad (RIGHT, scaled):\n", right_grad)
    print("max |full - right|:", (grad_full - right_grad).abs().max().item())

if __name__ == "__main__":
    main()
```

**출력:**

```
Full-batch grad:
 tensor([[-0.7326, -2.7886,  0.2189]])
Accumulated grad (WRONG, unscaled):
 tensor([[-1.4653, -5.5773,  0.4379]])
Accumulated grad (RIGHT, scaled):
 tensor([[-0.7326, -2.7886,  0.2189]])
max |full - right|: 5.960464477539063e-08
```

## 2. 논의

### 비교 조건 고정하기

두 선형 모델의 웨이트를 같게 복사하고, 같은 데이터에서 업데이트 없이 경사만 계산한다. 그래야 차이가 초기값이나 파라미터 변화가 아니라 로스의 스케일에서 비롯되는지 판단할 수 있다.

### 두 평균의 합은 전체 평균의 두 배

크기가 같은 두 마이크로배치의 평균 로스를 $L_1,L_2$라 하면 전체 로스는 $(L_1+L_2)/2$이다. 두 로스를 나누지 않고 역전파하면 $\nabla L_1+\nabla L_2$가 쌓여 전체 경사의 두 배가 된다. 각 로스를 2로 나누면 전체 배치의 평균 경사와 맞는다.

단순 SGD에서는 이 배율이 러닝레이트를 두 배로 한 효과를 내지만, Adam이나 경사 클리핑까지 포함한 모든 업데이트가 같은 방식으로 변하는 것은 아니다. 여기서 말하는 스케일 조정은 배치 평균을 맞추는 작업이며, 혼합 정밀도에서 언더플로를 줄이기 위한 `GradScaler`와 목적이 다르다.

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
크기가 같은 두 마이크로배치의 평균 로스를 나누지 않고 누적하면 전체 배치 경사와 어떤 차이가 나는가?

</div>

??? success "연습문제 2 풀이"
    전체 평균 로스는 두 배치 평균의 절반이므로 나누지 않고 더한 경사는 전체 경사의 두 배다. 각 로스를 2로 나누면 같은 스케일이 된다. 배치 크기가 다르면 실제 평균 대상 원소 수로 가중해야 한다.

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

누적 경사를 비교하려면 초기 파라미터와 로스의 평균 기준을 맞춘다. 마이크로배치 평균을 그대로 더하면 유효 러닝레이트가 의도와 달라질 수 있다.
