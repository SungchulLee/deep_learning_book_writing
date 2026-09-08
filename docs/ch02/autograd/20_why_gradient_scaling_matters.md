# 경사 크기 조정이 중요한 이유

이 스크립트는 경사 크기 조정이 중요한 이유을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 1. 코드

```python
"""기울기 척도 잡기가 왜 종요로운가."""
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

def main():

    torch.manual_seed(3)

    modelA = nn.Linear(3, 1, bias=False)
    modelB = nn.Linear(3, 1, bias=False)

    # 두 모델이 같은 지점에서 출발하도록 동일한 가중치를 복사한다
    with torch.no_grad():
        modelB.weight.copy_(modelA.weight)

    X = torch.randn(4, 3)
    y = torch.randn(4, 1)

    # -- 기준선: 전체 배치 평균 손실 → 참조 경사
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

    # (i) 잘못된 방법: 마이크로배치마다 **평균** 손실을 역전파하고 그냥 누적한다.
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

    # (ii) 올바른 방법: 각 마이크로배치의 평균 손실에 1 / (마이크로배치 개수)를 곱한다.
    # 마이크로배치가 K개일 때 마이크로배치마다 (loss_mean / K)를 역전파한다.
    # 그러면 누적된 경사가 전체 배치 평균 손실의 경사와 일치한다.
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

## 2. 논의

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

## 정리하며

**다룬 것** — 경사 크기 조정이 중요한 이유

이 코드는 `requires_grad=True`인 텐서에 대한 연산을 자동으로 추적하는 PyTorch의 autograd 체계를 보여준다.

앞의 연습문제 3개로 직접 확인할 수 있다.
