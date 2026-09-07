# Linear 가중치 관례

이 스크립트는 `nn.Linear`의 가중치 관례을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
#!/usr/bin/env python3
"""
PyTorch nn.Linear의 무게 관례와 같은 꼴들.

고갱이:
- nn.Linear(in_features, out_features)이 담는 것:
    weight.shape == (out_features, in_features)
    bias.shape   == (out_features,)  (bias=True이면)
- 앞으로 걸음은 다음과 같다.
    out = F.linear(x, weight, bias) == x @ weight.T + bias
  여기서 x.shape == (batch, in_features), out.shape == (batch, out_features)

We verify:
- 치우침이 있을 때와 없을 때의 꼴과 같음
- weight.T은 보기일 뿐(베끼지 않는다)이며 필요하면 .contiguous()을 쓴다
- 자동 미분 기울기의 꼴
- 무게의 줄을 잘라 보며 "한 줄 = 내놓음 뉴런 하나"라는 느낌 잡기
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================

def header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def main():
    torch.manual_seed(0)

    B, in_features, out_features = 4, 5, 3
    x = torch.randn(B, in_features)

    # -------------------------------------------------------------------------
    header("nn.Linear weight/bias shapes and equivalence to x @ W.T + b")
    lin = nn.Linear(in_features, out_features, bias=True)

    W = lin.weight  # (out_features, in_features)
    b = lin.bias    # (out_features,)
    print("W.shape:", W.shape, "| b.shape:", b.shape)

    # F.linear와 matmul의 동등성
    out_F = F.linear(x, W, b)        # preferred functional form
    out_mm = x @ W.t() + b           # explicit matmul
    print("Allclose(F.linear, x @ W.T + b):", torch.allclose(out_F, out_mm, atol=1e-6))

    # -------------------------------------------------------------------------
    header("Without bias: same equivalence")
    lin_nobias = nn.Linear(in_features, out_features, bias=False)
    W2 = lin_nobias.weight
    out_F2 = F.linear(x, W2, None)
    out_mm2 = x @ W2.t()
    print("W2.shape:", W2.shape, "| bias=None")
    print("Allclose(no-bias):", torch.allclose(out_F2, out_mm2, atol=1e-6))

    # -------------------------------------------------------------------------
    header("Transpose view & contiguity")
    WT = W.t()  # view with different strides (no data copy)
    print("W.is_contiguous:", W.is_contiguous(), "| W.t().is_contiguous:", WT.is_contiguous())
    WTc = WT.contiguous()
    print("After .contiguous(), WTc.is_contiguous:", WTc.is_contiguous())

    # -------------------------------------------------------------------------
    header("Autograd shapes: grads wrt W and b")
    # 역전파할 수 있도록 간단한 스칼라 손실
    x_req = x.clone().requires_grad_(True)  # data as leaf (normally requires_grad=False)
    out = F.linear(x_req, W, b)
    loss = out.sum()
    # 매개변수의 경사 초기화
    lin.zero_grad(set_to_none=True)
    loss.backward()
    print("W.grad.shape:", W.grad.shape, "| b.grad.shape:", b.grad.shape)
    print("x_req.grad.shape (grad wrt inputs):", x_req.grad.shape)

    # -------------------------------------------------------------------------
    header("Row = one output neuron (intuition)")
    # 각 출력 단위 k는 가중치 행 W[k](들어오는 가중치)와 편향 b[k]를 쓴다
    # 예제 하나 x[i]에 대해 output[k] ≈ x[i] @ W[k].T + b[k]
    i, k = 0, 1
    manual_k = x[i] @ W[k].t() + b[k]
    print(f"x[{i}].shape:", x[i].shape, "| W[{k}].shape:", W[k].shape)
    print(f"lin(x)[{i},{k}] =", lin(x)[i, k].item(), "| manual =", manual_k.item())

    # -------------------------------------------------------------------------
    header("Batch/out dims check with random shapes")
    for (Bb, inf, outf) in [(2, 4, 6), (7, 3, 1)]:
        xx = torch.randn(Bb, inf)
        ll = nn.Linear(inf, outf, bias=True)
        out1 = ll(xx)
        out2 = xx @ ll.weight.t() + ll.bias
        print(f"(B={Bb}, in={inf}, out={outf}) -> out.shape:", out1.shape,
              "| equal:", torch.allclose(out1, out2, atol=1e-6))

    # -------------------------------------------------------------------------
    header("Sanity: training step matches both forms")
    # F.linear와 x @ W.T + b를 비교하는 SGD 한 단계
    model = nn.Linear(in_features, out_features, bias=True)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    xbatch = torch.randn(B, in_features)
    ytarget = torch.randn(B, out_features)

    # 두 방식 모두로 순전파(결과는 같다)
    outA = F.linear(xbatch, model.weight, model.bias)
    outB = xbatch @ model.weight.t() + model.bias
    print("Forward equal (pre-step):", torch.allclose(outA, outB, atol=1e-6))

    lossA = F.mse_loss(outA, ytarget, reduction="mean")
    opt.zero_grad(set_to_none=True)
    lossA.backward()
    opt.step()

    outA2 = F.linear(xbatch, model.weight, model.bias)
    outB2 = xbatch @ model.weight.t() + model.bias
    print("Forward equal (post-step):", torch.allclose(outA2, outB2, atol=1e-6))

    # -------------------------------------------------------------------------
    header("Done")

if __name__ == "__main__":
    main()```

## 논의

이 코드는 `requires_grad=True`인 텐서에 대한 연산을 자동으로 추적하는 PyTorch의 autograd 체계를 보여준다. 스칼라 손실에 `.backward()`를 호출하면 autograd가 계산 그래프를 역방향으로 훑으며 연쇄 법칙을 적용해 모든 잎 텐서의 경사를 계산한다. 이 구조가 PyTorch의 모든 신경망 학습을 떠받친다.

PyTorch의 `nn.Module`은 신경망 구조를 정의하는 체계적인 방법을 제공한다. 각 모듈이 자신의 매개변수와 하위 모듈을 관리하므로 모델을 살펴보고, 저장하고, 장치 사이에 옮기기가 간편하다.

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
