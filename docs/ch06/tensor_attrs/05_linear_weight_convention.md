# Linear 웨이트 관례

수학식에서 웨이트를 어느 쪽에 곱하는지와 PyTorch가 웨이트를 저장하는 방향을 구별한다. `nn.Linear`의 입출력 모양을 알면 전치 실수를 줄일 수 있다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
PyTorch nn.Linear의 웨이트 관례와 같은 꼴들.

핵심:
- nn.Linear(in_features, out_features)이 담는 것:
    weight.shape == (out_features, in_features)
    bias.shape   == (out_features,)  (bias=True이면)
- 순전파은 다음과 같다.
    out = F.linear(x, weight, bias) == x @ weight.T + bias
  여기서 x.shape == (batch, in_features), out.shape == (batch, out_features)

We verify:
- 바이어스가 있을 때와 없을 때의 꼴과 같음
- weight.T은 보기일 뿐(베끼지 않는다)이며 필요하면 .contiguous()을 쓴다
- 자동 미분 그래디언트의 꼴
- 웨이트의 줄을 잘라 보며 "한 줄 = 출력 뉴런 하나"라는 느낌 잡기
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
    # 역전파할 수 있도록 간단한 스칼라 로스
    x_req = x.clone().requires_grad_(True)  # data as leaf (normally requires_grad=False)
    out = F.linear(x_req, W, b)
    loss = out.sum()
    # 파라메터의 경사 초기화
    lin.zero_grad(set_to_none=True)
    loss.backward()
    print("W.grad.shape:", W.grad.shape, "| b.grad.shape:", b.grad.shape)
    print("x_req.grad.shape (grad wrt inputs):", x_req.grad.shape)

    # -------------------------------------------------------------------------
    header("Row = one output neuron (intuition)")
    # 각 출력 단위 k는 웨이트 행 W[k](들어오는 웨이트)와 바이어스 b[k]를 쓴다
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
    main()
```

**출력:**

```
================================================================================
nn.Linear weight/bias shapes and equivalence to x @ W.T + b
================================================================================
W.shape: torch.Size([3, 5]) | b.shape: torch.Size([3])
Allclose(F.linear, x @ W.T + b): True

================================================================================
Without bias: same equivalence
================================================================================
W2.shape: torch.Size([3, 5]) | bias=None
Allclose(no-bias): True

================================================================================
Transpose view & contiguity
================================================================================
W.is_contiguous: True | W.t().is_contiguous: False
After .contiguous(), WTc.is_contiguous: True

================================================================================
Autograd shapes: grads wrt W and b
================================================================================
W.grad.shape: torch.Size([3, 5]) | b.grad.shape: torch.Size([3])
x_req.grad.shape (grad wrt inputs): torch.Size([4, 5])

================================================================================
Row = one output neuron (intuition)
================================================================================
x[0].shape: torch.Size([5]) | W[{k}].shape: torch.Size([5])
lin(x)[0,1] = 0.455205500125885 | manual = 0.455205500125885

================================================================================
Batch/out dims check with random shapes
================================================================================
(B=2, in=4, out=6) -> out.shape: torch.Size([2, 6]) | equal: True
(B=7, in=3, out=1) -> out.shape: torch.Size([7, 1]) | equal: True

================================================================================
Sanity: training step matches both forms
================================================================================
Forward equal (pre-step): True
Forward equal (post-step): True

================================================================================
Done
================================================================================
```

## 2. 논의

`nn.Linear(in_features, out_features)`의 웨이트 모양은 `(out_features, in_features)`다. 입력의 마지막 축이 `in_features`이면 출력의 마지막 축은 `out_features`가 된다.

배치의 표본을 행으로 두면 계산은 $Y=XW^\top+b$다. 한 표본을 열벡터로 적는 수학식 $y=Wx+b$와 같은 연산이며 모순이 아니다. 바이어스는 각 출력 피처에 하나씩 있으므로 파라메터 수는 $d_{out}d_{in}+d_{out}$이다.

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

수학식에서 웨이트를 어느 쪽에 곱하는지와 PyTorch가 웨이트를 저장하는 방향을 구별한다. `nn.Linear`의 입출력 모양을 알면 전치 실수를 줄일 수 있다.
