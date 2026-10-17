# 모듈로 선형 회귀 학습하기

앞에서 직접 관리한 파라미터와 업데이트를 `nn.Linear`, `nn.MSELoss`, `torch.optim.SGD`로 옮긴다. 추상화가 바뀌어도 학습의 수학적 의미는 같다.

## 1. 코드

```python
#!/usr/bin/env python3
"""단원을 쓰는 선형 회귀."""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

def main():
    torch.manual_seed(0)

    # ------------------------------------------------------------
    # 1) 합성 데이터: y = 1 + 2x + ε,  ε ~ N(0, 0.2^2)
    # ------------------------------------------------------------
    n = 100
    x = torch.randn(n, 1)                    # x ~ N(0,1)
    noise = 0.2 * torch.randn(n, 1)
    y = 1.0 + 2.0 * x + noise                # target

    # ------------------------------------------------------------
    # 2) 모델: nn.Linear(1,1)
    # ------------------------------------------------------------
    model = nn.Linear(1, 1)                  # y_hat = w*x + b
    criterion = nn.MSELoss()                 # mean squared error
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # ------------------------------------------------------------
    # 3) 학습 루프
    # ------------------------------------------------------------
    steps = 200
    losses = []
    alphas, betas = [], []   # store bias (α) and weight (β)

    for step in range(steps):
        # -------- 순전파 --------
        y_hat = model(x)
        loss = criterion(y_hat, y)

        # -------- 역전파 --------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 값 기록
        losses.append(loss.item())
        betas.append(model.weight.item())
        alphas.append(model.bias.item())

        if step % 20 == 0 or step == steps - 1:
            print(f"step {step:3d}: loss={loss.item():.6f} | β={model.weight.item():.4f} | α={model.bias.item():.4f}")

    print("Final params:", {"beta (slope)": model.weight.item(), "alpha (intercept)": model.bias.item()})

    # ------------------------------------------------------------
    # 4) 시각화
    # ------------------------------------------------------------
    x_np = x.detach().cpu().numpy().reshape(-1)
    y_np = y.detach().cpu().numpy().reshape(-1)

    # 매끄러운 적합 직선을 위해
    sort_idx = x_np.argsort()
    x_sorted = x_np[sort_idx]
    yhat_sorted = (model.weight.item() * x_sorted + model.bias.item())

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(18, 4))

    # (a) 데이터와 적합 결과
    ax0.scatter(x_np, y_np, alpha=0.5, label="data")
    ax0.plot(x_sorted, yhat_sorted, lw=3, label="fitted line")
    ax0.set_title("Linear Fit on Synthetic Data")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.legend()

    # (b) 로스 곡선
    ax1.plot(range(steps), losses, lw=2)
    ax1.set_title("Training Loss (MSE) per Step")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")

    # (c) α(바이어스)의 궤적
    #ax2.plot(range(steps), betas, label="β (그래디언트)", lw=2)
    ax2.plot(range(steps), alphas, label="α (intercept)", lw=2)
    #ax2.axhline(2.0, color="k", ls="--", lw=1, alpha=0.7, label="참 β=2")
    ax2.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.7, label="true α=1")
    ax2.set_title("Parameter Alpha Convergence")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("alpha")
    ax2.legend()

    # (d) β(웨이트)의 궤적
    ax3.plot(range(steps), betas, label="β (slope)", lw=2)
    #ax3.plot(range(steps), alphas, label="α (절편)", lw=2)
    ax3.axhline(2.0, color="k", ls="--", lw=1, alpha=0.7, label="true β=2")
    #ax3.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.7, label="참 α=1")
    ax3.set_title("Parameter Beta Convergence")
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("beta")
    ax3.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
```

**출력:**

```
step   0: loss=3.672616 | β=0.5150 | α=0.9367
step  20: loss=0.033739 | β=1.9631 | α=1.0336
step  40: loss=0.033449 | β=1.9761 | α=1.0320
step  60: loss=0.033449 | β=1.9762 | α=1.0319
step  80: loss=0.033449 | β=1.9762 | α=1.0319
step 100: loss=0.033449 | β=1.9762 | α=1.0319
step 120: loss=0.033449 | β=1.9762 | α=1.0319
step 140: loss=0.033449 | β=1.9762 | α=1.0319
step 160: loss=0.033449 | β=1.9762 | α=1.0319
step 180: loss=0.033449 | β=1.9762 | α=1.0319
step 199: loss=0.033449 | β=1.9762 | α=1.0319
Final params: {'beta (slope)': 1.9762046337127686, 'alpha (intercept)': 1.0319361686706543}
```

## 2. 논의

### 직접 만든 파라미터를 모듈로 옮기기

`nn.Linear(1, 1)`은 그래디언트와 절편에 해당하는 `weight`, `bias`를 등록한다. 입력과 정답은 모두 $(N,1)$이므로 예측과 정답을 원소별로 비교할 수 있다. 정답을 $(N,)$으로 바꾸면 브로드캐스팅으로 의도하지 않은 $(N,N)$ 오차가 만들어질 수 있다.

### 옵티마이저가 담당하는 일

`optimizer.zero_grad()`는 등록된 파라미터의 이전 경사를 지우고, `loss.backward()`는 새 경사를 계산한다. `optimizer.step()`은 그 경사와 SGD 규칙으로 파라미터를 업데이트한다. `step()`은 로스를 계산하거나 역전파를 대신하지 않는다.

네 개의 그래프는 데이터 적합, 로스, 절편, 그래디언트의 변화를 각각 보여준다. 잡음이 있는 표본을 학습하므로 파라미터가 생성 규칙의 값과 조금 다른 곳에서 안정되어도 이상한 것은 아니다. 기록된 로스는 각 업데이트 전 값이며, 기록된 파라미터는 업데이트 후 값이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
SGD 대신 Adam 옵티마이저를 쓰도록 코드를 수정하라. 100 에포크에 걸친 수렴 속도를 비교하라.

</div>

??? success "풀이"
    ```python
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Adam은 적응적 러닝레이트와 모멘텀 덕분에 보통 SGD보다
    # 빠르게 수렴한다. 다만 Adam의 최적 러닝레이트는
    # 보통 SGD보다 작다.
    ```

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
학습 루프에서 `optimizer.zero_grad()`를 없애면 어떤 일이 생기는가? 실험해 보고 학습 로스에 미치는 영향을 설명하라.

</div>

??? success "풀이"
    업데이트 사이에 경사를 초기화하지 않으면 서로 다른 파라미터에서 계산한 이전 경사까지 누적된다. 따라서 현재 로스의 경사만 사용하는 학습 규칙과 달라지고, 진동이나 발산을 일으킬 수 있다. 경사끼리 상쇄될 수도 있으므로 크기가 항상 증가하거나 반드시 발산하는 것은 아니다. 의도적인 경사 누적에서는 정해진 여러 마이크로배치를 처리한 뒤 한 번 업데이트하고 경사를 초기화한다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
옵티마이저에 L2 정규화(웨이트 감쇠)를 추가하고 그것이 최종 파라메터 값에 어떤 영향을 주는지 관찰하라.

</div>

??? success "풀이"
    ```python
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
    # weight_decay는 로스에 L2 벌점항 lambda * ||w||^2을 더한다.
    # 이는 웨이트를 작게 유도하여 과적합을 막을 수 있다.
    # 최종 웨이트의 크기가 조금 더 작아진다.
    ```

## 정리하며

모듈은 파라미터를 등록하고 옵티마이저는 업데이트를 수행한다. 순전파와 로스, 경사 계산의 역할을 구분하면 더 복잡한 모델에도 같은 학습 루프를 적용할 수 있다.
