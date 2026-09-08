# Module을 이용한 선형 회귀

이 스크립트는 `nn.Module`을 이용한 선형 회귀을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

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

    # (b) 손실 곡선
    ax1.plot(range(steps), losses, lw=2)
    ax1.set_title("Training Loss (MSE) per Step")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")

    # (c) α(편향)의 궤적
    #ax2.plot(range(steps), betas, label="β (기울기)", lw=2)
    ax2.plot(range(steps), alphas, label="α (intercept)", lw=2)
    #ax2.axhline(2.0, color="k", ls="--", lw=1, alpha=0.7, label="참 β=2")
    ax2.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.7, label="true α=1")
    ax2.set_title("Parameter Alpha Convergence")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("alpha")
    ax2.legend()

    # (d) β(가중치)의 궤적
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
    main()```

## 2. 논의

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사 초기화, 역전파, 매개변수 갱신이다. 각 구성 요소가 결정적인 역할을 한다. 최적화기는 갱신 규칙(SGD, Adam 등)을 캡슐화하고 학습률과 모멘텀 상태를 내부에서 관리한다.

경사 추적을 제어하는 것은 정확성과 성능 모두에 필수적이다. `torch.no_grad()` 컨텍스트 관리자는 매개변수 갱신이나 추론처럼 계산 그래프에 포함되어서는 안 되는 연산에 대해 autograd를 끈다. `.detach()` 메서드는 저장소는 공유하지만 그래프와는 분리된 텐서를 만들며, 값을 기록하거나 NumPy로 변환할 때 유용하다.

## 연습문제

**연습문제 1.**
SGD 대신 Adam 최적화기를 쓰도록 코드를 수정하라. 100 에폭에 걸친 수렴 속도를 비교하라.

??? success "연습문제 1 풀이"
    ```python
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Adam은 적응적 학습률과 모멘텀 덕분에 보통 SGD보다
    # 빠르게 수렴한다. 다만 Adam의 최적 학습률은
    # 보통 SGD보다 작다.
    ```

---


**연습문제 2.**
학습 루프에서 `optimizer.zero_grad()`를 없애면 어떤 일이 생기는가? 실험해 보고 학습 손실에 미치는 영향을 설명하라.

??? success "연습문제 2 풀이"
    `optimizer.zero_grad()`가 없으면 경사가 반복에 걸쳐 누적된다. 실효 경사가 매 단계 커져서 매개변수 갱신이 점점 커진다. 학습이 불안정해지고 손실은 대개 발산한다. PyTorch가 경사 누적 패턴을 지원하기 위해 기본적으로 경사를 누적하기 때문이다.

---


**연습문제 3.**
최적화기에 L2 정칙화(가중치 감쇠)를 추가하고 그것이 최종 매개변수 값에 어떤 영향을 주는지 관찰하라.

??? success "연습문제 3 풀이"
    ```python
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
    # weight_decay는 손실에 L2 벌점항 lambda * ||w||^2을 더한다.
    # 이는 가중치를 작게 유도하여 과적합을 막을 수 있다.
    # 최종 가중치의 크기가 조금 더 작아진다.
    ```

## 정리하며

**다룬 것** — Module을 이용한 선형 회귀

학습 루프는 표준적인 PyTorch 패턴을 따른다.

앞의 연습문제 3개로 직접 확인할 수 있다.
