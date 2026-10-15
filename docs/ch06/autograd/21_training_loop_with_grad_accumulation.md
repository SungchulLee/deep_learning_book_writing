# 경사 누적을 사용하는 학습 루프

마이크로배치 두 개마다 한 번 업데이트하는 학습 루프를 살펴본다. 역전파 횟수와 옵티마이저 단계 수를 구분한다.

## 1. 코드

```python
"""기울기를 쌓는 학습 루프."""
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

def main():

    torch.manual_seed(5)

    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)

    # 표본 N개의 가짜 데이터셋(특징 4개, 목표 1개)
    N = 12
    X = torch.randn(N, 4)
    y = torch.randn(N, 1)

    batch_size = 3
    accumulation_steps = 2  # effective batch size = batch_size * accumulation_steps = 6

    for epoch in range(2):
        opt.zero_grad(set_to_none=True)
        running_loss = 0.0

        for step in range(0, N, batch_size):
            xb = X[step : step + batch_size]
            yb = y[step : step + batch_size]

            # 순전파
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb, reduction="mean")

            # 누적된 경사가 전체 배치 평균 손실과 맞도록 손실 크기를 조정한다
            (loss / accumulation_steps).backward()
            running_loss += loss.item()

            # "accumulation_steps"개의 마이크로배치 후에만 가중치를 갱신한다
            if ((step // batch_size) + 1) % accumulation_steps == 0:
                opt.step()                     # apply one update
                opt.zero_grad(set_to_none=True)  # clear grads for next accumulation

        # 관찰을 위해 (마이크로배치당) 평균 손실을 보고한다
        print(f"Epoch {epoch}: avg loss per microbatch = {running_loss / (N / batch_size):.6f}")

    # 최종 학습된 매개변수 출력(detach()로 평범한 텐서로 만든다)
    print("Final model weights:\n", {n: p.detach() for n, p in model.named_parameters()})

if __name__ == "__main__":
    main()
```

**출력:**

```
Epoch 0: avg loss per microbatch = 2.736446
Epoch 1: avg loss per microbatch = 2.008073
Final model weights:
 {'weight': tensor([[ 0.1366, -0.4661,  0.2192, -0.1120]]), 'bias': tensor([0.4167])}
```

## 2. 논의

### 누적 구간별 처리

표본 12개를 크기 3의 마이크로배치 네 개로 처리한다. 각 손실을 2로 나누어 역전파하고, 두 번째와 네 번째 마이크로배치 뒤에 업데이트한다. 따라서 한 에포크의 역전파는 네 번, 옵티마이저 단계는 두 번이다. 경사 초기화는 에포크 시작과 각 업데이트 직후에 수행한다.

### 마지막에 남는 배치 처리

이 예제에서는 마이크로배치 수가 누적 횟수로 나누어떨어진다. 일반적인 데이터에서는 마지막 누적 구간이 짧을 수 있으므로, 남은 경사도 업데이트하고 실제 표본 수에 맞춰 손실 스케일을 정해야 한다. 그렇지 않으면 마지막 표본을 반영하지 않거나 경사를 과소평가할 수 있다.

출력의 평균 손실은 스케일을 나누기 전 마이크로배치 손실을 기록한 것이다. 경사 계산용 손실과 보고용 손실을 분리하면 누적 횟수가 바뀌어도 지표를 해석하기 쉽다. 이 평균에는 학습 도중 서로 다른 파라미터에서 계산한 손실이 섞여 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
SGD 대신 Adam 옵티마이저를 쓰도록 코드를 수정하라. 100 에포크에 걸친 수렴 속도를 비교하라.

</div>

??? success "연습문제 1 풀이"
    ```python
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Adam은 적응적 학습률과 모멘텀 덕분에 보통 SGD보다
    # 빠르게 수렴한다. 다만 Adam의 최적 학습률은
    # 보통 SGD보다 작다.
    ```

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
학습 루프에서 `optimizer.zero_grad()`를 없애면 어떤 일이 생기는가? 실험해 보고 학습 손실에 미치는 영향을 설명하라.

</div>

??? success "연습문제 2 풀이"
    업데이트 사이에 경사를 초기화하지 않으면 서로 다른 파라미터에서 계산한 이전 경사까지 누적된다. 따라서 현재 손실의 경사만 사용하는 학습 규칙과 달라지고, 진동이나 발산을 일으킬 수 있다. 경사끼리 상쇄될 수도 있으므로 크기가 항상 증가하거나 반드시 발산하는 것은 아니다. 의도적인 경사 누적에서는 정해진 여러 마이크로배치를 처리한 뒤 한 번 업데이트하고 경사를 초기화한다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
옵티마이저에 L2 정규화(가중치 감쇠)를 추가하고 그것이 최종 매개변수 값에 어떤 영향을 주는지 관찰하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
    # weight_decay는 손실에 L2 벌점항 lambda * ||w||^2을 더한다.
    # 이는 가중치를 작게 유도하여 과적합을 막을 수 있다.
    # 최종 가중치의 크기가 조금 더 작아진다.
    ```

## 정리하며

경사 누적에서는 역전파, 업데이트, 기록의 단위를 각각 정한다. 마지막 누적 구간의 처리와 손실의 분모도 빠뜨리지 않는다.
