# 경사 누적을 쓰는 학습 루프

이 스크립트는 경사 누적을 사용하는 학습 루프을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

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
    main()```

## 논의

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
