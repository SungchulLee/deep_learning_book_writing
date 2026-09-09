# 큰 배치를 위한 경사 누적

이 스크립트는 큰 배치를 위한 경사 누적을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 1. 코드

```python
"""큰 배치을 위한 기울기 쌓기."""
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

def main():
    
    torch.manual_seed(2)

    # 아주 작은 선형 모델: y = Wx + b
    model = nn.Linear(5, 1, bias=True)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    # 목표: 더 작은 미니배치로 더 큰 "실효 배치 크기"를 흉내 낸다
    # 예: 실효 배치 크기 8을 원하지만 배치 크기 2까지만 들어간다
    batch_size = 2
    accumulation_steps = 4  # 2 × 4 = 8 effective batch size
    N = batch_size * accumulation_steps

    # 크기 N인 가짜 데이터셋
    X = torch.randn(N, 5)
    y = torch.randn(N, 1)

    # ------------------------------------------------------------
    # 경사 초기화
    # ------------------------------------------------------------
    opt.zero_grad()

    # ------------------------------------------------------------
    # 누적 루프
    # ------------------------------------------------------------
    for step in range(accumulation_steps):
        # 각 미니배치를 잘라낸다
        xb = X[step * batch_size : (step + 1) * batch_size]
        yb = y[step * batch_size : (step + 1) * batch_size]

        # 미니배치에 대해 순전파 + 손실 계산
        pred = model(xb)
        loss = nn.functional.mse_loss(pred, yb, reduction="mean")

        # 역전파 전에 손실의 크기를 조정한다:
        #   - 크기 조정을 하지 않으면: 각 .backward()가 전체 경사를 누적하므로
        #     따라서 미니배치 4개 후에는 경사가 4배 커진다.
        #   - 크기 조정을 하면: 누적된 경사 ≈ 크기 8인 큰 배치 하나.
        (loss / accumulation_steps).backward()

        # 중요: 아직 opt.step()을 호출하지 말 것. 모든 미니배치를 처리할 때까지 기다린다

    # 누적이 끝나면 최적화기 갱신을 한 번 적용한다
    opt.step()
    opt.zero_grad()
    print("Finished one optimizer step using accumulation with proper scaling.")

if __name__ == "__main__":
    main()
```

**출력:**

```
Finished one optimizer step using accumulation with proper scaling.
```

## 2. 논의

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사 초기화, 역전파, 매개변수 갱신이다. 각 구성 요소가 결정적인 역할을 한다. 최적화기는 갱신 규칙(SGD, Adam 등)을 캡슐화하고 학습률과 모멘텀 상태를 내부에서 관리한다.

여기서 보여준 패턴들은 실무적인 PyTorch 개발의 토대이다. 각 개념은 데이터 표현, 자동 미분, 하드웨어 가속을 하나의 일관된 API로 통합하는 텐서 추상화 위에 세워진다.

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

**다룬 것** — 큰 배치를 위한 경사 누적

학습 루프는 표준적인 PyTorch 패턴을 따른다.

앞의 연습문제 3개로 직접 확인할 수 있다.
