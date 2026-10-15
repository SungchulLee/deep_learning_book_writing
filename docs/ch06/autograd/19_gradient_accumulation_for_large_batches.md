# 큰 배치를 위한 경사 누적

작은 마이크로배치 네 개의 경사를 모아 한 번 업데이트한다. 전체 평균 손실에 맞춰 각 손실을 나누는 이유를 설명한다.

## 1. 코드

```python
"""큰 배치를 위한 기울기 쌓기."""
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

### 유효 배치 크기와 업데이트 횟수

예제는 표본 8개를 크기 2의 마이크로배치 네 개로 나눈다. 각 순전파 직후 역전파하여 경사를 모으고, 네 번을 마친 뒤 `opt.step()`을 한 번 호출한다. 계산 그래프를 네 개 모두 보존할 필요 없이 `.grad`에 결과만 누적할 수 있다.

### 평균 손실의 스케일 맞추기

각 마이크로배치 손실은 내부 표본의 평균이다. 이를 네 개 더하면 전체 배치 평균의 네 배가 되므로 `loss / accumulation_steps`를 역전파한다. 배치 크기가 다르면 단순히 개수로 나누는 대신 각 손실이 평균 낸 원소 수에 맞춰 가중한다.

이 예제의 선형 모델에서는 같은 파라미터에서 계산한 누적 경사가 전체 배치 경사와 반올림 오차 범위에서 일치한다. BatchNorm처럼 배치 내부 통계에 의존하는 연산이나 확률적 연산이 있으면 누적 학습이 큰 배치 실행과 완전히 같지는 않을 수 있다.

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

경사 누적은 메모리 사용을 줄이며 업데이트당 처리 표본 수를 늘리는 방법이다. 손실의 평균 기준과 실제 업데이트 경계를 함께 맞춘다.
