# 범위를 텐서로

이 스크립트는 범위를 텐서로 바꾸는 방법을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 1. 코드

```python
"""범위에서 텐서로."""
import torch

# ========================================================================
# 메인
# ========================================================================

def print_info(t):
    # 텐서를 빠르게 살펴보는 도우미 함수.
    # - t: 값 미리보기(PyTorch가 dtype에 따라 보기 좋게 출력한다)
    # - t.shape: 1차원은 [N], 2차원은 [R,C], 스칼라(0차원)는 []
    # - t.dtype: 지정하지 않으면 추론된다(기본적으로 정수→int64, 실수→float32)
    # - requires_grad: autograd 플래그(실수/복소수 텐서에 True로 설정하지 않으면 False)
    print(f"{t = }", f"{t.shape = }", f"{t.dtype = }", f"{t.requires_grad = }", sep="\n", end="\n\n")

def main():
    # --------------------------------------------
    # 1) 파이썬 range  →  텐서 (COPY)
    # --------------------------------------------
    # torch.tensor(range(...))는 반복 가능 객체에서 원소를 복사한다.
    # dtype이 추론된다. 정수는 기본적으로 모두 torch.int64가 된다.
    r1 = range(5)   # [0, 1, 2, 3, 4]
    t1 = torch.tensor(r1)
    print_info(t1)
    # 기댓값: tensor([0, 1, 2, 3, 4])   torch.Size([5])   torch.int64

    # --------------------------------------------
    # 2) (start, stop, step)을 쓰는 range
    # --------------------------------------------
    # 파이썬의 range는 반개구간이다. start는 포함하고 stop은 제외한다.
    r2 = range(2, 10, 2)  # [2, 4, 6, 8]
    t2 = torch.tensor(r2)
    print_info(t2)
    # 기댓값: tensor([2, 4, 6, 8])   torch.Size([4])   torch.int64

    # --------------------------------------------
    # 3) torch.arange  — range를 감싸는 것보다 낫다
    # --------------------------------------------
    # 텐서를 곧바로 만든다(dtype/device/requires_grad를 지정할 수 있다).
    # 파이썬 range와 같다: 반개구간 [start, stop).
    t3 = torch.arange(0, 10, 2)
    print_info(t3)
    # 기댓값: tensor([0, 2, 4, 6, 8])   torch.Size([5])   torch.int64

    # --------------------------------------------
    # 4) 실수 보폭을 쓰는 torch.arange
    # --------------------------------------------
    # 실수 보폭은 반올림 오차가 누적될 수 있다. 결과는 여전히 반개구간이다.
    # dtype=...으로 지정하지 않으면 기본 실수 dtype은 float32이다
    t4 = torch.arange(0.0, 1.0, 0.2)
    print_info(t4)
    # 예시 출력: tensor([0.0000, 0.2000, 0.4000, 0.6000, 0.8000])

    # --------------------------------------------
    # 5) torch.linspace — 양쪽 끝점을 모두 포함한다
    # --------------------------------------------
    # start부터 end까지(양끝 포함) 균등 간격의 점 `steps`개를 반환한다.
    # 이는 arange의 반개구간 동작과 다르다.
    t5 = torch.linspace(0, 1, steps=5)
    print_info(t5)
    # 기댓값: tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])

    # --------------------------------------------
    # 6) torch.range — 폐기 예정(대신 arange를 쓴다)
    # --------------------------------------------
    # 예전에는 있었지만 모호성 때문에 폐기되었다. torch.arange를 쓰는 것이 낫다.
    # 여기서는 폐기 경고를 피하기 위해 *대체* 호출을 보여준다.
    t6 = torch.arange(1, 5)  # [1, 2, 3, 4]
    print_info(t6)

    # ---------------------- 추가 참고 사항 ----------------------
    # • torch.arange/linspace에서는 dtype/device를 직접 제어할 수 있다:
    #     torch.arange(0, 10, 2, dtype=torch.float32, device='cpu', requires_grad=False)
    # • 균등 간격의 *개수*가 중요하면(양끝 포함) linspace가 낫다.
    #   보폭 기반 수열(끝값 제외)에는 arange가 낫다.
    # • 실수 보폭에서 마지막 값이 정확히 떨어지는 것이 중요하다면
    #   끝점이 중요하다면 보통 linspace가 더 안전한 선택이다.

if __name__ == "__main__":
    main()```

## 2. 논의

여기서 보여준 패턴들은 실무적인 PyTorch 개발의 토대이다. 각 개념은 데이터 표현, 자동 미분, 하드웨어 가속을 하나의 일관된 API로 통합하는 텐서 추상화 위에 세워진다.

이 연산들을 익히면 고수준 프레임워크가 제공하는 것을 넘어서는 사용자 정의 모델과 학습 절차를 효율적으로 구현할 수 있다.

## 연습문제

**연습문제 1.**
경사 누적을 사용하여 크기 16인 마이크로배치 4개로 실효 배치 크기 64를 흉내 내라. 완전한 학습 단계를 보여라.

??? success "연습문제 1 풀이"
    ```python
    accumulation_steps = 4
    optimizer.zero_grad()
    for i in range(accumulation_steps):
        xb = X[i*16:(i+1)*16]
        yb = y[i*16:(i+1)*16]
        loss = criterion(model(xb), yb) / accumulation_steps
        loss.backward()
    optimizer.step()
    ```

---


**연습문제 2.**
올바른 경사 누적을 위해 `.backward()`를 호출하기 전에 손실을 `accumulation_steps`로 나누어야 하는 이유를 설명하라.

??? success "연습문제 2 풀이"
    마이크로배치마다의 `.backward()` 호출이 경사를 누적한다. 크기를 조정하지 않으면 $K$개의 마이크로배치 후 누적된 경사가 배치당 경사의 $K$배가 되는데, 이는 손실을 평균 내는 것이 아니라 더하는 것과 같다. $K$로 나누면 누적된 경사가 전체 배치에 대해 `reduction='mean'`으로 순전파를 한 번 했을 때의 결과와 일치한다.

---


**연습문제 3.**
표본 8개를 한 배치로 한꺼번에 처리했을 때의 경사와, (적절히 크기를 조정하여) 표본 2개짜리 마이크로배치 4개로 경사를 누적했을 때의 경사를 비교하라. 둘이 일치함을 확인하라.

??? success "연습문제 3 풀이"
    ```python
    model_a = nn.Linear(3, 1)
    model_b = nn.Linear(3, 1)
    with torch.no_grad():
        model_b.weight.copy_(model_a.weight)
        model_b.bias.copy_(model_a.bias)
    X = torch.randn(8, 3); y = torch.randn(8, 1)
    # 전체 배치
    loss_a = nn.functional.mse_loss(model_a(X), y)
    loss_a.backward()
    # 누적
    model_b.zero_grad()
    for i in range(4):
        loss_b = nn.functional.mse_loss(model_b(X[i*2:(i+1)*2]), y[i*2:(i+1)*2]) / 4
        loss_b.backward()
    print(torch.allclose(model_a.weight.grad, model_b.weight.grad, atol=1e-5))
    ```

## 정리하며

**다룬 것** — 범위를 텐서로

여기서 보여준 패턴들은 실무적인 PyTorch 개발의 토대이다.

앞의 연습문제 3개로 직접 확인할 수 있다.
