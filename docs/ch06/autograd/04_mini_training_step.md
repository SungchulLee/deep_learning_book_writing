# 최소 학습 단계

세 개의 데이터로 직선 모델을 세 번 업데이트한다. 작은 루프 안에서 순전파, 로스, 경사 초기화, 역전파, 파라미터 업데이트의 관계를 확인한다.

## 1. 코드

```python
"""작은 학습 걸음."""
import torch

# ========================================================================
# 메인
# ========================================================================

def main():

    torch.manual_seed(0)

    # 아주 작은 데이터셋: 선형 회귀 장난감 예제
    # 입출력 쌍: x ∈ {-1, 0, 1}, 목표 y ∈ {1, 0, -1}
    x = torch.tensor([[-1.0], [0.0], [1.0]])  # shape (3,1)
    y = torch.tensor([[1.0], [0.0], [-1.0]])  # shape (3,1)

    # 학습 가능한 파라메터(잎 텐서).
    # 잎 텐서는 사용자가 requires_grad=True로 만든 텐서이다.
    # autograd가 이들에 대한 경사를 직접 계산한다.
    w = torch.randn(1, requires_grad=True)  # dtype is torch.float32 by default
    b = torch.zeros(1, requires_grad=True)  # dtype is torch.float32 by default

    lr = 0.1
    for step in range(3):
        # ---------------- 순전파 ----------------
        y_hat = x * w + b                     # linear model, shape (3,1)
        loss = torch.mean((y_hat - y) ** 2)   # mean squared error (scalar)
        # 참고:
        #   - loss는 0차원 텐서(스칼라 텐서)이다.
        #   - loss.grad_fn = <MeanBackward0>이다. 그 이유는
        #     계산 그래프의 평균 연산으로 만들어졌기 때문이다.
        #   - grad_fn은 다음을 계산할 줄 아는 함수를 가리킨다.
        #     이 텐서의 부모에 대한 경사를 계산하는 함수를 가리킨다.
        #   - (w, b 같은) 잎 텐서는 grad_fn=None이다.

        # ---------------- 역전파 ----------------
        # 경사는 기본적으로 누적되므로 먼저 예전 경사를 지운다.
        # 경사를 0이 아니라 None으로 되돌리려면 (제자리인) .zero_()를 쓴다.
        # 경사를 0이 아니라 None으로 만드는 것이 zero_()의 기본 동작이다.
        # PyTorch에는 .zero()가 없다(.zero_()만 있다).
        if w.grad is not None: w.grad.zero_()
        if b.grad is not None: b.grad.zero_()
        print(f"{w.grad = }")

        # 계산 그래프를 훑으며 d(loss)/dw와 d(loss)/db를 계산한다
        # loss.backward() → grad_fn → 부모 텐서 → 연쇄 법칙의 순서로 이루어진다.
        loss.backward()

        # ---------------- 파라메터 갱신 ----------------
        # 중요: autograd가 기록하지 않도록 torch.no_grad()로 감싼다
        # 이 제자리 갱신들을 계산 그래프에 기록하지 않도록 한다.
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad

        # ---------------- 기록 ----------------
        # 왜 .item()인가? 이유는:
        #   - loss, w, b는 텐서이다(모양이 []인 스칼라).
        #   - 텐서를 그대로 출력하면 "tensor(..., grad_fn=...)"이 보인다.
        #   - .item()은 순수한 파이썬 실수를 뽑아내므로 다음과 같은 형식 지정이
        #     :.6f나 :.4f가 오류 없이 동작한다.
        print(f"step {step}: loss={loss.item():.6f} | w={w.item():.4f} | b={b.item():.4f}")

    # 최종 학습된 파라메터
    print("Final params:", {"w": w.item(), "b": b.item()})
    # - loss.item()에 관하여:
    #   * loss.item()은 **CPU의 파이썬 실수를 반환한다**. `loss`가 GPU/MPS에 있으면
    #     .item()을 호출하면 **장치→호스트 복사와 동기화 장벽** 이 발생한다
    #     (`loss`를 만들어 내는 앞선 GPU 연산이 모두 끝날 때까지 호스트가 기다린다).
    #   * 이는 GPU 파이프라인을 멈추게 하므로, 빡빡한 학습 루프에서는 **
    #     매 단계마다 .item()을 호출하는 것을 피하는** 편이 낫다. 다음을 권한다:
    #       - 기록 빈도를 줄인다(예: k 단계마다).
    #       - 텐서 값을 누적해 두었다가 성능이 중요한 구간 밖에서 `.detach().cpu()`를 호출한다.
    #       - 평균을 낼 때는 이따금 동기화한 **뒤** 누적합을 CPU 실수로 유지한다.
    #   * 파이썬 실수가 정말로 필요할 때에만 .item()을 쓴다(출력, 조기 종료 판정 등).
    #     다만 그것이 만드는 동기화 지점에 유의해야 한다.

if __name__ == "__main__":
    main()
```

**출력:**

```
w.grad = None
step 0: loss=4.304441 | w=1.2022 | b=0.0000
w.grad = tensor([0.])
step 1: loss=3.233113 | w=0.9086 | b=0.0000
w.grad = tensor([0.])
step 2: loss=2.428427 | w=0.6541 | b=0.0000
Final params: {'w': 0.6540943384170532, 'b': 0.0}
```

## 2. 논의

### 모델과 로스

데이터는 $y=-x$를 따르며 모델은 $\hat y=wx+b$이다. 평균제곱오차를 줄이면 그래디언트 $w$는 -1, 절편 $b$는 0에 가까워지는 방향으로 학습된다. 세 번의 업데이트는 절차를 보여주기 위한 횟수이며, 수렴을 보장하는 설정은 아니다.

### 한 단계의 흐름

현재 파라미터로 예측과 로스를 계산한 뒤 이전 `.grad`를 지우고 `loss.backward()`를 호출한다. 이어서 `torch.no_grad()` 안에서 $w\leftarrow w-\eta\,\partial L/\partial w$와 같은 규칙으로 값을 바꾼다. 업데이트 연산을 기록하지 않아도 다음 순전파에서는 다시 경사를 추적한다.

출력의 로스는 업데이트 직전에 계산한 값이고, 같은 줄의 파라미터는 업데이트 후 값이다. 따라서 두 값이 정확히 같은 시점의 모델을 나타내는 것은 아니다. 역전파 전의 `w.grad` 출력은 첫 단계에서는 `None`, 이후에는 초기화된 경사를 보여준다.

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
학습 루프에서 경사 초기화와 파라미터 업데이트를 각각 언제 수행하는가? 업데이트에 `no_grad()`를 사용하는 이유는 무엇인가?

</div>

??? success "연습문제 2 풀이"
    현재 업데이트 단위의 첫 역전파 전에 이전 경사를 지우고, 필요한 경사를 계산한 뒤 파라미터를 바꾼다. 일반적인 학습에서 업데이트 자체를 기록하지 않도록 `no_grad()`를 사용한다. 이후 반복의 순전파는 갱신된 파라미터로 새로 수행한다.

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

학습 단계는 현재 파라미터의 로스를 미분하고 그 경사로 값을 바꾸는 과정이다. 로스와 파라미터를 기록할 때는 업데이트 전후의 시점을 구분한다.
