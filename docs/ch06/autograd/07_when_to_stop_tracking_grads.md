# 경사 추적을 멈추는 시점

학습에 필요한 미분 경로와 예측·기록용 계산을 구분한다. 경사 추적을 끄는 기준은 텐서를 사용하는 목적이다.

## 1. 코드

```python
"""언제 기울기 좇기를 멈출까."""
import torch

# ========================================================================
# 메인
# ========================================================================

def main():

    x = torch.randn(3, requires_grad=True)
    y = x.sin().sum()  # scalar loss built from differentiable ops
    y.backward()
    # 기본적으로 PyTorch는 requires_grad=True인 텐서의 경사를 추적한다
    print("Grad tracked by default → x.grad:", x.grad)

    # 어떤 상황에서는 torch.no_grad()로 경사 추적을 멈추고 싶다:
    #   - 추론/평가 중: 경사가 필요 없으므로 메모리와 계산을 아낀다
    #   - 매개변수 갱신 중: 예를 들어 w -= lr * w.grad는 추적되면 안 된다
    #   - 이후 경사에 기여하지 않기를 명시적으로 바라는 모든 연산에 대해
    print("Typical places to STOP tracking:")
    print("- Inference/evaluation forward pass")
    print("- Optimizer-like parameter updates")
    print("- Any operation that should NOT influence future gradients")

if __name__ == "__main__":
    main()
```

**출력:**

```
Grad tracked by default → x.grad: tensor([ 0.4500, -0.5676, -0.4430])
Typical places to STOP tracking:
- Inference/evaluation forward pass
- Optimizer-like parameter updates
- Any operation that should NOT influence future gradients
```

## 2. 논의

### 입력이나 파라미터를 미분해야 하는가

예제의 $L=\sum_i\sin x_i$는 입력의 경사를 확인하려는 계산이므로 추적이 필요하다. 반면 학습이 끝난 모델의 예측값만 구할 때는 보통 역전파가 필요 없다. 이때 `torch.no_grad()`로 계산 기록에 드는 비용을 줄일 수 있다.

추론 중에도 입력 최적화나 민감도 분석처럼 입력에 대한 미분을 구한다면 추적을 유지해야 한다. 학습·평가라는 이름만으로 결정하지 말고 어떤 값에 대한 미분을 사용할지 판단한다.

### 목적에 맞는 제어 방법

특정 파라미터를 고정하려면 `requires_grad_(False)`, 계산 구간의 기록을 끄려면 `torch.no_grad()`, 이미 계산한 결과의 연결을 끊으려면 `detach()`를 사용한다. 모델의 `eval()`은 Dropout과 BatchNorm 등의 동작을 바꾸는 별도 기능이며 autograd를 끄지 않는다. 파라미터 업데이트는 보통 경사 기록을 끈 상태에서 수행한다.

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
입력 민감도를 분석하려는 추론에서도 `no_grad()`를 사용해도 되는가? `model.eval()`과의 차이를 설명하라.

</div>

??? success "연습문제 2 풀이"
    입력에 대한 미분이 필요하면 해당 경로의 추적을 유지해야 하므로 `no_grad()`로 감싸지 않는다. `model.eval()`은 Dropout과 BatchNorm 등의 평가 동작을 선택하며, autograd를 끄지는 않는다. 따라서 평가 모드에서도 입력 경사를 구할 수 있다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
잎 텐서 `w`를 만들고, 반복마다 같은 식으로 손실을 새로 계산하라. 경사를 초기화하지 않고 `.backward()`를 세 번 호출하며 매번 `w.grad`를 출력하라. 관찰된 값을 설명하라.

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

경사 추적은 필요한 미분 경로에만 유지한다. 파라미터 동결, 계산 구간 제어, 텐서 분리, 모델 평가 모드를 서로 구분한다.
