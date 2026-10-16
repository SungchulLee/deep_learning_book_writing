# 경사 추적 끄기

잎 텐서의 `requires_grad`를 바꾸어 이후 연산의 기록 여부를 제어한다. 추적할 대상이 없는 로스에 역전파를 요청하면 어떻게 되는지도 살펴본다.

## 1. 코드

```python
"""그래디언트 좇기를 끈다."""
import torch

# ========================================================================
# 메인
# ========================================================================

def main():

    # 기본적으로 텐서는 requires_grad=False로 만들어진다
    # (대부분의 텐서는 파라메터가 아니라 그냥 데이터이기 때문이다).
    # 최적화하려는 파라메터에는 requires_grad=True를 명시적으로 설정해야 한다.
    w = torch.randn(4, requires_grad=True)  # Equivalent: w = torch.randn(4).requires_grad_()
    print("Before: w.requires_grad =", w.requires_grad)  # True

    # --- requires_grad_와 torch.no_grad 비교(논의) -----------------
    # • w.requires_grad_(False)
    #     - 메타데이터 플래그만 제자리에서 바꾼다. 텐서 값은 바뀌지 않는다.
    #     - 이 동작은 autograd가 추적하지 않으므로 torch.no_grad()로 감싸는 것은
    #       불필요하며 아무 이득이 없다.
    #
    # • with torch.no_grad():
    #     - 값을 바꾸는 연산을 autograd 그래프에
    #       기록하고 싶지 않을 때 쓴다(예: w.add_(...), w.copy_(...), 수동 갱신).
    #     - 예:
    #         with torch.no_grad():
    #             w.add_(update)   # 추적 없이 값을 바꾼다
    # ---------------------------------------------------------------------------
    print("Disabling gradient tracking for w ...")
    w.requires_grad_(False)  # metadata toggle; no value change; no need for torch.no_grad()
    print("After:  w.requires_grad =", w.requires_grad)  # False

    # autograd의 규칙:
    #   어떤 연산의 모든 입력이 requires_grad=False이면
    #   결과도 requires_grad=False이다(autograd 이력이 없다).
    loss = (w ** 2).sum()
    print("loss.requires_grad (expect False):", loss.requires_grad)

    try:
        loss.backward()
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
```

**출력:**

```
Before: w.requires_grad = True
Disabling gradient tracking for w ...
After:  w.requires_grad = False
loss.requires_grad (expect False): False
element 0 of tensors does not require grad and does not have a grad_fn
```

## 2. 논의

### 잎 텐서의 추적 설정 변경

예제의 `w`는 직접 만든 잎 텐서다. `w.requires_grad_(False)` 이후 제곱합을 계산하면, 이 연산에는 경사를 요구하는 입력이 없으므로 로스도 경사를 추적하지 않는다. 이 로스의 `.backward()`는 미분할 그래프가 없어 실패한다.

다만 입력 하나의 추적을 껐다고 연산 전체의 기록이 항상 꺼지는 것은 아니다. 다른 입력에 `requires_grad=True`가 설정되어 있으면 그 입력에 대한 미분 경로는 기록될 수 있다. 이미 만들어진 그래프를 소급해서 없애는 기능으로 해석해서도 안 된다.

### 중간 텐서와 파라미터 동결

미분 가능한 연산으로 만들어진 비잎 텐서의 `requires_grad`를 같은 방식으로 끄려고 하면 오류가 날 수 있다. 중간 결과를 분리하려면 `detach()`를 사용한다. 파라미터를 동결할 때는 기존 `.grad`도 점검해야 한다. 이전 경사가 남아 있으면 옵티마이저가 이를 사용해 값을 바꿀 수 있기 때문이다.

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
한 파라미터의 `requires_grad`를 끄면 관련 출력도 항상 경사를 추적하지 않는가?

</div>

??? success "연습문제 2 풀이"
    아니다. 기록이 켜져 있고 다른 입력이 경사를 요구하면 그 입력에 대한 경로는 기록될 수 있다. 또한 `requires_grad=False`는 이전 `.grad`를 지우지 않으므로 학습 중 파라미터를 동결할 때 남은 경사도 정리해야 한다.

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

`requires_grad_(False)`는 잎 텐서의 이후 경사 추적을 제어한다. 중간 결과의 분리나 기존 경사의 제거까지 자동으로 수행하는 것은 아니다.
