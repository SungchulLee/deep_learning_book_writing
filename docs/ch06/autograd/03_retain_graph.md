# 계산 그래프 재사용

같은 손실을 두 번 역전파하는 경우와, 손실을 새로 계산해 역전파하는 경우를 비교한다. 경사 초기화와 그래프 보존은 서로 다른 문제다.

## 1. 코드

```python
"""그래프를 남긴다."""
import torch

# ========================================================================
# 메인
# ========================================================================

def main():

    torch.manual_seed(0)

    x = torch.randn(3, requires_grad=True)
    print("x:", x)

    # 계산 그래프를 한 번만 만든다(loss = 제곱합).
    loss = (x ** 2).sum()

    # 첫 번째 역전파: 경사를 계산하여 x.grad를 채운다.
    # retain_graph=True는 사용 후에도 그래프를 해제하지 말라고 PyTorch에 알린다.
    # 이렇게 하면 같은 그래프에 대해 역전파를 또 할 수 있다.
    loss.backward(retain_graph=True)
    print("After 1st backward, x.grad:", x.grad)

    # 같은 그래프에 대한 두 번째 역전파(여기서 새 순전파는 없다).
    # 위에서 그래프를 유지했으므로 이 호출은 유효하다.
    # 여기서는 retain_graph=False(기본값)이므로 이 두 번째 역전파 후에
    # 그래프가 해제되어 다시 쓸 수 없게 된다.
    # 경사는 기본적으로 x.grad에 누적된다.
    loss.backward()
    print("After 2nd backward (accumulated), x.grad:", x.grad)

    # 다음 계산 전에 깨끗한 상태를 보기 위해 경사를 지운다.
    # 중요:
    #   - PyTorch 텐서는 제자리로 동작하는 .zero_()를 제공한다.
    #   - 텐서에는 .zero() 메서드가 없다. 호출하면 AttributeError가 난다.
    #   - 끝에 붙는 밑줄(_) 관례: 그 연산이 텐서를 변경한다는 뜻이다.
    #   - 이것이 경사를 초기화하는 표준적이고 관용적인 방법이다.
    x.grad.zero_()
    print("After zero_, x.grad:", x.grad)

    # 새로운 순전파: 완전히 새 계산 그래프를 만든다.
    # 이제 역전파가 0에서 시작해 경사를 다시 계산한다.
    loss = (x ** 2).sum()
    loss.backward()
    print("After 3rd backward (from zero), x.grad:", x.grad)

    # 참고:
    #   - 모델의 모든 매개변수 경사를 한 번에 초기화하려면
    #     optimizer.zero_grad()를 쓴다.
    #   - optimizer.zero_grad(set_to_none=True)는 경사를 0이 아니라 None으로 만든다.
    #     이렇게 하면 메모리를 아끼고 미묘한 누적 문제를 피할 수 있다.
    #   - 다만 텐서별로 직접 지울 때는 언제나 .zero_()를 쓴다.

if __name__ == "__main__":
    main()
```

**출력:**

```
x: tensor([ 1.5410, -0.2934, -2.1788], requires_grad=True)
After 1st backward, x.grad: tensor([ 3.0820, -0.5869, -4.3576])
After 2nd backward (accumulated), x.grad: tensor([ 6.1640, -1.1737, -8.7152])
After zero_, x.grad: tensor([0., 0., 0.])
After 3rd backward (from zero), x.grad: tensor([ 3.0820, -0.5869, -4.3576])
```

## 2. 논의

### 역전파에 필요한 중간 결과 보존하기

제곱합의 역전파에는 순전파에서 얻은 값이 필요하다. 기본 역전파는 사용을 마친 저장 텐서를 해제하므로, 같은 손실로 다시 역전파하면 필요한 값이 없어 오류가 발생할 수 있다. 첫 호출에 `retain_graph=True`를 주면 두 번째 호출에도 필요한 중간 결과를 유지한다. 이는 모든 그래프 메타데이터가 즉시 사라진다는 뜻은 아니다.

### 그래프 보존은 경사를 초기화하지 않는다

예제의 첫 역전파는 $2x$를 저장하고, 두 번째 역전파는 같은 경사를 더해 $4x$를 만든다. `x.grad.zero_()`는 이 누적값만 지운다. 이미 해제된 중간 결과를 복원하지는 않는다. 세 번째 역전파에서는 제곱합을 새로 계산해 새로운 그래프를 만든다.

일반적인 학습에서는 파라미터가 바뀔 때마다 순전파와 손실을 다시 계산한다. 이전 손실을 보존해 재사용하면 새 파라미터에서의 손실을 계산하는 작업을 대신할 수 없다. 그래프 보존은 같은 순전파에 여러 미분이 필요한 경우에 한정해 사용한다.

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
제곱합처럼 저장된 중간값이 필요한 손실에서 `retain_graph=True` 없이 같은 그래프에 `.backward()`를 두 번 호출하면 오류가 나는 이유를 설명하라. `retain_graph=True`는 메모리 사용량에 어떤 영향을 주는가?

</div>

??? success "연습문제 2 풀이"
    기본 역전파는 사용을 마친 저장 텐서를 해제한다. 같은 그래프의 두 번째 역전파에 이 값들이 필요하면 `RuntimeError`가 발생한다. 저장 텐서가 필요 없는 단순 연산은 반복 호출이 가능할 수도 있으므로 모든 그래프에서 반드시 오류가 나는 것은 아니다. 첫 호출의 `retain_graph=True`는 필요한 중간 결과를 보존해 재사용을 가능하게 하지만 그만큼 메모리를 더 오래 사용한다. `.grad` 초기화는 이 보존 여부와 별개의 작업이다.

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

`retain_graph`는 역전파용 중간 결과를 보존하고, 경사 초기화는 `.grad`의 누적값을 지운다. 새 파라미터에서 학습을 이어가려면 새 순전파가 필요하다.
