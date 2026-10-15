# 경사 초기화의 세 가지 방법

옵티마이저의 기본 초기화, 0으로 채우기, `None`으로 설정하기를 비교한다. 0인 경사와 경사가 없는 상태는 옵티마이저에서 다르게 처리될 수 있다.

## 1. 코드

```python
"""기울기를 0으로 만드는 세 가지 길."""
import torch

# ========================================================================
# 메인
# ========================================================================

def main():
    torch.manual_seed(1)

    w = torch.randn(3, requires_grad=True)
    opt = torch.optim.SGD([w], lr=0.1)

    # (a) 기본 동작: 이제 optimizer.zero_grad()는 경사를 None으로 만든다
    #     (최근 PyTorch 버전에서 바뀌었다. 예전 버전은 0을 썼다).
    loss = (w ** 2).sum()
    loss.backward()
    print("1) .grad after backward:", w.grad)
    opt.zero_grad()  # default set_to_none=True → clears grads by setting them to None
    print("   after opt.zero_grad():", w.grad)

    # (b) 명시적으로 0으로 채우기: set_to_none=False → 경사를 0 텐서로 바꾼다
    #     이렇게 하면 .grad가 0으로 채워진 텐서임이 보장된다(예전 동작).
    loss = (w ** 2).sum()
    loss.backward()
    print("2) .grad after backward:", w.grad)
    opt.zero_grad(set_to_none=False)
    print("   after opt.zero_grad(set_to_none=False):", w.grad)  # zero tensor

    # (c) 수동 초기화: w.grad=None을 직접 대입한다(그 텐서에 대해 set_to_none=True와 같다)
    loss = (w ** 2).sum()
    loss.backward()
    print("3) .grad after backward:", w.grad)
    w.grad = None
    print("   after w.grad=None:", w.grad)

if __name__ == "__main__":
    main()
```

**출력:**

```
1) .grad after backward: tensor([1.3227, 0.5338, 0.1234])
   after opt.zero_grad(): None
2) .grad after backward: tensor([1.3227, 0.5338, 0.1234])
   after opt.zero_grad(set_to_none=False): tensor([0., 0., 0.])
3) .grad after backward: tensor([1.3227, 0.5338, 0.1234])
   after w.grad=None: None
```

## 2. 논의

### None과 0의 차이

현재 PyTorch의 `optimizer.zero_grad()`는 기본적으로 `set_to_none=True`를 사용한다. 기존 경사를 `None`으로 설정하며, 다음 역전파에서 실제로 경사를 받는 파라미터에 경사 텐서가 만들어진다. 예제의 출력은 이 동작을 보여주지만 이전 버전에서는 기본값이 다를 수 있다.

`zero_grad(set_to_none=False)`는 기존 경사 텐서를 0으로 채운다. 이미 `None`인 경사를 모두 새 0 텐서로 만드는 기능은 아니다. `w.grad = None`은 한 텐서에 같은 초기화 방식을 직접 적용한다.

### 업데이트에서의 차이

일반적인 PyTorch 옵티마이저는 경사가 `None`인 파라미터를 건너뛴다. 반면 경사가 0인 텐서는 모멘텀이나 가중치 감쇠에 의해 값이 바뀔 수 있다. 초기화 방법을 선택할 때는 메모리 사용뿐 아니라 이번 역전파에 참여하지 않은 파라미터의 처리도 고려한다.

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
경사가 0인 파라미터와 `.grad=None`인 파라미터는 옵티마이저에서 같은가?

</div>

??? success "연습문제 2 풀이"
    일반적인 PyTorch 옵티마이저는 `.grad=None`인 파라미터를 건너뛴다. 반면 0인 경사는 업데이트 대상이 될 수 있으며, 모멘텀이나 가중치 감쇠에 의해 값이 바뀔 수 있다. 두 초기화 방식은 이 차이까지 고려해 선택한다.

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

경사 초기화는 이전 누적값을 제거하는 작업이다. `None`과 0은 이후 업데이트에서 의미가 다를 수 있으므로 사용하는 옵티마이저와 함께 이해한다.
