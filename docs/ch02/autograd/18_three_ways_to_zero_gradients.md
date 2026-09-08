# 경사를 0으로 만드는 세 가지 방법

이 스크립트는 경사를 0으로 만드는 세 가지 방법을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

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

## 2. 논의

이 코드는 `requires_grad=True`인 텐서에 대한 연산을 자동으로 추적하는 PyTorch의 autograd 체계를 보여준다. 스칼라 손실에 `.backward()`를 호출하면 autograd가 계산 그래프를 역방향으로 훑으며 연쇄 법칙을 적용해 모든 잎 텐서의 경사를 계산한다. 이 구조가 PyTorch의 모든 신경망 학습을 떠받친다.

여기서 보여준 패턴들은 실무적인 PyTorch 개발의 토대이다. 각 개념은 데이터 표현, 자동 미분, 하드웨어 가속을 하나의 일관된 API로 통합하는 텐서 추상화 위에 세워진다.

## 연습문제

**연습문제 1.**
함수 $f(x) = x^3 - 2x^2 + x$를 생각하자. PyTorch autograd를 사용하여 $f'(3)$을 계산하라.

??? success "연습문제 1 풀이"
    ```python
    import torch

    x = torch.tensor(3.0, requires_grad=True)
    f = x**3 - 2*x**2 + x
    f.backward()
    print(x.grad)  # f'(x) = 3x^2 - 4x + 1 = 27 - 12 + 1 = 16.0
    ```

---


**연습문제 2.**
`retain_graph=True` 없이 같은 계산 그래프에 `.backward()`를 두 번 호출하면 오류가 나는 이유를 설명하라. `retain_graph=True`는 메모리 사용량에 어떤 영향을 주는가?

??? success "연습문제 2 풀이"
    기본적으로 PyTorch는 메모리를 아끼기 위해 `.backward()` 후에 계산 그래프를 해제한다. `.backward()`를 두 번째로 호출하면 더 이상 존재하지 않는 그래프를 훑으려 하므로 `RuntimeError`가 발생한다. `retain_graph=True`로 두면 그래프가 메모리에 남아 재사용할 수 있지만, 모든 중간 텐서가 할당된 채로 남으므로 메모리 소비가 늘어난다.

---


**연습문제 3.**
잎 텐서 `w`를 만들고 손실을 계산한 뒤, 경사를 초기화하지 않고 `.backward()`를 세 번 호출하며 매번 `w.grad`를 출력하는 코드를 작성하라. 관찰된 값을 설명하라.

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

**다룬 것** — 경사를 0으로 만드는 세 가지 방법

이 코드는 `requires_grad=True`인 텐서에 대한 연산을 자동으로 추적하는 PyTorch의 autograd 체계를 보여준다.

앞의 연습문제 3개로 직접 확인할 수 있다.
