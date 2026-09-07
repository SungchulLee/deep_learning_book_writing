# retain_graph

이 스크립트는 `retain_graph`의 사용법을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

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
    main()```

## 논의

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
