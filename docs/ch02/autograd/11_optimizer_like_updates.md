# 최적화기 방식의 갱신

이 스크립트는 최적화기와 같은 방식의 매개변수 갱신을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 1. 코드

```python
"""최적화기처럼 고치기."""
import torch

# ========================================================================
# 메인
# ========================================================================

def main():

    w = torch.randn(3, requires_grad=True)
    lr = 0.1
    for step in range(3):
        # ---------------- 순전파 ----------------
        # 간단한 이차 손실 = 제곱합.
        # w.requires_grad=True이므로 이 손실 텐서도 경사를 필요로 한다
        # grad_fn(<SumBackward0>)을 가지므로 autograd가 backward()를 호출할 때
        # ∂loss/∂w를 계산할 줄 안다.
        loss = (w ** 2).sum()

        # ---------------- 경사 초기화 ----------------
        # 기본적으로 PyTorch는 .grad 버퍼에 경사를 누적한다.
        # 따라서 backward()를 다시 호출하기 전에 명시적으로 지워야 한다.
        # w.grad.zero_()는 안전하다:
        #   - .grad는 버퍼 텐서일 뿐 계산 그래프의 일부가 아니다.
        #   - .grad에 대한 제자리 연산은 autograd를 혼란시키지 않는다.
        #   - 여기서는 torch.no_grad()가 필요 없다.
        if w.grad is not None:
            w.grad.zero_()

        # ---------------- 역전파 ----------------
        # d(loss)/dw를 계산하여 결과를 w.grad에 저장한다.
        # 내부적으로는 loss.grad_fn을 통해 그래프를 훑으며 이루어진다.
        loss.backward()

        # ---------------- 매개변수 갱신 ----------------
        # ❌ 잘못됨: w -= lr * w.grad   (no_grad 밖에서)
        #   - 그래프의 일부로 기록될 것이다(grad_fn=<SubBackward0>).
        #   - w가 "잎" 텐서가 아니게 되고 .grad가 더 이상 올바르게 갱신되지 않는다.
        #   - 그래프가 매 단계 커진다 → 메모리 누수.
        #
        # ✅ 올바름: torch.no_grad()로 감싼다
        #   - 경사 추적을 일시적으로 끈다.
        #   - 제자리 갱신이 일어나지만 그래프에서는 제외된다.
        #   - w는 requires_grad=True인 잎 텐서로 남는다.
        with torch.no_grad():
            w -= lr * w.grad

        # 형식 지정(:.6f)이 동작하도록 스칼라 손실에는 .item()을 쓴다.
        print(f"step {step} | loss={loss.item():.6f} | w={w}")

    # ---------------- 학습 후 참고 ----------------
    # - w.requires_grad는 여전히 True이다. w는 여전히 학습 가능하다.
    # - no_grad() 때문에 autograd가 그 갱신들을 건너뛰었다.
    # - torch.optim.SGD와 Adam이 .step()을 구현하는 방식이 정확히 이것이다.
    print("Final w.requires_grad (still True, updates not tracked):", w.requires_grad)

if __name__ == "__main__":
    main()
```

## 2. 논의

이 코드는 `requires_grad=True`인 텐서에 대한 연산을 자동으로 추적하는 PyTorch의 autograd 체계를 보여준다. 스칼라 손실에 `.backward()`를 호출하면 autograd가 계산 그래프를 역방향으로 훑으며 연쇄 법칙을 적용해 모든 잎 텐서의 경사를 계산한다. 이 구조가 PyTorch의 모든 신경망 학습을 떠받친다.

경사 추적을 제어하는 것은 정확성과 성능 모두에 필수적이다. `torch.no_grad()` 컨텍스트 관리자는 매개변수 갱신이나 추론처럼 계산 그래프에 포함되어서는 안 되는 연산에 대해 autograd를 끈다. `.detach()` 메서드는 저장소는 공유하지만 그래프와는 분리된 텐서를 만들며, 값을 기록하거나 NumPy로 변환할 때 유용하다.

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

**다룬 것** — 최적화기 방식의 갱신

이 코드는 `requires_grad=True`인 텐서에 대한 연산을 자동으로 추적하는 PyTorch의 autograd 체계를 보여준다.

앞의 연습문제 3개로 직접 확인할 수 있다.
