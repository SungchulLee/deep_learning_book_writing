# 스칼라를 텐서로

이 스크립트는 스칼라를 텐서로 바꾸는 방법을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 1. 코드

```python
"""홑값에서 텐서로."""
import torch

# ========================================================================
# 메인
# ========================================================================

def print_info(t):
    """얼른 들여다보기 위한 예쁜 찍개.

    Shows:
      - `t` 자체  : 값이 어떻게 보이는지는 데이터 클래스와 촘촘함에 달렸다
      - `t.shape`   : `torch.Size([])`은 홑값(계수 0)이라는 뜻이다
      - `t.dtype`   : 따로 밝히지 않으면 파이썬 값에서 미루어 정한다
      - `t.requires_grad` : 자동 미분이 `t`의 셈을 좇을지 여부
    """
    print(f"{t = }", f"{t.shape = }", f"{t.dtype = }", f"{t.requires_grad = }",
          sep="\n", end="\n\n")

def main():
    # --------------------------------------------
    # 1) 파이썬 int를 그대로 감싸기 → 스칼라 텐서
    # --------------------------------------------
    scalar_val = 42
    t1 = torch.tensor(scalar_val)  # dtype inferred → torch.int64
    print_info(t1)
    # 결과: tensor(42), shape=[], dtype=int64 → 진짜 스칼라(계수 0).
    # 정수 텐서는 경사를 요구할 수 없다(autograd는 실수/복소수에서 동작한다).

    # --------------------------------------------
    # 2) 같지만 dtype을 강제한다(여기서는 float32)
    # --------------------------------------------
    t2 = torch.tensor(scalar_val, dtype=torch.float32)
    print_info(t2)
    # 여전히 스칼라이며 이제 float32이다. requires_grad=True로 두면 autograd가 가능해진다.
    # 예를 들어 torch.tensor(scalar_val, dtype=torch.float32, requires_grad=True)는
    # 경사 계산에 참여하는 **잎** 스칼라를 만든다.

    # --------------------------------------------
    # 3) 스칼라를 리스트에 넣으면 → 더 이상 스칼라가 아니다
    # --------------------------------------------
    t3 = torch.tensor([scalar_val])
    print_info(t3)
    # 모양이 [1]이다. 계수 0이 아니라 계수 1(길이 1인 벡터)이다.

    # --------------------------------------------
    # 4) torch.scalar_tensor: 스칼라 입력을 위한 편리한 별칭
    # --------------------------------------------
    t4 = torch.scalar_tensor(scalar_val)
    print_info(t4)
    # 스칼라 입력에 대해 torch.tensor(scalar_val)과 동등하다(dtype은 추론된다).

    # --------------------------------------------
    # 5) 파이썬 float에서 → dtype의 기본값은 float32
    # --------------------------------------------
    float_val = 3.14
    t5 = torch.tensor(float_val)  # default float dtype is float32
    print_info(t5)

    # --------------------------------------------
    # 6) 원소가 1개인 텐서를 파이썬 스칼라로 바꾸고 되돌리기
    # --------------------------------------------
    vec = torch.tensor([10])
    scalar_extracted = vec.item()   # works only when numel()==1
    t6 = torch.tensor(scalar_extracted)  # back to a scalar tensor
    print_info(t6)
    # ❓ `item()`은 계수 0/1/2...에서 동작하는가?
    # • **텐서의 원소가 정확히 하나일 때에만 가능하다**:
    #     가능: 모양 [], [1], [1,1], ... (numel()==1)
    #     오류: 모양 [2], [1,2], ... (numel()>1)

    # 간단 시연: item()의 성공과 실패
    ok1 = torch.tensor(7)          # shape []
    ok2 = torch.tensor([[7]])      # shape [1,1]
    bad = torch.tensor([1, 2])     # shape [2]
    _ = ok1.item()                 # OK
    _ = ok2.item()                 # OK (still one element)
    try:
        _ = bad.item()             # ValueError: only one element tensors can be converted
    except ValueError as e:
        print("item() on multi-element tensor →", e, "\n")

    # --------------------------------------------
    # 7) 빈 모양 `()`을 쓰는 torch.full로 스칼라 만들기
    # --------------------------------------------
    t7 = torch.full((), 7.7)  # empty shape → rank-0 scalar
    print_info(t7)

    # --------------------------------------------
    # 8) autograd를 명시적으로 켜기(실수/복소수 텐서에 대해)
    # --------------------------------------------
    t8 = torch.tensor(5.0, requires_grad=True)  # leaf scalar with grad tracking
    print_info(t8)
    # 이 스칼라는 이제 autograd에 참여한다. 참고: requires_grad=True는 다음에만 유효하다
    # 실수/복소수 dtype에만 해당한다(정수는 아니다).

    # 간단 시연: 실수 스칼라를 통한 역전파
    y = 0.5 * (t8 ** 2)  # y = 1/2 x^2
    y.backward()         # dy/dx = x
    print("t8:", t8.item(), "requires_grad:", t8.requires_grad)
    print("t8.grad (expected 5.0):", t8.grad.item(), "\n")

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

**다룬 것** — 스칼라를 텐서로

이 코드는 `requires_grad=True`인 텐서에 대한 연산을 자동으로 추적하는 PyTorch의 autograd 체계를 보여준다.

앞의 연습문제 3개로 직접 확인할 수 있다.
