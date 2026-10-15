# torch.no_grad 사용하기

블록 안에서만 역전파용 연산 기록을 끄는 방법을 살펴본다. 입력 텐서의 추적 설정은 유지된다.

## 1. 코드

```python
"""torch.no_grad 쓰기."""
import torch

# ========================================================================
# 메인
# ========================================================================

def main():

    x = torch.randn(3, requires_grad=True)

    # 평상시의 추적:
    # ----------------
    # x가 requires_grad=True이므로 x가 관여하는 모든 연산도
    # requires_grad=True인 텐서를 만들어낸다. autograd는 grad_fn을 기록하여
    # (아래 y처럼) 잎이 아닌 결과에 대해서도 나중에 경사를 계산할 수 있도록 기록한다.
    y = (x ** 2).sum()
    print("y.requires_grad (expect True):", y.requires_grad)

    # 범위가 한정된 no_grad 문맥:
    # -----------------------
    # with torch.no_grad(): autograd 기록을 일시적으로 끈다.
    # - 이 블록 안에서 만들어지는 새 텐서는 기본적으로 requires_grad=False이다.
    # - 보통은 경사가 필요한 텐서로부터 만들어졌더라도 그렇다.
    # - 최적화기가 매개변수 갱신을 no_grad로 감싸는 이유가 이것이다. 그래야
    #   w -= lr * w.grad가 계산 그래프의 일부가 되지 않는다.
    #
    # 참고:
    # - 이는 x.grad.zero_()와 다르다:
    #     * x.grad.zero_()는 경사 버퍼만 지운다(안전하며 추적되지 않는다).
    #     * x 자체에 대한 갱신(예: x -= lr * x.grad)은 반드시 no_grad() 안에 있어야 한다
    #       그렇지 않으면 autograd가 갱신 연산을 그래프의 일부로 여겨
    #       메모리 누수가 생기고 잎 텐서 규칙이 깨진다.
    with torch.no_grad():
        y_ng = (x ** 2).sum()
        print("Inside no_grad → y_ng.requires_grad (expect False):", y_ng.requires_grad)

    # no_grad() 문맥을 벗어나면:
    # ------------------------------
    # 블록을 벗어나면 autograd 추적이 자동으로 복원된다.
    # 이제 x에 대한 새 연산은 다시 경사를 필요로 하고 grad_fn이 붙는다.
    z = (x + 1).sum()
    print("After block → z.requires_grad (expect True):", z.requires_grad)

if __name__ == "__main__":
    main()
```

**출력:**

```
y.requires_grad (expect True): True
Inside no_grad → y_ng.requires_grad (expect False): False
After block → z.requires_grad (expect True): True
```

## 2. 논의

### 블록의 범위와 텐서의 속성

예제의 `x`는 계속 `requires_grad=True`다. 블록 밖의 제곱합은 추적되지만, `torch.no_grad()` 안에서 계산한 제곱합은 추적되지 않는다. 블록을 나와 계산한 `z`는 다시 추적된다. 따라서 `no_grad()`는 입력의 속성을 영구적으로 바꾸는 기능이 아니다.

일반적인 연산의 기록을 끄는 규칙에는 예외도 있다. `requires_grad` 인수를 받는 팩토리 함수로 새 텐서나 파라미터를 만드는 경우에는 명시한 설정이 적용될 수 있다. 또한 이 문맥 관리자는 역방향 모드의 기록을 제어하며 전방향 자동 미분까지 끄는 기능은 아니다.

### 평가 모드와 함께 사용하기

예측 시에는 보통 `model.eval()`과 `torch.no_grad()`를 함께 사용한다. 전자는 모델 계층의 학습·평가 동작을 전환하고, 후자는 연산 기록을 제어한다. `no_grad()`로 만든 결과를 손실에 연결해도 블록 내부를 거슬러 입력까지 미분할 수는 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
같은 `x`를 블록 전, `no_grad()` 내부, 블록 후에서 제곱하면 각 결과의 `requires_grad`는 어떻게 되는가?

</div>

??? success "연습문제 1 풀이"
    `x.requires_grad=True`이고 다른 설정을 바꾸지 않았다면 일반적인 제곱 연산의 결과는 차례로 `True`, `False`, `True`다. 입력 텐서 자체의 추적 설정은 변하지 않는다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`no_grad()` 안에서 `nn.Parameter`를 새로 만들면 반드시 경사 추적이 꺼지는가?

</div>

??? success "연습문제 2 풀이"
    아니다. `requires_grad` 인수를 지원하는 팩토리 함수나 파라미터 생성에는 예외가 있다. 새 객체에 명시한 설정을 확인해야 한다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
평가 때 `model.eval()`과 `no_grad()`를 함께 사용하는 이유를 설명하라.

</div>

??? success "연습문제 3 풀이"
    `eval()`은 일부 계층의 동작을 평가 방식으로 바꾸고, `no_grad()`는 역전파용 기록을 끈다. 두 기능은 서로를 대신하지 않는다.

## 정리하며

`no_grad()`는 지정한 구간의 연산 기록을 제어한다. 입력의 추적 설정이나 모델의 평가 모드는 별도로 관리한다.
