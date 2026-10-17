# 경사 추적 점검하기

서로 다른 추적 설정을 가진 텐서의 연산과 `no_grad()` 문맥을 비교한다. 결과 텐서의 속성을 직접 확인하는 습관을 익힌다.

## 1. 코드

```python
"""함정과 맛보기 살핌."""
import torch

# ========================================================================
# 메인
# ========================================================================

def main():

    # (a) 추적되는 텐서와 그렇지 않은 텐서를 섞기:
    # 규칙: 입력 중 하나라도 requires_grad=True이면 결과도 경사를 추적한다.
    a = torch.randn(3, requires_grad=True)
    b = torch.randn(3, requires_grad=False)
    c = a + b
    print("a.requires_grad:", a.requires_grad, "| b.requires_grad:", b.requires_grad)
    print("c = a + b → c.requires_grad (expect True):", c.requires_grad)

    # (b) 추론 시점의 문맥:
    # 경사 추적을 일시적으로 끄려면 torch.no_grad()를 쓴다.
    # 역전파 그래프가 만들어지지 않으므로 메모리와 계산을 아낀다.
    m = torch.nn.Linear(4, 2)
    x = torch.randn(8, 4, requires_grad=True)
    with torch.no_grad():
        y = m(x)
    print("Inference y.requires_grad (expect False):", y.requires_grad)

if __name__ == "__main__":
    main()
```

**출력:**

```
a.requires_grad: True | b.requires_grad: False
c = a + b → c.requires_grad (expect True): True
Inference y.requires_grad (expect False): False
```

## 2. 논의

### 입력 중 하나만 경사를 요구하는 경우

일반적인 미분 가능 연산에서는 입력 중 하나라도 경사를 요구하고 기록이 켜져 있으면 결과도 경사를 추적한다. 예제의 `c = a + b`에서 `b.requires_grad=False`여도 `a`에 대한 미분 경로는 유지된다. 다만 정수형 연산이나 명시적 `detach()`처럼 미분 경로를 만들지 않는 경우까지 이 규칙으로 일반화할 수는 없다.

### 문맥이 연산 기록을 제어하는 경우

선형 계층의 파라미터와 입력이 경사를 요구하더라도 `torch.no_grad()` 안에서 계산한 출력은 일반적으로 추적되지 않는다. 이 설정은 파라미터나 입력의 `requires_grad`를 바꾸지 않는다.

예상과 달리 경사가 없으면 로스의 `requires_grad`와 `grad_fn`, 대상 텐서의 `is_leaf`, 중간의 `detach()`나 NumPy 변환 여부를 차례로 점검한다. 비잎 텐서의 `.grad`가 `None`인 것만으로 미분 경로가 끊겼다고 판단하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
`a.requires_grad=True`, `b.requires_grad=False`일 때 `a+b`의 결과는 경사를 추적하는가?

</div>

??? success "풀이"
    일반적인 추적 문맥에서는 `a`에 대한 미분 경로를 기록하므로 결과도 경사를 추적한다. `b`에는 기본적으로 경사를 저장하지 않는다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
비잎 텐서의 `.grad=None`을 보고 그래프가 끊겼다고 판단해도 되는가?

</div>

??? success "풀이"
    그렇지 않다. 비잎 텐서는 기본적으로 경사를 보관하지 않는다. `grad_fn`과 추적 경로를 점검하고, 중간 경사가 필요하면 역전파 전에 `retain_grad()`를 호출한다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
예상과 달리 로스의 `requires_grad`가 `False`라면 어떤 경로를 점검할 것인가?

</div>

??? success "풀이"
    `no_grad()` 구간, `detach()`, NumPy 변환, 텐서로 다시 감싸는 연산, 모든 입력의 추적 설정을 확인한다. 값을 계산한 것과 원래 미분 경로가 유지된 것은 별개다.

## 정리하며

경사 추적은 입력 설정과 실행 문맥을 함께 따른다. 값이 계산되는지, 그래프가 기록되는지, 경사가 보관되는지를 나누어 점검한다.
