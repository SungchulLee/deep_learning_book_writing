# 사용자 정의 Autograd 함수 - PyTorch 미분 확장하기

기본 연산을 조합하기 어려운 연산은 순전파와 역전파를 직접 정의할 수 있다. 이 페이지는 제곱과 ReLU를 예로 들어 `torch.autograd.Function`의 역할을 설명한다.

## 1. 코드

```python
"""튜토리얼 20: 맞춤 자동 미분 함수 - PyTorch의 미분 넓히기"""
import torch
# 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
# 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
torch.manual_seed(0)

# ========================================================================
# 메인
# ========================================================================

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

class SquareFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input ** 2
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * 2 * input

class ReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

def main():
    header("1. Custom Function: Square")
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    square_fn = SquareFunction.apply
    y = square_fn(x)
    print(f"x = {x}")
    print(f"y = x^2 = {y}")
    y.sum().backward()
    print(f"x.grad = {x.grad}")
    
    header("2. Custom Function: ReLU")
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    relu_fn = ReLUFunction.apply
    y = relu_fn(x)
    print(f"x = {x}")
    print(f"ReLU(x) = {y}")
    y.sum().backward()
    print(f"x.grad = {x.grad}")
    
    header("3. When to Use Custom Functions")
    print("""
    다음 때에 맞춤 자동 미분 함수를 쓴다.
    1. PyTorch에 없는 새 셈을 짤 때
    2. 남다른 그래디언트 액션이 필요할 때
    3. 어떤 셈의 성능을 다듬을 때
    4. 바깥 꾸러미와 이을 때
    5. 연구: 새 살림 함수를 시험할 때
    """)
    
    header("4. Built-in vs Custom")
    x = torch.randn(5, requires_grad=True)
    built_in = torch.relu(x)
    custom = ReLUFunction.apply(x)
    print(f"Built-in ReLU: {built_in}")
    print(f"Custom ReLU: {custom}")
    print(f"Match? {torch.allclose(built_in, custom)}")

if __name__ == "__main__":
    main()
```

**출력:**

```

======================================================================
1. Custom Function: Square
======================================================================
x = tensor([2., 3.], requires_grad=True)
y = x^2 = tensor([4., 9.], grad_fn=<SquareFunctionBackward>)
x.grad = tensor([4., 6.])

======================================================================
2. Custom Function: ReLU
======================================================================
x = tensor([-2., -1.,  0.,  1.,  2.], requires_grad=True)
ReLU(x) = tensor([0., 0., 0., 1., 2.], grad_fn=<ReLUFunctionBackward>)
x.grad = tensor([0., 0., 1., 1., 1.])

======================================================================
3. When to Use Custom Functions
======================================================================

    다음 때에 맞춤 자동 미분 함수를 쓴다.
    1. PyTorch에 없는 새 셈을 짤 때
    2. 남다른 그래디언트 액션이 필요할 때
    3. 어떤 셈의 성능을 다듬을 때
    4. 바깥 꾸러미와 이을 때
    5. 연구: 새 살림 함수를 시험할 때
    

======================================================================
4. Built-in vs Custom
======================================================================
Built-in ReLU: tensor([1.5410, 0.0000, 0.0000, 0.5684, 0.0000], grad_fn=<ReluBackward0>)
Custom ReLU: tensor([1.5410, 0.0000, 0.0000, 0.5684, 0.0000],
       grad_fn=<ReLUFunctionBackward>)
Match? True
```

## 2. 논의

`forward`는 출력을 계산하고, `backward`는 출력 쪽에서 전달된 그래디언트를 입력 쪽으로 보내는 VJP를 계산한다. 제곱 함수에서는 상류 그래디언트에 $2x$를 곱한다. ReLU는 양수 입력에서 그대로 전달하고 음수 입력에서 0으로 만든다. 0에서의 값은 구현이 정한 관례다.

역전파에 필요한 입력은 `ctx.save_for_backward`로 저장한다. 저장하지 않아도 되는 값까지 남기면 메모리 비용이 늘고, 필요한 값을 생략하면 올바른 미분을 계산할 수 없다. 사용자 정의 미분은 작은 double 정밀도 입력의 `gradcheck`로 검증할 수 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
함수 $f(x) = x^3 - 2x^2 + x$를 생각하자. PyTorch autograd를 사용하여 $f'(3)$을 계산하라.

</div>

??? success "풀이"
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
제곱합처럼 저장된 중간값이 필요한 로스에서 `retain_graph=True` 없이 같은 그래프에 `.backward()`를 두 번 호출하면 오류가 나는 이유를 설명하라. `retain_graph=True`는 메모리 사용량에 어떤 영향을 주는가?

</div>

??? success "풀이"
    기본 역전파는 사용을 마친 저장 텐서를 해제한다. 같은 그래프의 두 번째 역전파에 이 값들이 필요하면 `RuntimeError`가 발생한다. 저장 텐서가 필요 없는 단순 연산은 반복 호출이 가능할 수도 있으므로 모든 그래프에서 반드시 오류가 나는 것은 아니다. 첫 호출의 `retain_graph=True`는 필요한 중간 결과를 보존해 재사용을 가능하게 하지만 그만큼 메모리를 더 오래 사용한다. `.grad` 초기화는 이 보존 여부와 별개의 작업이다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
잎 텐서 `w`를 만들고, 반복마다 같은 식으로 로스를 새로 계산하라. 경사를 초기화하지 않고 `.backward()`를 세 번 호출하며 매번 `w.grad`를 출력하라. 관찰된 값을 설명하라.

</div>

??? success "풀이"
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

기본 연산을 조합하기 어려운 연산은 순전파와 역전파를 직접 정의할 수 있다. 이 페이지는 제곱과 ReLU를 예로 들어 `torch.autograd.Function`의 역할을 설명한다.
