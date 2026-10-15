# 벡터 출력과 경사 인수

벡터 출력에 경사 인수 `v`를 전달해 가중합의 미분을 계산한다. `.backward(v)`가 전체 야코비안을 반환하는 연산은 아니라는 점을 이해한다.

## 1. 코드

```python
"""v를 주는 벡터 역전파."""
import torch

# ========================================================================
# 메인
# ========================================================================

def main():

    torch.manual_seed(0)

    # 순전파: x ∈ R^3 → y ∈ R^3 (스칼라가 아닌 벡터 출력).
    x = torch.randn(3, requires_grad=True)
    y = torch.sin(x)  # elementwise sine, shape [3]
    print("x:", x)
    print("y = sin(x):", y, "| shape:", tuple(y.shape))

    # 인수 없이 .backward()를 호출하려면 autograd는 스칼라 손실을 요구한다.
    # y가 스칼라가 아니면 PyTorch는 v^T J를 만들기 위해 명시적인 "벡터" v를 필요로 한다.
    try:
        y.backward()
    except RuntimeError as e:
        print("As expected, calling y.backward() fails for non-scalar y:")
        print("  ", e)

    # y와 같은 모양의 v를 넘긴다. autograd는 다음을 계산한다:
    #   x.grad = J^T v이며 J = ∂y/∂x이다.
    # 이것이 벡터-야코비 곱(VJP)이다.
    v = torch.tensor([0.1, 1.0, 0.01], dtype=torch.float32)
    y.backward(v)
    print("Chosen v:", v)
    print("x.grad (v^T * J):", x.grad)

if __name__ == "__main__":
    main()
```

**출력:**

```
x: tensor([ 1.5410, -0.2934, -2.1788], requires_grad=True)
y = sin(x): tensor([ 0.9996, -0.2892, -0.8208], grad_fn=<SinBackward0>) | shape: (3,)
As expected, calling y.backward() fails for non-scalar y:
   grad can be implicitly created only for scalar outputs
Chosen v: tensor([0.1000, 1.0000, 0.0100])
x.grad (v^T * J): tensor([ 0.0030,  0.9573, -0.0057])
```

## 2. 논의

### 경사 인수가 정하는 스칼라 목적

$y=f(x)$의 야코비안을 $J$라 하고 고정된 벡터 $v$를 선택하면, `y.backward(v)`는 $v^\top y$의 입력 경사를 계산한다. 행벡터 표기로는 $v^\top J$, 열벡터 표기로는 $J^\top v$이며 같은 성분을 표현한다. 결과는 입력 텐서의 모양에 맞춰 `.grad`에 누적된다.

예제는 출력 성분에 서로 다른 가중치를 준다. `v`를 모두 1로 두면 출력 합의 경사이고, 원소 수로 나눈 1로 두면 출력 평균의 경사다. 여러 원소를 가진 출력에는 어떤 조합을 미분할지 지정해야 한다. 원소가 하나인 출력은 모양이 `()`가 아니어도 경사 인수를 생략할 수 있다.

### 전체 야코비안과 구분하기

한 번의 VJP는 야코비안의 행들을 가중합한 결과다. 각 출력 성분의 편미분을 모두 알고 싶다면 별도의 야코비안 계산이 필요하다. `v`는 출력과 같은 모양이어야 하며, 이를 자동으로 브로드캐스팅하는 가중치 인수로 생각하면 안 된다.

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
벡터 출력에서 `v`를 모두 1로 주는 경우와 원소 수로 나눈 1로 주는 경우의 차이는 무엇인가?

</div>

??? success "연습문제 2 풀이"
    모두 1이면 출력 합의 입력 경사를, 원소 수로 나눈 값이면 출력 평균의 입력 경사를 계산한다. 두 경우 모두 전체 야코비안이 아니라 지정한 출력 가중합의 경사를 구한다.

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

`.backward(v)`의 `v`는 출력 쪽에서 전달되는 경사다. 어떤 스칼라 목적의 미분을 구하는지 먼저 정하면 VJP의 의미가 분명해진다.
