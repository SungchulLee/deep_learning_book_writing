# PCA를 위한 PyTorch 기초

이 절에서는 텐서 연산과 자동 미분을 살펴보고 선형 회귀의 파라미터를 직접 갱신하는 과정을 읽는다. PCA의 분해 자체에는 경사 학습이 필요하지 않지만, 이후의 오토인코더나 미분 가능한 전처리를 이해하는 데 이 개념이 쓰인다.

## 1. 예제 코드와 출력

```python
"""PyTorch 기본."""
import torch
import numpy as np
import matplotlib.pyplot as plt

# === 텐서 만들기 ========================================================
tensor_from_list = torch.tensor([1, 2, 3, 4, 5])
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
zeros = torch.zeros(2, 3)
random = torch.randn(2, 2)
identity = torch.eye(3)

# === 기본 연산 ========================================================
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
mat_product = A @ B

# === 저절로 미분하기 ================================================================
x = torch.tensor([2.0], requires_grad=True)
y = 3 * x**2 + 2 * x + 1
y.backward()
print(f"dy/dx at x=2: {x.grad}")  # 14.0

# === 선형 회귀 =======================================================
torch.manual_seed(42)
true_w, true_b = 2.0, 1.0
x_data = torch.linspace(0, 10, 100).unsqueeze(1)
y_data = true_w * x_data + true_b + 0.5 * torch.randn(100, 1)

w = torch.randn(1, 1, requires_grad=True)
b_param = torch.zeros(1, requires_grad=True)
learning_rate = 0.01

losses = []
for epoch in range(50):
    y_pred = x_data @ w + b_param
    loss = ((y_pred - y_data) ** 2).mean()
    losses.append(loss.item())
    loss.backward()
    with torch.no_grad():
        w -= learning_rate * w.grad
        b_param -= learning_rate * b_param.grad
    w.grad.zero_()
    b_param.grad.zero_()

print(f"Learned w={w.item():.4f}, b={b_param.item():.4f}")

if __name__ == "__main__":
    pass
```

```
dy/dx at x=2: tensor([14.])
Learned w=2.0726, b=0.5421
```

## 2. 텐서와 행렬 연산

텐서에는 모양, dtype, 장치와 메모리 배치가 있다. 정수 목록의 `torch.tensor`는 보통 정수 dtype을 만들고 실수 텐서는 기본 실수 dtype을 따른다. 코드의 `A@B`는 행렬곱이며 같은 모양의 `A*B`는 원소별 곱이다. 브로드캐스팅이 허용하는 모양과 의도한 계산이 같은지 확인한다.

NumPy 배열을 `torch.from_numpy`로 변환하면 지원되는 CPU 배열의 메모리를 공유한다. 한쪽 변경이 다른 쪽에 나타날 수 있다. `torch.tensor`는 데이터를 복사한다. dtype 변환이나 장치 이동이 추가되면 공유 관계가 달라질 수 있으므로 필요한 경우 명시적인 복사를 사용한다.

## 3. 자동 미분과 파라미터 갱신

x=2에서 y=3x²+2x+1의 미분은 6x+2=14다. `requires_grad=True`인 실수 텐서로 추적한 연산에 `backward()`를 호출하면 그래디언트를 계산한다. 일반적으로 `.grad`는 그래디언트가 필요한 리프 텐서에 누적된다. 모든 중간 텐서가 자동으로 `.grad`를 보관하는 것은 아니다.

회귀 모델은 ŷ=xw+b이며 로스는 표본별 제곱 오차의 평균이다. 데이터와 파라미터 모양은 각각 100×1과 1×1, 바이어스는 길이 1이다. 학습 루프는 순전파, 로스, 역전파, 갱신과 그래디언트 초기화로 이루어진다.

수동 갱신을 `no_grad()` 안에서 수행해 갱신 연산을 새 그래프에 기록하지 않는다. 그래디언트가 필요한 리프 텐서에 이런 제자리 연산을 추적 상태로 수행하면 단순한 메모리 낭비를 넘어 오류가 발생할 수 있다. 기존 그래디언트를 지우지 않으면 다음 역전파가 더해진다.

## 4. 출력과 수렴의 의미

회귀 데이터에는 잡음이 있고 50회의 갱신만 하므로 w와 b가 생성 모수 2와 1에 정확히 같아질 필요는 없다. 출력은 한 실행의 예시다. 학습 로스 감소와 테스트 일반화 성능은 별개이며, 다른 초기값의 SGD 실행과 수동 갱신이 같은 최종값을 내는 것도 보장되지 않는다.

파일의 실행문 대부분은 최상위에 있다. 마지막 `if __name__ == '__main__': pass`가 import 시 앞의 데이터 생성과 학습을 막는 것은 아니다.

## 5. 장치와 텐서 모양

선형대수 연산에서는 보통 입력 텐서의 장치와 dtype이 맞아야 한다. 가속기 사용 가능 여부와 연산 지원을 확인하고 이동 결과를 변수에 받는다. 작은 계산에서는 전송·동기화 비용이 가속 이득보다 클 수 있다. GPU의 `.item()`이나 CPU 변환을 자주 호출하면 동기화 비용이 생길 수 있다.

`view`는 원소 수와 stride가 요청 모양에 호환될 때 메모리를 공유한다. 연속 텐서는 충분조건이지만 모든 비연속 텐서에서 실패하는 것은 아니다. `reshape`는 가능하면 뷰를 만들고 필요하면 복사한다. 두 경우 모두 원소 수는 맞아야 한다. 정확한 조건은 [PyTorch view 문서](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html)를 참고한다.

## 6. PCA에 자동 미분을 연결할 때

고정 PCA 기저에 입력을 투영하는 것은 선형 행렬곱이므로 입력 그래디언트를 쉽게 계산할 수 있다. 데이터에 따라 기저를 다시 추정하는 경로까지 미분하려면 SVD의 그래디언트를 고려해야 한다.

개별 특이벡터는 반복·근접 특잇값에서 비유일하거나 불안정하다. 선택된 부분공간의 투영은 내부 기저 회전에 불변이지만 선택 경계에 스펙트럼 간격이 필요하고, 이를 일반 SVD로 구현한 역전파가 항상 안정적인 것은 아니다. 특잇값만 필요하면 `svdvals`를 검토하되 최종 로스 자체의 미분 가능성도 확인한다. [PyTorch SVD 문서](https://docs.pytorch.org/docs/stable/generated/torch.linalg.svd.html)에 수치·그래디언트 조건이 설명되어 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
행렬식과 고윳값 곱의 관계를 설명하고 실수 행렬의 복소 고윳값 결과를 어떻게 비교할지 서술하라.

</div>

??? success "풀이"

    정사각 행렬의 행렬식은 대수적 중복도를 포함한 고윳값의 곱이다. 실수 행렬에도 복소 고윳값이 나올 수 있지만 켤레 쌍의 곱 때문에 전체 곱은 이론적으로 실수다. 아래 코드는 실수 부분을 비교하며 실제 수치 검사에서는 허수 잔차와 절대·상대 허용 오차도 고려한다.

    ```python
    M = torch.randn(3, 3)
    det = torch.det(M)
    eigvals, _ = torch.linalg.eig(M)
    product = eigvals.prod()
    print(f"det(M) = {det.item():.4f}")
    print(f"Product of eigenvalues = {product.real.item():.4f}")
    print(f"Match: {torch.allclose(det, product.real, atol=1e-4)}")
    ```


---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
수동 갱신과 관성 없는 SGD가 같은 결과를 내기 위한 조건을 설명하라.

</div>

??? success "풀이"

    같은 데이터, 초기 파라미터, 로스 평균, 러닝레이트와 갱신 횟수에서 두 방법은 w←w−η∇L 규칙이 같다. 아래 예제의 w2는 새 무작위 초기값이므로 기존 실행과 최종값이 반드시 같지는 않다. 저장된 loss는 마지막 갱신 전의 값이다.

    ```python
    w2 = torch.randn(1, 1, requires_grad=True)
    b2 = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.SGD([w2, b2], lr=0.01)
    for epoch in range(50):
        y_pred = x_data @ w2 + b2
        loss = ((y_pred - y_data) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"w={w2.item():.4f}, b={b2.item():.4f}, loss={loss.item():.4f}")
    ```


---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
f(x,y)=x²y+y³의 (1,2)에서의 그래디언트를 구하라.

</div>

??? success "풀이"

    ∂f/∂x=2xy=4, ∂f/∂y=x²+3y²=13이다. 아래 코드는 하나의 스칼라 f에서 두 리프 텐서에 대해 이 그래디언트를 누적한다.

    ```python
    x = torch.tensor(1.0, requires_grad=True)
    y = torch.tensor(2.0, requires_grad=True)
    f = x**2 * y + y**3
    f.backward()
    print(f"df/dx = {x.grad.item()}")  # 2*x*y = 2*1*2 = 4
    print(f"df/dy = {y.grad.item()}")  # x^2 + 3*y^2 = 1 + 12 = 13
    ```


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
torch.tensor와 torch.from_numpy의 복사·dtype 차이를 설명하라.

</div>

??? success "풀이"

    tensor는 입력을 복사하고 from_numpy는 지원되는 NumPy CPU 배열의 저장 공간을 공유한다. from_numpy는 배열 dtype도 유지하므로 float64 배열이면 float64 텐서다. 아래 .float()는 float32로 바꾸며 dtype이 실제로 바뀌면 새 저장 공간을 사용한다. 이미 같은 dtype이면 독립 복사를 보장하지 않는다.

    ```python
    a = np.ones(3)
    t = torch.from_numpy(a)
    a[0] = 99
    print(t)        # tensor([99., 1., 1.])  같이 바뀐다
    ```
    ```python
    t = torch.from_numpy(a).float()      # 대개 이렇게 맞춰 준다
    ```


---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span>
view와 reshape의 차이를 설명하고 아래 전치 텐서 예제가 중간에 실패하는 이유를 서술하라.

</div>

??? success "풀이"

    view는 요청 모양이 기존 stride에 호환되어야 한다. 전치한 4×3 텐서를 그대로 길이 12로 보는 예시는 이 조건을 만족하지 않아 실패하며 같은 블록의 다음 reshape 줄은 그대로 연속 실행되지 않는다. reshape는 필요하면 복사한다. contiguous().view는 호환되는 연속 배치를 먼저 만드는 대안이다.

    ```python
    t = torch.arange(12).reshape(3, 4)
    t.t().view(12)        # 오류. 전치해서 이어져 있지 않다
    t.t().reshape(12)     # 된다. 알아서 베낀다
    ```
    ```python
    t.t().contiguous().view(12)     # view 를 고집하려면 이렇게
    ```


---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
requires_grad, no_grad와 eval의 역할을 구분하라.

</div>

??? success "풀이"

    requires_grad는 텐서의 그래디언트 추적 속성이고 no_grad는 구역 안의 연산 기록을 끈다. eval은 Dropout·BatchNorm 등의 동작 모드를 바꾸며 자동 미분을 끄지 않는다. detach는 그래프에서 분리하지만 저장 공간을 공유할 수 있다. 고정 PCA 계산과 추론에서 추적이 필요 없으면 해당 구역을 no_grad로 둘 수 있다.

    ```python
    x = torch.randn(3, requires_grad=True)   # 이 텐서의 그래디언트를 모으겠다
    with torch.no_grad():                    # 이 구역에서는 아무것도 모으지 않는다
        y = x * 2
    ```


---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
아래 MPS·CPU 텐서 덧셈 예제의 전제와 오류를 설명하라.

</div>

??? success "풀이"

    먼저 MPS가 사용 가능해야 첫 장치 이동이 된다. 사용 가능해도 x는 MPS, y는 CPU여서 일반적인 벡터 덧셈의 장치 조건이 맞지 않는다. 같은 장치로 옮기고 전송 비용을 고려한다. 가속기를 사용한다고 작은 연산이 항상 빨라지는 것은 아니다.

    ```python
    x = torch.randn(3).to('mps')
    y = torch.randn(3)
    x + y      # RuntimeError: 장치가 다르다
    ```


---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
PCA를 계산할 때와 PCA를 포함한 모델을 학습할 때 자동 미분의 필요성을 비교하라.

</div>

??? success "풀이"

    데이터에서 기저를 한 번 구하는 PCA는 분해로 계산할 수 있어 역전파가 필요 없다. 고정 기저의 투영을 지나 입력 그래디언트가 필요하거나 기저 추정 자체가 로스에 연결되면 자동 미분을 사용할 수 있다. MDS도 고전적 고유분해와 stress 최소화가 다르므로 모든 MDS가 경사 학습을 요구하는 것은 아니다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
개별 특이벡터를 사용하는 SVD 역전파에서 스펙트럼 간격이 중요한 이유를 설명하라.

</div>

??? success "풀이"

    벡터 미분에는 특잇값 차이의 역수가 나타나 반복·근접 특잇값에서 정의나 수치 안정성 문제가 생긴다. 투영 행렬은 선택 공간 내부 회전에 불변이지만 경계의 중복은 여전히 문제다. 단순히 공분산에 εI를 더하면 모든 고윳값이 같이 이동하므로 중복을 깨지 못한다. 사용하는 로스와 미분 구현에 맞는 처리가 필요하다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
그래디언트를 초기화하지 않으면 무엇이 일어나는가? 아래 누적 코드의 빠진 조건도 설명하라.

</div>

??? success "풀이"

    backward는 기존 grad에 더한다. 의도하지 않은 누적은 갱신 규칙을 바꾸지만 크기가 매번 커지거나 반드시 발산하는 것은 아니다. 아래 코드는 시작 전 초기화가 필요하고 배치 수가 accum의 배수가 아니면 마지막 묶음에 step이 없다. 크기가 다른 마지막 묶음과 마이크로배치의 로스 평균도 실제 표본 수에 맞춰 처리해야 한다.

    ```python
    for i, batch in enumerate(loader):
        loss = criterion(model(batch)) / accum
        loss.backward()                      # 쌓는다
        if (i + 1) % accum == 0:
            optimizer.step(); optimizer.zero_grad()
    ```


## 정리하며

PyTorch 계산에서는 모양·dtype·장치와 저장 공간 공유를 함께 확인한다. 역전파는 그래디언트를 누적하고 수동 갱신은 그래프 기록 밖에서 수행한다. 고정 PCA 투영과 데이터 의존 SVD의 미분은 조건이 다르며 가속 성능과 수치 안정성은 실제 환경에서 따로 확인해야 한다.
