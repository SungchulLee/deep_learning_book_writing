# PyTorch 기초

PyTorch는 깊은 배움 연구에서 으뜸가는 얼거리이며 실전 펼치기에도 점점 많이 쓰인다. 이 길잡이는 바탕 벽돌인 텐서 만들기, 셈하기, 꼴 바꾸기, GPU로 빠르게 하기, 저절로 미분하기를 들여오고, 기울기 내려가기로 익히는 온전한 선형 회귀 보기로 맺는다. PyTorch에서 주성분 분석, 자기 부호기, 어떤 신경망 얼개든 다루기 앞서 이 밑감을 익혀야 한다.

## 1. 코드

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

**출력:**

```
dy/dx at x=2: tensor([14.])
Learned w=2.0726, b=0.5421
```

## 2. 논의

PyTorch의 텐서는 NumPy 배열처럼 움직이되 결정적인 것 둘이 더 있다. 곧 셈을 빠르게 하려 GPU에 놓을 수 있고, 저절로 미분하려 연산을 좇을 수 있다. `requires_grad=True` 깃발은 앞먹임 동안 셈 그래프를 세우라고 PyTorch에 이르며, 스칼라 손실에 `.backward()`을 부르면 그 그래프를 거꾸로 훑어 좇던 매개변수마다의 기울기를 셈한다.

선형 회귀 보기는 깊은 배움 전체에서 쓰이는 고갱이 익히기 되풀이를 보여 준다. 곧 앞먹임(어림 셈하기), 손실 셈하기(어긋남 재기), 뒤먹임(기울기 셈하기), 매개변수 새로 고침(무게 고치기)이다. 새로 고치는 걸음에서 `torch.no_grad()` 맥락 다루개가 꼭 필요하다. 그러지 않으면 PyTorch가 새로 고침 연산까지 셈 그래프에 넣어 기억 공간을 낭비하고 다음 되풀이에서 기울기가 틀리게 된다.

텐서를 CPU와 GPU 사이에서 옮길 때는 `.to('cuda')`이나 `.cuda()`을 쓰며, 한 셈에 드는 모든 것이 같은 기기에 있어야 한다. 이 길잡이 같은 작은 문제에는 CPU 셈으로 넉넉하지만, 큰 자료 묶음의 주성분 분석 행렬 연산과 신경망 익히기에서는 GPU로 빠르게 하는 것이 결정적이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
3x3 아무 텐서를 만들고 `torch.det`과 `torch.linalg.eig`으로 행렬식과 고윳값을 셈해, 행렬식이 고윳값의 곱과 같은지 확인하라.

</div>

??? success "연습문제 1 풀이"
    ```python
    M = torch.randn(3, 3)
    det = torch.det(M)
    eigvals, _ = torch.linalg.eig(M)
    product = eigvals.prod()
    print(f"det(M) = {det.item():.4f}")
    print(f"Product of eigenvalues = {product.real.item():.4f}")
    print(f"Match: {torch.allclose(det, product.real, atol=1e-4)}")
    ```
    정의상 행렬식은 고윳값의 곱과 같다. `torch.linalg.eig`은 복소 고윳값을 돌려주므로 그 곱의 실수 부분을 취한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
손으로 매개변수를 고치는 대신 `torch.optim.SGD`을 쓰도록 선형 회귀를 고쳐라. 50번 돈 뒤의 마지막 손실을 견주어 같음을 확인하라.

</div>

??? success "연습문제 2 풀이"
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
    관성이 없는 `torch.optim.SGD`은 손으로 하는 것과 같은 $w \leftarrow w - \eta \nabla_w L$ 새로 고침을 하므로 결과가 똑같다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
저절로 미분하기로 점 $(x, y) = (1, 2)$에서 $f(x, y) = x^2 y + y^3$의 기울기를 셈하라. 편미분을 손으로 셈해 결과를 확인하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    x = torch.tensor(1.0, requires_grad=True)
    y = torch.tensor(2.0, requires_grad=True)
    f = x**2 * y + y**3
    f.backward()
    print(f"df/dx = {x.grad.item()}")  # 2*x*y = 2*1*2 = 4
    print(f"df/dy = {y.grad.item()}")  # x^2 + 3*y^2 = 1 + 12 = 13
    ```
    손으로 셈하면 $\partial f/\partial x = 2xy = 4$, $\partial f/\partial y = x^2 + 3y^2 = 13$이며 저절로 미분한 결과와 정확히 맞는다.


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
`torch.tensor`와 `torch.from_numpy`는 어떻게 다른가?

</div>

??? success "연습문제 4 풀이"
    기억 장치를 나누어 쓰는지가 다르다.

    | | 자료를 베끼는가 | 원본과 이어져 있는가 |
    |---|---|---|
    | `torch.tensor(a)` | 베낀다 | 아니다 |
    | `torch.from_numpy(a)` | 안 베낀다 | **그렇다** |

    ```python
    a = np.ones(3)
    t = torch.from_numpy(a)
    a[0] = 99
    print(t)        # tensor([99., 1., 1.])  같이 바뀐다
    ```

    이 성질이 빠르기에는 좋고 버그에는 나쁘다. 큰 배열을 베끼지 않아 값싸지만, 원본을
    고치면 텐서가 조용히 바뀐다.

    자료형도 다르다. `from_numpy`는 NumPy의 자료형을 그대로 물려받으므로 `float64`가
    되기 쉽다. PyTorch의 기본은 `float32`이고 신경망도 그것을 쓰므로, 자료형이 섞여
    오류가 나는 일이 잦다.

    ```python
    t = torch.from_numpy(a).float()      # 대개 이렇게 맞춰 준다
    ```

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span>
`view`와 `reshape`의 차이는 무엇인가?

</div>

??? success "연습문제 5 풀이"
    `view`는 기억 장치가 이어져 있을 때만 되고, `reshape`은 필요하면 베껴서 해 준다.

    ```python
    t = torch.arange(12).reshape(3, 4)
    t.t().view(12)        # 오류. 전치해서 이어져 있지 않다
    t.t().reshape(12)     # 된다. 알아서 베낀다
    ```

    그래서 `reshape`이 늘 듣고 `view`는 실패할 수 있다. 대신 `view`는 **절대 베끼지
    않음이 보장**되므로, 값싼 것이 확실하다.

    실무에서는 `reshape`을 쓰는 것이 편하다. `view`를 쓰다가 `contiguous()`를 불러야
    하는 상황을 만나는 일이 잦다.

    ```python
    t.t().contiguous().view(12)     # view 를 고집하려면 이렇게
    ```

    주성분 분석에서 그림을 펼 때 이 일이 나온다. `(N, 1, 28, 28)`을 `(N, 784)`로 만들
    때 `reshape(N, -1)`이나 `flatten(1)`을 쓰면 안전하다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
`requires_grad`와 `torch.no_grad()`는 각각 무엇을 하는가?

</div>

??? success "연습문제 6 풀이"
    앞은 텐서에 대한 것이고 뒤는 구역에 대한 것이다.

    ```python
    x = torch.randn(3, requires_grad=True)   # 이 텐서의 기울기를 모으겠다
    with torch.no_grad():                    # 이 구역에서는 아무것도 모으지 않는다
        y = x * 2
    ```

    `requires_grad=True`인 텐서에서 나온 결과는 셈하기 그림을 들고 다니므로 기억 장치를
    더 쓴다. `no_grad()`는 그 그림을 만들지 않게 한다.

    쓰는 자리가 또렷하다.

    | 자리 | 무엇을 쓰는가 |
    |---|---|
    | 배울 매개변수 | `requires_grad=True` (`nn.Parameter`가 알아서) |
    | 값매김, 추론 | `with torch.no_grad():` |
    | 기울기를 빼고 값만 | `.detach()` |

    빠뜨리면 나는 증상이 다르다. 값매김에서 `no_grad()`를 빠뜨리면 오류 없이 **기억
    장치만 낭비**하고 느려진다. 자료 텐서에 `requires_grad`를 켜 두면 쓸데없는 기울기가
    모인다.

    주성분 분석은 쪼개기로 한 번에 풀리므로 기울기가 필요 없다. 그래서 `no_grad()`
    안에서 하는 것이 맞다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
CPU와 GPU 사이에서 텐서를 옮길 때 무엇을 조심해야 하는가?

</div>

??? success "연습문제 7 풀이"
    **섞어 쓰면 오류가 난다.** 한 연산에 들어가는 텐서들이 같은 장치에 있어야 한다.

    ```python
    x = torch.randn(3).to('mps')
    y = torch.randn(3)
    x + y      # RuntimeError: 장치가 다르다
    ```

    그리고 옮기는 것 자체가 비싸다. 자료를 주고받는 통로가 셈하기보다 훨씬 느리므로,
    작은 텐서를 자주 옮기면 GPU를 쓰는 이득이 사라진다.

    | 요령 | 왜 |
    |---|---|
    | 자료를 한 번에 옮긴다 | 옮기는 횟수를 줄인다 |
    | `.item()`을 반복문 안에서 피한다 | 값을 꺼내려면 기다려야 한다 |
    | 그림을 그릴 때만 `.cpu()` | 필요할 때만 되옮긴다 |

    두 번째가 놓치기 쉽다. 손실을 기록하려고 매 걸음 `.item()`을 부르면 GPU가 끝나기를
    기다리게 되어 느려진다. 텐서로 모아 두고 에포크 끝에 한 번 꺼내는 편이 낫다.

    MNIST 주성분 분석에서는 자료가 176 MB라 한 번 올려 두면 되고, 특잇값 쪼개기는
    GPU에서 몇 배 빠르다.

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
자동 미분이 주성분 분석에 필요한가?

</div>

??? success "연습문제 8 풀이"
    필요 없다. 주성분 분석은 **닫힌 꼴로 풀리는 문제**이므로 쪼개기 한 번이면 정확한
    답이 나온다.

    반복 최적화로 풀면 느리고 덜 정확하기만 하다. 실제로 선형 자기 부호기를 기울기로
    익혀 보면 주성분 분석의 어긋남에 위에서 다가가되 정확히 닿지 않고, 얻는 기저도
    직교하지 않는다([기본 연습문제 6](pca_fundamentals.md)).

    그러면 왜 이 쪽이 PyTorch를 가르치는가. 주성분 분석 **다음에 오는 것들** 때문이다.

    | 무엇 | 닫힌 꼴이 있는가 |
    |---|---|
    | 주성분 분석 | 있다 |
    | 알맹이 주성분 분석 | 있다 |
    | 계량 MDS | **없다** |
    | 자기 부호기, t-SNE | 없다 |

    셋째 줄부터 기울기가 필요하다. 이 장에서 계량 MDS를 PyTorch로 짜는 것이 그
    첫 자리다([MDS](../manifold/mds.md)).

    그리고 주성분 분석을 미분 가능한 물길의 한 조각으로 쓸 때는 자동 미분이 필요하다.
    `torch.linalg.svd`가 미분 가능하므로 주성분 분석을 지나 기울기를 흘릴 수 있다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
`torch.linalg.svd`를 지나 기울기를 흘릴 때 무엇을 조심해야 하는가?

</div>

??? success "연습문제 9 풀이"
    **특잇값이 겹치면 미분이 정의되지 않는다.**

    특이 벡터에 대한 미분식에 $1/(s_i^2 - s_j^2)$ 꼴의 항이 들어가므로, 두 특잇값이
    같으면 0으로 나누게 된다. 거의 같으면 폭발한다.

    실제로 나타나는 증상이 이렇다.

    - 손실이 `nan`이 된다
    - 기울기가 갑자기 아주 커진다
    - 대칭성이 있는 자료에서 특히 자주 일어난다

    이것이 [유도 연습문제 5](pca_derivation.md)에서 말한 겹침 문제의 미분 버전이다.
    겹치면 성분 개개가 정해지지 않으니 미분이 정의되지 않는 것이 당연하다.

    다룰 방법이 몇 가지다.

    - **특잇값만 쓴다.** $s$에 대한 미분은 겹쳐도 괜찮다. 특이 벡터를 쓰지 않으면 안전하다
    - **부분 공간만 쓴다.** 개별 성분이 아니라 사영 행렬 $V_k V_k^\top$을 쓰면 겹침에
      딸리지 않는다
    - **조금 흔든다.** 대각선에 작은 값을 더해 겹침을 깬다

    둘째가 가장 깔끔하다. 우리가 정말 쓰는 것이 대개 부분 공간이지 개별 축이 아니기
    때문이다. 주성분 분석의 답에서 정해지는 것도 부분 공간뿐이었다.

    아예 쪼개기를 피하는 길도 있다. 사영을 배울 파라미터로 두고 직교성을 벌점이나
    다시 매개변수화로 강제하면, 쪼개기 없이 미분 가능한 차원 줄이기가 된다.

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
이 페이지의 선형 회귀 보기에서 `optimizer.zero_grad()`를 빠뜨리면 어떻게 되는가?

</div>

??? success "연습문제 10 풀이"
    기울기가 **쌓인다.** PyTorch는 `.backward()`를 부를 때마다 기존 기울기에 더하기
    때문이다.

    그래서 걸음마다 기울기가 커져 걸음이 점점 커지고, 대개 발산한다. 손실이 `nan`으로
    가는 흔한 원인이다.

    쌓이는 것이 기본 동작인 까닭은 그것이 필요한 때가 있기 때문이다. 묶음을 여러 조각으로
    나누어 기울기를 모으고 한 번에 갱신하는(gradient accumulation) 방식이 그렇다. 기억
    장치가 모자라 큰 묶음을 한 번에 못 넣을 때 쓴다.

    ```python
    for i, batch in enumerate(loader):
        loss = criterion(model(batch)) / accum
        loss.backward()                      # 쌓는다
        if (i + 1) % accum == 0:
            optimizer.step(); optimizer.zero_grad()
    ```

    순서도 알아 두면 좋다. `zero_grad` → `forward` → `backward` → `step`이며,
    `zero_grad`를 `step` 바로 뒤에 두어도 같다. 빠뜨리지만 않으면 된다.

## 정리하며

**다룬 것** — PyTorch 기초

PyTorch의 텐서는 NumPy 배열처럼 움직이되 결정적인 것 둘이 더 있다.

앞의 연습문제 3개로 스스로 따져 볼 수 있다.
