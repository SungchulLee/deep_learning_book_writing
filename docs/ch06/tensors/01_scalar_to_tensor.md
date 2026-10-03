# 스칼라를 텐서로

값 하나를 텐서로 감싸는 데에도 고를 것이 둘 있다. **계수**(rank)와 **자료형**(dtype)이다. 둘 다 적지 않으면 PyTorch가 대신 정하는데, 그 규칙이 함수마다 다르다. 이 쪽은 같은 수 42를 네 가지 방법으로 감싸면서 무엇이 달라지는지 본다.

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
    # 주의: torch.tensor(42)와 **같지 않다.** torch.tensor는 파이썬 값의 꼴을
    # 보고 int64로 미루지만, scalar_tensor는 값을 보지 않고 기본 실수형
    # (float32)으로 만든다. 아래 출력에서 t1은 tensor(42), t4는 tensor(42.)이다.

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
        _ = bad.item()             # 원소가 2개라 스칼라로 바꿀 수 없다
    except RuntimeError as e:
        # PyTorch는 ValueError가 아니라 RuntimeError를 낸다.
        # ValueError만 잡으면 예외가 그대로 빠져나가 코드가 멈춘다
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

**출력:**

```
t = tensor(42)
t.shape = torch.Size([])
t.dtype = torch.int64
t.requires_grad = False

t = tensor(42.)
t.shape = torch.Size([])
t.dtype = torch.float32
t.requires_grad = False

t = tensor([42])
t.shape = torch.Size([1])
t.dtype = torch.int64
t.requires_grad = False

t = tensor(42.)
t.shape = torch.Size([])
t.dtype = torch.float32
t.requires_grad = False

t = tensor(3.1400)
t.shape = torch.Size([])
t.dtype = torch.float32
t.requires_grad = False

t = tensor(10)
t.shape = torch.Size([])
t.dtype = torch.int64
t.requires_grad = False

item() on multi-element tensor → a Tensor with 2 elements cannot be converted to Scalar 

t = tensor(7.7000)
t.shape = torch.Size([])
t.dtype = torch.float32
t.requires_grad = False

t = tensor(5., requires_grad=True)
t.shape = torch.Size([])
t.dtype = torch.float32
t.requires_grad = True

t8: 5.0 requires_grad: True
t8.grad (expected 5.0): 5.0 
```

## 2. 논의

**계수 0과 계수 1은 다르다.** `torch.tensor(42)`의 모양은 `torch.Size([])`이고 `torch.tensor([42])`의 모양은 `torch.Size([1])`이다. 원소 개수는 둘 다 하나지만 앞의 것은 수이고 뒤의 것은 길이 1인 벡터다. 대괄호 하나가 계수를 하나 올린다. 뒤에 나오는 브로드캐스팅과 축약은 이 차이를 보고 움직이므로, 여기서 헷갈리면 거기서 모양이 어긋난다.

**자료형을 미루는 규칙은 함수마다 다르다.** 위 출력에서 같은 42가 두 가지 자료형으로 나왔다.

| 만드는 법 | 42를 주면 | 7.7을 주면 | 무엇을 보고 정하나 |
|---|---|---|---|
| `torch.tensor(…)` | `int64` | `float32` | **파이썬 값의 꼴** |
| `torch.scalar_tensor(…)` | `float32` | `float32` | 값을 보지 않는다 — 늘 기본 실수형 |
| `torch.full((), …)` | `int64` | `float32` | **파이썬 값의 꼴** |

`scalar_tensor`만 입력을 보지 않는다. 정수를 넣었는데 실수가 나오므로, 정수 색인이나 정수 나눗셈을 기대한 자리에 쓰면 조용히 어긋난다.

**정수 텐서는 경사를 지닐 수 없다.** 자동 미분은 미분이 정의되는 자료형, 곧 실수와 복소수에서만 돌아간다. 그래서 `requires_grad=True`는 `float32` 같은 실수형에만 붙는다. 이것이 자료형을 미루는 규칙을 그냥 넘길 수 없는 까닭이다 — `torch.tensor(42)`로 만든 잎에는 경사를 켤 수 없고, 그 사실은 모양이 아니라 자료형에 적혀 있다.

**`item()`은 계수가 아니라 원소 개수를 본다.** `numel() == 1`이면 계수가 0이든 1이든 2이든 통한다. 원소가 둘 이상이면 `RuntimeError`다 — `ValueError`가 아니므로 `except ValueError`로는 잡히지 않는다. 위 코드가 `RuntimeError`를 잡는 까닭이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`torch.tensor(5)`, `torch.tensor([5])`, `torch.tensor([[5]])` 셋을 만들어 각각의 `.shape`, `.dim()`, `.numel()`을 찍어라. 원소 개수가 모두 같은데도 서로 다른 텐서인 까닭을 한 문장으로 적고, 셋 가운데 `.item()`이 되는 것이 몇 개인지 답하라.

</div>

??? success "연습문제 1 풀이"
    ```python
    import torch

    for t in [torch.tensor(5), torch.tensor([5]), torch.tensor([[5]])]:
        print(tuple(t.shape), t.dim(), t.numel(), t.item())
    ```

    ```
    () 0 1 5
    (1,) 1 1 5
    (1, 1) 2 1 5
    ```

    원소 개수는 셋 다 1이지만 **계수가 0, 1, 2로 다르다.** 대괄호 한 겹이 차원 하나를
    더한다. 곧 "값이 몇 개인가"와 "그 값들이 몇 차원으로 놓여 있는가"는 따로
    적히는 정보다.

    `.item()`은 셋 다 된다. `numel() == 1`만 보고 계수는 보지 않기 때문이다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
아래 다섯 가지의 `dtype`을 **찍어 보기 전에** 적어 보고, 그 다음 실제로 확인하라. 틀린 것이 있으면 어떤 규칙을 잘못 짚었는지 밝혀라.

```python
torch.tensor(42)         torch.tensor(42.0)      torch.scalar_tensor(42)
torch.full((), 7)        torch.full((), 7.7)
```

</div>

??? success "연습문제 2 풀이"
    ```python
    import torch

    print(torch.tensor(42).dtype)
    print(torch.tensor(42.0).dtype)
    print(torch.scalar_tensor(42).dtype)
    print(torch.full((), 7).dtype)
    print(torch.full((), 7.7).dtype)
    ```

    ```
    torch.int64
    torch.float32
    torch.float32
    torch.int64
    torch.float32
    ```

    헷갈리는 것은 셋째뿐이다. `torch.tensor`와 `torch.full`은 **파이썬 값이 정수인지
    실수인지 보고** 미루므로 42는 `int64`, 7.7은 `float32`가 된다. 그런데
    `torch.scalar_tensor`는 값을 보지 않고 늘 기본 실수형을 쓴다. 그래서 정수 42를
    주어도 `float32`가 나온다.

    실수를 부르는 자리는 정수를 기대한 곳이다. 색인이나 정수 나눗셈에 쓰려고
    `scalar_tensor`로 수를 만들면 조용히 실수가 되어 있다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
`torch.tensor(42, requires_grad=True)`를 실행하면 무엇이 일어나는가? 오류가 난다면 메시지를 그대로 옮기고, 42라는 값은 그대로 두면서 고치는 방법을 **두 가지** 제시하라. 그리고 이 오류가 모양이 아니라 자료형의 문제인 까닭을 설명하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    import torch

    try:
        torch.tensor(42, requires_grad=True)
    except RuntimeError as e:
        print(type(e).__name__, "-", e)
    ```

    ```
    RuntimeError - Only Tensors of floating point and complex dtype can require gradients
    ```

    고치는 두 가지 방법은 모두 자료형을 실수로 바꾸는 것이다.

    ```python
    a = torch.tensor(42.0, requires_grad=True)                      # 값을 실수로 적는다
    b = torch.tensor(42, dtype=torch.float32, requires_grad=True)   # dtype을 밝힌다
    print(a.dtype, a.requires_grad)   # torch.float32 True
    print(b.dtype, b.requires_grad)   # torch.float32 True
    ```

    모양의 문제가 아닌 까닭은 이렇다. 경사는 **미분**이고, 미분은 값을 조금
    움직였을 때의 변화율이다. `int64`에는 "조금"이 없다 — 42와 43 사이에 값이
    없으므로 도함수를 정의할 자리가 없다. 그래서 PyTorch는 계수가 0이든 100이든
    상관하지 않고 자료형만 보고 거절한다. 같은 이유로 `.item()`이 돌려주는 파이썬
    `float`에는 `grad_fn`이 없다. 텐서를 벗어난 순간 그래프에서 떨어져 나온다.

## 정리하며

값 하나를 텐서로 감쌀 때 PyTorch가 대신 정하는 것이 둘이다.

- **계수** — `torch.tensor(42)`는 계수 0, `torch.tensor([42])`는 계수 1이다. 대괄호 한 겹이 차원 하나다. 원소 개수가 같다고 같은 텐서가 아니다.
- **자료형** — `torch.tensor`와 `torch.full`은 파이썬 값이 정수인지 실수인지 보고 미룬다. `torch.scalar_tensor`는 값을 보지 않고 늘 기본 실수형을 쓴다.

이 둘이 뒤에서 각각 발목을 잡는다. 계수는 브로드캐스팅과 축약에서 모양을 어긋나게 하고, 자료형은 경사에서 막는다 — 정수 텐서에는 `requires_grad=True`를 붙일 수 없다. 미분할 자리가 없기 때문이다.

`.item()`은 셋 중 어느 것도 보지 않는다. `numel() == 1`이기만 하면 계수와 무관하게 통하고, 원소가 둘 이상이면 `ValueError`가 아니라 `RuntimeError`를 낸다.
