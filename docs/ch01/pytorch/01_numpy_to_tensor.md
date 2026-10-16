# NumPy에서 PyTorch로

---

## 1. 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

- 텐서와 `ndarray`가 같은 점과 다른 점을 말하기
- 배열과 텐서를 서로 오가며, 메모리를 함께 쓰는지 확인하기
- 텐서가 어느 장치에 있는지 알고 옮기기
- 자동 미분이 무엇을 대신해 주는지 한 예로 보이기

---

## 2. 텐서는 장치와 미분을 아는 배열

PyTorch의 `Tensor`는 NumPy의 `ndarray`와 비슷한 배열 인터페이스를 제공하면서 장치 관리와 자동 미분을 지원한다. 다음 식은 기능상의 차이를 요약하는 비유다.

$$\text{Tensor} = \text{ndarray} + \text{장치}(\texttt{device}) + \text{자동 미분}(\texttt{autograd})$$

문법은 거의 그대로 옮겨 간다.

| 하는 일 | NumPy | PyTorch |
|---|---|---|
| 0으로 채우기 | `np.zeros((2,3))` | `torch.zeros(2,3)` |
| 정규 난수 | `np.random.randn(2,3)` | `torch.randn(2,3)` |
| 모양 | `a.shape` | `a.shape` |
| 모양 바꾸기 | `a.reshape(3,2)` | `a.reshape(3,2)` |
| 행렬 곱 | `A @ B` | `A @ B` |
| 축 방향 합 | `a.sum(axis=0)` | `a.sum(dim=0)` |
| 자료형 바꾸기 | `a.astype(np.float32)` | `a.float()` |

이 표의 연산은 문법이 비슷하지만, 두 도구의 모든 동작이 같은 것은 아니다. 자료형, 메모리 공유, 제자리 연산, 미분 지원 여부도 확인해야 한다.

```python
import numpy as np
import torch

# 같은 일을 두 라이브러리로
a_np = np.random.default_rng(0).standard_normal((3, 4))
a_pt = torch.randn(3, 4)

print(f"NumPy  : shape={a_np.shape}, dtype={a_np.dtype}")
print(f"PyTorch: shape={tuple(a_pt.shape)}, dtype={a_pt.dtype}")
print()
print(f"NumPy   합(axis=0): {a_np.sum(axis=0).shape}")
print(f"PyTorch 합(dim=0) : {tuple(a_pt.sum(dim=0).shape)}")
print()
# 기본 자료형이 다르다는 점에 주의한다
print(f"NumPy 기본형  : {a_np.dtype}   <- 배정밀도")
print(f"PyTorch 기본형: {a_pt.dtype} <- 단정밀도, 딥러닝의 표준")
```

**출력:**

```
NumPy  : shape=(3, 4), dtype=float64
PyTorch: shape=(3, 4), dtype=torch.float32

NumPy   합(axis=0): (4,)
PyTorch 합(dim=0) : (4,)

NumPy 기본형  : float64   <- 배정밀도
PyTorch 기본형: torch.float32 <- 단정밀도, 딥러닝의 표준
```

---

## 3. 서로 오가기

```python
import numpy as np
import torch

arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)

# ========================================================
# NumPy -> PyTorch
# ========================================================
# from_numpy는 메모리를 함께 쓴다. 복사가 없으므로 빠르지만,
# 한쪽을 고치면 다른 쪽도 함께 바뀐다.
shared = torch.from_numpy(arr)

# tensor(...)는 복사본을 만든다.
copied = torch.tensor(arr)

arr[0] = 99.0
print(f"원본 배열을 고친 뒤")
print(f"  arr            = {arr}")
print(f"  from_numpy 결과 = {shared}   <- 함께 바뀐다")
print(f"  tensor 결과     = {copied}   <- 그대로다")

# ========================================================
# PyTorch -> NumPy
# ========================================================
t = torch.ones(3)
back = t.numpy()          # 이것도 메모리를 함께 쓴다
t[0] = 5.0
print(f"\n텐서를 고친 뒤 numpy() 결과: {back}")
```

**출력:**

```
원본 배열을 고친 뒤
  arr            = [99.  2.  3.]
  from_numpy 결과 = tensor([99.,  2.,  3.])   <- 함께 바뀐다
  tensor 결과     = tensor([1., 2., 3.])   <- 그대로다

텐서를 고친 뒤 numpy() 결과: [5. 1. 1.]
```

이 예제의 `torch.from_numpy`와 CPU 텐서의 `.numpy()`는 조건이 맞으면 메모리를 공유한다. GPU 텐서는 CPU로 옮겨야 하며, 경사를 추적하는 텐서는 보통 `tensor.detach().cpu().numpy()`로 변환한다. 장치 이동이나 자료형 변환으로 복사가 생기면 메모리 공유 관계가 달라진다.

---

## 4. 장치

텐서는 자기가 어디에 있는지 안다.

```python
import torch

# 이 기계에서 쓸 수 있는 장치를 고르는 표준적인 방식
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

x = torch.randn(2, 3)
print(f"고른 장치     : {device}")
print(f"만든 직후     : {x.device}")

x_gpu = x.to(device)
print(f"옮긴 뒤       : {x_gpu.device}")

# 서로 다른 장치에 있는 텐서끼리는 계산할 수 없다
y = torch.randn(2, 3)
try:
    _ = x_gpu + y
    print("\n같은 장치에 있어 계산이 된다 (CPU만 있는 기계)")
except RuntimeError as e:
    print(f"\n장치가 다르면: {str(e).splitlines()[0]}")

# 되돌리려면 .cpu()
print(f"다시 CPU로    : {x_gpu.cpu().device}")
```

**출력:**

```
고른 장치     : mps
만든 직후     : cpu
옮긴 뒤       : mps:0

장치가 다르면: Expected all tensors to be on the same device, but found at least two devices, mps:0 and cpu!
다시 CPU로    : cpu
```

`Expected all tensors to be on the same device`는 PyTorch를 쓰며 가장 자주 만나는 오류이다. 모델은 GPU에 올려 놓고 데이터는 CPU에 둔 채로 넣으면 이 오류가 난다. **모델과 데이터를 같은 장치로 보내는 것**이 학습 루프의 기본 규율이다.

---

## 5. 자동 미분

여기서부터가 NumPy로는 할 수 없는 일이다. 텐서에 `requires_grad=True`를 붙이면 PyTorch가 그 텐서에 일어난 계산을 기록해 두었다가, 미분을 대신 구해 준다.

```python
import torch

# f(x) = x^3 + 2x 의 미분은 3x^2 + 2 이다.
x = torch.tensor([2.0], requires_grad=True)
f = x ** 3 + 2 * x

f.backward()              # 미분을 계산한다

print(f"x        = {x.item()}")
print(f"f(x)     = {f.item()}      (2^3 + 2*2 = 12)")
print(f"자동 미분 = {x.grad.item()}      <- PyTorch가 구한 값")
print(f"손으로    = {3 * 2.0**2 + 2}      <- 3x^2 + 2 에 x=2")
print(f"일치      : {x.grad.item() == 3 * 2.0**2 + 2}")
```

**출력:**

```
x        = 2.0
f(x)     = 12.0      (2^3 + 2*2 = 12)
자동 미분 = 14.0      <- PyTorch가 구한 값
손으로    = 14.0      <- 3x^2 + 2 에 x=2
일치      : True
```

식이 이렇게 단순하면 손으로 미분해도 된다. 그러나 뉴럴 넷트웍은 레이어가 수십 개이고 파라메터가 수백만 개이다. 각 연산의 미분을 직접 구현하는 것은 가능하지만, 큰 모델의 계산 그래프를 수작업으로 관리하는 일은 복잡하고 오류가 생기기 쉽다.

**자동 미분이 딥러닝을 실용적으로 만든 장치**이며, 이것이 NumPy 대신 PyTorch를 쓰는 가장 큰 까닭이다. 자세한 원리는 [7장의 자동 미분](../../ch06/index.md)에서 다룬다.

---

## 6. 세 줄 요약

```python
import torch

torch.manual_seed(0)               # 아래 실린 로스값이 다시 나오게 하려면 있어야 한다

# 1) NumPy처럼 쓴다
x = torch.randn(1000, 784)

# 2) 장치를 골라 옮긴다
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
x = x.to(device)

# 3) 미분이 필요한 값에는 requires_grad를 붙인다
#
# CPU에서 뽑아 옮긴다. `torch.randn(..., device="mps")` 처럼 장치에서 바로
# 뽑으면 장치마다 난수 흐름이 달라, 같은 씨앗을 물려도 CPU에서는 768.7119,
# MPS에서는 773.8671 이 나온다. 아래 실린 값이 어느 기계에서나 같게 하려면
# CPU에서 뽑은 뒤 옮겨야 한다
W = torch.randn(784, 10).to(device).requires_grad_(True)
y = x @ W
loss = y.pow(2).mean()
loss.backward()

print(f"자료   : {tuple(x.shape)} on {x.device}")
print(f"웨이트 : {tuple(W.shape)} on {W.device}")
print(f"로스   : {loss.item():.4f}")
print(f"경사   : {tuple(W.grad.shape)}  <- W와 같은 모양")
```

**출력:**

```
자료   : (1000, 784) on mps:0
웨이트 : (784, 10) on mps:0
로스   : 768.7119
경사   : (784, 10)  <- W와 같은 모양
```

이 세 줄이 이 책에 나오는 거의 모든 학습 코드의 뼈대이다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
`torch.from_numpy(arr)`로 만든 텐서를 GPU로 옮긴 뒤 `arr`를 고치면 GPU의 텐서도 바뀌겠는가?

</div>

??? success "연습문제 1 풀이"
    바뀌지 않는다. `.to(device)`는 GPU 메모리에 **새로 복사**하는 일이므로 그 순간 둘의 연결이 끊긴다.

    같은 CPU 메모리를 참조할 때만 원래 NumPy 배열과 값을 공유한다. 같은 장치에 있다는 조건만으로 메모리 공유가 보장되지는 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`loss.backward()`를 두 번 부르면 어떻게 되는가?

</div>

??? success "연습문제 2 풀이"
    일반적인 뉴럴 넷트웍 계산에서는 첫 역전파가 저장된 중간 텐서를 해제하므로, 같은 그래프에서 다시 역전파하면 오류가 날 수 있다. 같은 그래프를 재사용해야 한다면 첫 호출에 `retain_graph=True`를 지정한다. 새 순전파로 그래프를 다시 만들었다면 별도의 역전파가 가능하다.

    더 중요한 것은, 경사가 **더해진다**는 점이다. 그래서 학습 루프에서는 매 스텝 `optimizer.zero_grad()`로 지워야 한다. 이를 빠뜨리면 경사가 계속 쌓여 학습이 이상해진다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
NumPy만으로 뉴럴 넷트웍을 학습시킬 수 있는가?

</div>

??? success "연습문제 3 풀이"
    할 수 있다. 실제로 이 책의 여러 곳에서 NumPy로 순전파와 역전파를 손수 짜 본다.

    다만 두 가지를 직접 해야 한다. 레이어마다 미분을 손으로 유도해 코드로 옮겨야 하고, 일반적인 NumPy 연산은 CPU에서 실행되므로 대규모 학습에서는 실행 시간이 길어질 수 있다. 원리를 배울 때는 손수 짜 보는 것이 값지지만, 실제로 쓸 모델에는 PyTorch가 필요하다.

## 정리하며

**다룬 것** — 텐서가 배열에 무엇을 더했는가

텐서는 `ndarray`에 **장치**와 **자동 미분**을 더한 것이다. 문법은 거의 그대로여서 NumPy를 익혔다면 PyTorch의 기본 배열 연산을 배우기 쉽다.

기억할 것은 셋이다. `from_numpy`와 `.numpy()`는 **메모리를 함께 쓴다**는 것, 계산에 참여하는 텐서는 **같은 장치**에 있어야 한다는 것, 그리고 `requires_grad=True`를 붙인 텐서에는 PyTorch가 미분을 대신 구해 준다는 것이다.

다음 절에서는 장치를 바꾸는 일이 실제로 얼마나 이득인지 재어 본다.

앞의 연습문제 3개로 직접 확인할 수 있다.
