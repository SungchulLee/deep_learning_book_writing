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

PyTorch의 `Tensor`는 NumPy의 `ndarray`에 두 가지를 더한 것이다.

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

다른 점은 축을 가리키는 이름이 `axis`에서 `dim`으로 바뀌는 정도이다.

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

메모리를 함께 쓰는 것은 CPU에 있는 텐서에만 해당한다. GPU에 있는 텐서는 `.numpy()`를 바로 부를 수 없고, 먼저 `.cpu()`로 내려야 한다.

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

`Expected all tensors to be on the same device`는 PyTorch를 쓰며 가장 자주 만나는 오류이다. 모델은 GPU에 올려 놓고 자료는 CPU에 둔 채로 넣으면 이 오류가 난다. **모델과 자료를 같은 장치로 보내는 것**이 학습 루프의 기본 규율이다.

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

식이 이렇게 단순하면 손으로 미분해도 된다. 그러나 신경망은 층이 수십 개이고 매개변수가 수백만 개이다. 그 전체를 손으로 미분해 코드로 옮기는 일은 사람이 할 수 있는 일이 아니다.

**자동 미분이 딥러닝을 실용적으로 만든 장치**이며, 이것이 NumPy 대신 PyTorch를 쓰는 가장 큰 까닭이다. 자세한 원리는 [4장의 자동 미분](../../ch04/index.md)에서 다룬다.

---

## 6. 세 줄 요약

```python
import torch

# 1) NumPy처럼 쓴다
x = torch.randn(1000, 784)

# 2) 장치를 골라 옮긴다
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
x = x.to(device)

# 3) 미분이 필요한 값에는 requires_grad를 붙인다
W = torch.randn(784, 10, device=device, requires_grad=True)
y = x @ W
loss = y.pow(2).mean()
loss.backward()

print(f"자료   : {tuple(x.shape)} on {x.device}")
print(f"가중치 : {tuple(W.shape)} on {W.device}")
print(f"손실   : {loss.item():.4f}")
print(f"경사   : {tuple(W.grad.shape)}  <- W와 같은 모양")
```

**출력:**

```
자료   : (1000, 784) on mps:0
가중치 : (784, 10) on mps:0
손실   : 770.3815
경사   : (784, 10)  <- W와 같은 모양
```

이 세 줄이 이 책에 나오는 거의 모든 학습 코드의 뼈대이다.

---

## 연습문제

**연습문제 1.**
`torch.from_numpy(arr)`로 만든 텐서를 GPU로 옮긴 뒤 `arr`를 고치면 GPU의 텐서도 바뀌겠는가?

??? success "연습문제 1 풀이"
    바뀌지 않는다. `.to(device)`는 GPU 메모리에 **새로 복사**하는 일이므로 그 순간 둘의 연결이 끊긴다.

    메모리를 함께 쓰는 관계는 같은 장치에 있을 때만 성립한다.

---

**연습문제 2.**
`loss.backward()`를 두 번 부르면 어떻게 되는가?

??? success "연습문제 2 풀이"
    오류가 난다. PyTorch는 역전파를 마치면 계산 기록을 지워 메모리를 돌려주기 때문이다. 굳이 두 번 하려면 `backward(retain_graph=True)`를 준다.

    더 중요한 것은, 경사가 **더해진다**는 점이다. 그래서 학습 루프에서는 매 걸음 `optimizer.zero_grad()`로 지워야 한다. 이를 빠뜨리면 경사가 계속 쌓여 학습이 이상해진다.

---

**연습문제 3.**
NumPy만으로 신경망을 학습시킬 수 있는가?

??? success "연습문제 3 풀이"
    할 수 있다. 실제로 이 책의 여러 곳에서 NumPy로 순전파와 역전파를 손수 짜 본다.

    다만 두 가지를 직접 해야 한다. 층마다 미분을 손으로 유도해 코드로 옮겨야 하고, GPU를 쓸 수 없어 큰 모델은 학습이 사실상 불가능하다. 원리를 배울 때는 손수 짜 보는 것이 값지지만, 실제로 쓸 모델에는 PyTorch가 필요하다.

## 정리하며

**다룬 것** — 텐서가 배열에 무엇을 더했는가

텐서는 `ndarray`에 **장치**와 **자동 미분**을 더한 것이다. 문법은 거의 그대로여서 NumPy를 알면 PyTorch의 절반은 이미 아는 셈이다.

기억할 것은 셋이다. `from_numpy`와 `.numpy()`는 **메모리를 함께 쓴다**는 것, 계산에 참여하는 텐서는 **같은 장치**에 있어야 한다는 것, 그리고 `requires_grad=True`를 붙인 텐서에는 PyTorch가 미분을 대신 구해 준다는 것이다.

다음 절에서는 장치를 바꾸는 일이 실제로 얼마나 이득인지 재어 본다.

앞의 연습문제 3개로 직접 확인할 수 있다.
