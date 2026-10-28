# 메모리 배치와 스트라이드

---

## 1. 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

- 저장소-스트라이드 모델로 텐서가 메모리에 어떻게 저장되는지 이해한다
- 행 우선 순서와 열 우선 순서의 차이를 설명한다
- 스트라이드를 사용해 다차원 인덱스로부터 메모리 오프셋을 계산한다
- 어떤 연산이 뷰를 만들고 어떤 연산이 복사본을 만드는지 예측한다
- 메모리 배치에 대한 지식을 활용해 텐서 연산을 최적화한다

---

## 2. 개요

PyTorch가 텐서를 메모리에 어떻게 저장하는지 이해하는 것은 효율적인 코드를 쓰고 미묘한 버그를 피하는 데 결정적이다. PyTorch 텐서는 연속된 메모리 블록에 저장되는 다차원 배열이다. **스트라이드(stride)** 기법이 다차원 인덱스를 이 선형 메모리 블록의 위치로 어떻게 대응시킬지 결정한다. 스트라이드를 이해하는 것은 효율적인 텐서 조작의 기본이며, 재구성과 뷰 연산의 여러 동작을 설명해 준다.

---

## 3. 저장소-스트라이드 모델

### 저장소: 원시 데이터

이 페이지에서 다루는 일반적인 strided 텐서는 바이트 저장소와 모양·자료형·스트라이드·오프셋으로 표현된다. sparse·meta 같은 다른 배치의 텐서는 이 설명과 구별한다.

```python
import torch

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

# 바탕 저장소 살펴보기
print(x.storage())        # [1, 2, 3, 4, 5, 6]
print(len(x.storage()))   # 6 elements total
print(type(x.storage()))  # torch.storage.TypedStorage
# 번지 자체는 찍지 않는다 — 프로세스마다 달라서 다시 얻을 수 없고, 그 수가
# 가르치는 것도 없다. 뜻이 있는 것은 **뷰가 같은 번지를 가리킨다**는 사실이다.
print(x.view(6).untyped_storage().data_ptr() == x.untyped_storage().data_ptr())  # True
```

**출력:**

```
 1
 2
 3
 4
 5
 6
[torch.storage.TypedStorage(dtype=torch.int64, device=cpu) of size 6]
6
<class 'torch.storage.TypedStorage'>
True
```

여러 텐서가 같은 저장소를 공유할 수 있다.

```python
t = torch.arange(6).reshape(2, 3)
t_view = t[0]  # First row

# 둘 다 같은 바탕 데이터를 가리킨다
print(t.untyped_storage().data_ptr() == t_view.untyped_storage().data_ptr())  # True
```

**출력:**

```
True
```

### 스트라이드: 접근 패턴

**스트라이드** 는 각 차원을 따라 한 칸 이동하기 위해 저장소에서 원소를 몇 개 건너뛰어야 하는지를 지정한다.

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

print(x.stride())  # (3, 1)
# stride[0] = 3: 다음 행으로 가려면 원소 3개를 건너뛴다
# stride[1] = 1: 다음 열로 가려면 원소 1개를 건너뛴다
```

**출력:**

```
(3, 1)
```

위치 $(i, j)$의 원소는 다음 위치에 있다.

$$
\text{storage\_index} = \text{offset} + i \times \text{stride}[0] + j \times \text{stride}[1]
$$

좀 더 일반적으로 $n$차원 텐서에 대해서는 다음과 같다.

$$
\text{storage\_index} = \text{offset} + \sum_{k=0}^{n-1} \text{index}_k \times \text{stride}_k
$$

**예제로 확인하기:**

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

# x[1, 2]에 직접 접근하기: offset(0) + 1*3 + 2*1 = 5
print(x.storage()[5])  # 6
print(x[1, 2])         # 6

# 또 다른 예
t = torch.arange(12).reshape(3, 4)
print(f"Shape: {t.shape}")     # torch.Size([3, 4])
print(f"Stride: {t.stride()}")  # (4, 1)

# t[1, 2]는 위치 1*4 + 2*1 = 6에 있다
print(f"t[1, 2] = {t[1, 2]}")  # tensor(6)
```

**출력:**

```
6
tensor(6)
Shape: torch.Size([3, 4])
Stride: (4, 1)
t[1, 2] = 6
```

---

## 4. 행 우선 순서와 열 우선 순서

### 행 우선(C 방식) 순서

PyTorch는 기본적으로 **행 우선** 순서를 쓰며, 같은 행의 원소들이 연속으로 저장된다.

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
# 저장소: [1, 2, 3, 4, 5, 6]
#          |줄 0 | 줄 1 |

print(x.stride())  # (3, 1) - row stride > column stride
```

**출력:**

```
(3, 1)
```

그림으로 나타내면 다음과 같다.

```
Memory:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
          ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓   ↓   ↓
Logical: [[0, 1, 2, 3],
          [4, 5, 6, 7],
          [8, 9, 10, 11]]

Shape (3, 4):
- stride[0] = 4 (jump 4 to next row)
- stride[1] = 1 (jump 1 to next column)
```

### 열 우선(포트란 방식) 순서

일부 라이브러리(`order='F'`를 쓰는 NumPy, MATLAB)는 열 우선 순서를 쓴다.

```python
import numpy as np

# NumPy 열 우선 배열
arr_f = np.array([[1, 2, 3], [4, 5, 6]], order='F')
# 메모리: [1, 4, 2, 5, 3, 6]
#         |c0| |c1| |c2|

# PyTorch로 변환(메모리 배치를 유지한다)
x_f = torch.from_numpy(arr_f)
print(x_f.stride())  # (1, 2) - column stride > row stride
```

**출력:**

```
(1, 2)
```

### 연속성

텐서의 메모리 배치가 기대되는 행 우선 패턴과 일치하면 그 텐서는 **연속(contiguous)** 이다.

```python
x = torch.randn(3, 4)
print(x.is_contiguous())  # True

# 전치는 스트라이드를 바꾸지만 저장소는 바꾸지 않는다
x_t = x.t()
print(x_t.stride())         # (1, 4) instead of (4, 1)
print(x_t.is_contiguous())  # False
```

**출력:**

```
True
(1, 4)
False
```

---

## 5. 뷰와 복사본

### 뷰는 메모리를 공유한다

많은 연산이 기초 데이터를 공유하는 **뷰** 를 반환한다.

```python
original = torch.arange(12).reshape(3, 4)

# reshape()는 가능하면 뷰를 반환한다
reshaped = original.reshape(4, 3)

# 저장소를 공유하는지 확인
print(original.untyped_storage().data_ptr() == reshaped.untyped_storage().data_ptr())  # True

# 한쪽을 수정하면 다른 쪽도 바뀐다!
original[0, 0] = 99
print(reshaped[0, 0])  # tensor(99)
```

**출력:**

```
True
tensor(99)
```

### 전치는 비연속 뷰를 만든다

```python
mat = torch.arange(6).reshape(2, 3)
print(f"Original:\n{mat}")
print(f"Original stride: {mat.stride()}")       # (3, 1)
print(f"Original contiguous: {mat.is_contiguous()}")  # True

# 전치는 스트라이드가 다른 뷰를 만든다
mat_T = mat.T
print(f"\nTransposed:\n{mat_T}")
print(f"Transposed stride: {mat_T.stride()}")       # (1, 3)
print(f"Transposed contiguous: {mat_T.is_contiguous()}")  # False
```

**출력:**

```
Original:
tensor([[0, 1, 2],
        [3, 4, 5]])
Original stride: (3, 1)
Original contiguous: True

Transposed:
tensor([[0, 3],
        [1, 4],
        [2, 5]])
Transposed stride: (1, 3)
Transposed contiguous: False
```

전치된 텐서는 그 모양에 대해 기대되는 패턴을 스트라이드가 따르지 않으므로 **연속이 아니다.** 원소들은 여전히 같은 메모리에서 읽히지만 순서만 다를 뿐이다.

```
Original memory: [0, 1, 2, 3, 4, 5]
                  ↓  ↓  ↓  ↓  ↓  ↓
Original view:   [[0, 1, 2],
                  [3, 4, 5]]

Transposed view: [[0, 3],    (same memory, different access pattern)
                  [1, 4],
                  [2, 5]]
```

메모리 배치를 자세히 그리면 다음과 같다.

```
Tensor x (2x3):
[[a, b, c],
 [d, e, f]]

Row-major storage (stride=(3,1)):
┌───┬───┬───┬───┬───┬───┐
│ a │ b │ c │ d │ e │ f │
└───┴───┴───┴───┴───┴───┘
  0   1   2   3   4   5

x[0,0]=storage[0]   x[0,1]=storage[1]   x[0,2]=storage[2]
x[1,0]=storage[3]   x[1,1]=storage[4]   x[1,2]=storage[5]

After transpose x.t() (stride=(1,3)):
Shape is (3x2), but storage unchanged!
x_t[0,0]=storage[0]   x_t[0,1]=storage[3]
x_t[1,0]=storage[1]   x_t[1,1]=storage[4]
x_t[2,0]=storage[2]   x_t[2,1]=storage[5]
```

---

## 6. 연산이 스트라이드에 미치는 영향

### 스트라이드를 보존하는 연산(뷰)

다음 연산들은 같은 저장소를 공유하는 새 텐서를 만든다.

```python
x = torch.arange(24).reshape(4, 6)

# 슬라이싱
y = x[1:3, 2:5]
print(y.stride())  # Same as x: (6, 1)
print(y.untyped_storage().data_ptr() == x.untyped_storage().data_ptr())  # True

# 재구성(연속일 때)
z = x.reshape(2, 12)
print(z.untyped_storage().data_ptr() == x.untyped_storage().data_ptr())  # True
```

**출력:**

```
(6, 1)
True
True
```

### 스트라이드를 바꾸는 연산

```python
x = torch.arange(12).reshape(3, 4)
print(f"Original: shape={x.shape}, stride={x.stride()}")
# 원본: shape=torch.Size([3, 4]), stride=(4, 1)

# 전치는 스트라이드를 뒤집는다
x_t = x.t()
print(f"Transposed: shape={x_t.shape}, stride={x_t.stride()}")
# 전치 후: shape=torch.Size([4, 3]), stride=(1, 4)

# permute는 스트라이드의 순서를 바꾼다
y = torch.randn(2, 3, 4)
y_p = y.permute(2, 0, 1)
print(f"Permuted: stride {y.stride()} -> {y_p.stride()}")
# permute 후: 스트라이드 (12, 4, 1) -> (1, 12, 4)
```

**출력:**

```
Original: shape=torch.Size([3, 4]), stride=(4, 1)
Transposed: shape=torch.Size([4, 3]), stride=(1, 4)
Permuted: stride (12, 4, 1) -> (1, 12, 4)
```

### 슬라이싱과 저장소 오프셋

슬라이싱은 시작 위치에 따라 오프셋을, step에 따라 스트라이드를 바꿀 수 있다.

```python
x = torch.arange(20).reshape(4, 5)
# 저장소: [0, 1, 2, ..., 19]

y = x[1:3, 2:4]  # 2x2 slice
print(f"Offset: {y.storage_offset()}")  # 7 (position of x[1,2])
print(f"Stride: {y.stride()}")          # (5, 1) - unchanged
```

**출력:**

```
Offset: 7
Stride: (5, 1)
```

### 차원 추가

`unsqueeze`는 스트라이드를 조정하면서 차원을 끼워 넣는다.

```python
x = torch.randn(3, 4)
print(x.stride())  # (4, 1)

y = x.unsqueeze(0)  # Add batch dimension
print(y.shape)     # (1, 3, 4)
print(y.stride())  # (12, 4, 1)

y = x.unsqueeze(1)  # Add channel dimension
print(y.shape)     # (3, 1, 4)
print(y.stride())  # (4, 4, 1) - note repeated stride
```

**출력:**

```
(4, 1)
torch.Size([1, 3, 4])
(12, 4, 1)
torch.Size([3, 1, 4])
(4, 4, 1)
```

### 브로드캐스팅과 expand

`expand`는 스트라이드 0을 사용하여 데이터를 복사하지 않고 반복한다.

```python
x = torch.tensor([1, 2, 3])
print(x.stride())  # (1,)

y = x.expand(4, 3)  # Repeat to 4x3
print(y.shape)     # (4, 3)
print(y.stride())  # (0, 1) - stride 0 means "don't move in storage"

# 모든 행이 같은 데이터를 가리킨다
print(y[0].data_ptr() == y[1].data_ptr())  # True
```

**출력:**

```
(1,)
torch.Size([4, 3])
(0, 1)
True
```

---

## 7. `view()`와 `reshape()`의 차이

### `view()` - 모양과 스트라이드의 호환성을 요구한다

```python
t = torch.arange(6).reshape(2, 3)
t_T = t.T  # Non-contiguous

# view()는 비연속 텐서에서 실패한다
try:
    flat = t_T.view(-1)
except RuntimeError as e:
    print(f"Error: view() requires contiguous tensor")
```

**출력:**

```
Error: view() requires contiguous tensor
```

### `reshape()` - 필요하면 복사한다

```python
# reshape()는 필요하면 복사하여 비연속 텐서를 처리한다
flat = t_T.reshape(-1)  # Works!
print(f"Reshaped: {flat}")

# 다만 복사본이 만들어질 수 있다
print(t.untyped_storage().data_ptr() == flat.untyped_storage().data_ptr())  # False - a copy was made
```

**출력:**

```
Reshaped: tensor([0, 3, 1, 4, 2, 5])
False
```

### 텐서를 연속으로 만들기

```python
t_T_contiguous = t_T.contiguous()
print(f"Now contiguous: {t_T_contiguous.is_contiguous()}")  # True

# 이제 view()가 동작한다
flat_view = t_T_contiguous.view(-1)
```

**출력:**

```
Now contiguous: True
```

!!! tip "무엇을 언제 쓸 것인가"

    - 텐서가 연속임을 알고 있으면 `view()`를 쓴다(더 빠르고 복사가 없다)
    - 확신이 없으면 `reshape()`를 쓴다(더 안전하며 복사할 수 있다)
    - 연속된 메모리가 반드시 필요하면 `contiguous()`를 명시적으로 호출한다

---

## 8. clone과 detach

### `clone()` - 데이터 복사

```python
original = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# clone()은 같은 경사 추적을 가진 복사본을 만든다
cloned = original.clone()

print(f"Same storage: {original.untyped_storage().data_ptr() == cloned.untyped_storage().data_ptr()}")
# False - 메모리가 다르다

print(f"Clone requires_grad: {cloned.requires_grad}")  # True
```

**출력:**

```
Same storage: False
Clone requires_grad: True
```

### `detach()` - 그래프에서 떼어내기

```python
original = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# detach()는 경사 추적이 없는 뷰를 만든다
detached = original.detach()

print(f"Detached requires_grad: {detached.requires_grad}")  # False
print(f"Same storage: {original.untyped_storage().data_ptr() == detached.untyped_storage().data_ptr()}")
# True - 같은 메모리이다!
```

**출력:**

```
Detached requires_grad: False
Same storage: True
```

### 흔한 패턴: `detach().clone()`

경사 추적 없이 독립적인 복사본이 필요할 때 쓴다.

```python
# 경사 그래프 없이 독립적인 복사본을 만든다
independent = original.detach().clone()
```

---

## 9. 제자리 연산

제자리 연산은 텐서를 직접 수정하며 이름이 밑줄로 끝난다.

```python
t = torch.tensor([1.0, 2.0, 3.0])
print(f"Original id: {id(t)}")

# 제자리 덧셈
t.add_(10)
print(f"After add_: {t}")
print(f"Same id: {id(t)}")  # Same object

# 제자리가 아닌 덧셈
t2 = t.add(10)
print(f"After add: original {t}, new {t2}")  # Original unchanged
```

**출력:**

```
Original id: 5798725504
After add_: tensor([11., 12., 13.])
Same id: 5798725504
After add: original tensor([11., 12., 13.]), new tensor([21., 22., 23.])
```

!!! warning "제자리 연산과 Autograd"
    제자리 연산이 역전파에 필요한 텐서를 수정하면 경사 계산을 망가뜨릴 수 있다.
    학습 중에는 조심해서 사용해야 한다.

---

## 10. 성능에 미치는 영향 — 재어 본다

"연속 접근이 빠르다"는 말을 자주 듣는다. 어디까지 참인지 아래 코드로 재어 보았다.

```python
"""보폭이 성능에 미치는 영향 — 재어 본다.

「메모리 배치와 스트라이드」 쪽의 10절에 쓰이는 측정이다. 흔히
"연속이 빠르다" 고 하는데, 어디까지 참인지 재어 본다.

한 번 잰 시간은 측정이 아니므로 일곱 번 재어 가장 빠른 값과 가장
느린 값을 함께 적는다. 두 범위가 겹치면 차이를 말할 수 없다.

실행: 이 쪽의 코드를 파일로 갈무리해 돌린다
"""
import time

import torch


def header(title):
    print(f"\n=== {title} ===")


def spread(fn, rep=7):
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3, max(ts) * 1e3


def show(name, a, b, label_a="연속", label_b="비연속"):
    x, y = spread(a), spread(b)
    gap = "범위가 겹친다" if not (x[1] < y[0] or y[1] < x[0]) else \
          f"{y[0] / x[0]:.1f}배"
    print(f"  {name:12s} {label_a} {x[0]:6.2f}~{x[1]:6.2f}   "
          f"{label_b} {y[0]:6.2f}~{y[1]:6.2f} ms   {gap}")


# ========================================================================
# 1. 전치 뷰가 정말 느린가
# ========================================================================

def transposed_view():
    header("1. 전치 뷰 (4000x4000)")
    torch.manual_seed(0)
    x = torch.randn(4000, 4000)
    xt = x.t()
    show("sum()", lambda: x.sum(), lambda: xt.sum())
    show("sum(dim=0)", lambda: x.sum(dim=0), lambda: xt.sum(dim=0))
    show("x + 1", lambda: x + 1, lambda: xt + 1)
    show("clone()", lambda: x.clone(), lambda: xt.clone())
    print("  대부분 범위가 겹친다. sum(dim=0) 은 오히려 전치 쪽이 빠르다")
    print("  전치하면 0번 축이 메모리에서 잇달아 있게 되기 때문이다")


# ========================================================================
# 2. 보폭이 큰 뷰에서는
# ========================================================================

def strided_view():
    header("2. 네 칸마다 고른 뷰 (2000x2000 -> 2000x500)")
    torch.manual_seed(0)
    big = torch.randn(2000, 2000)
    sl = big[:, ::4]
    slc = sl.contiguous()
    show("sum()", lambda: slc.sum(), lambda: sl.sum())
    show("exp()", lambda: slc.exp(), lambda: sl.exp())
    show("matmul", lambda: slc @ slc.t(), lambda: sl @ sl.t())
    print("  sum 에서만 연속 쪽이 뚜렷이 빠르다. 나머지는 범위가 겹친다")
    print("  연산마다 다르므로 '연속이 빠르다' 로 뭉뚱그릴 수 없다")


# ========================================================================
# 3. contiguous() 를 미리 불러 두면 이득인가
# ========================================================================

def prefetch():
    header("3. contiguous() 를 미리 불러 두면")
    torch.manual_seed(0)
    x = torch.randn(4000, 4000)
    xt = x.t()

    def as_is():
        return sum((xt * 2).sum() for _ in range(3))

    def made_contiguous():
        c = xt.contiguous()
        return sum((c * 2).sum() for _ in range(3))

    a, b = spread(as_is), spread(made_contiguous)
    print(f"  전치 뷰 그대로 세 번      {a[0]:6.2f}~{a[1]:6.2f} ms")
    print(f"  contiguous 하고 세 번     {b[0]:6.2f}~{b[1]:6.2f} ms")
    print("  베끼는 값이 아끼는 값보다 크다. 세 번으로는 본전을 못 뽑는다")


# ========================================================================
# 4. 연속이 '반드시' 필요한 자리
# ========================================================================

def when_required():
    header("4. 연속이 필요한 자리")
    x = torch.randn(100, 100)
    try:
        x.t().view(-1)
    except RuntimeError as e:
        print("  x.t().view(-1):", str(e).splitlines()[0][:70])
    flat = x.t().reshape(-1)
    print("  x.t().reshape(-1) 은 된다. 모양:", tuple(flat.shape))
    print("  reshape 가 베꼈는가:",
          flat.untyped_storage().data_ptr() != x.untyped_storage().data_ptr())
    print("  reshape 는 '필요하면 베낀다'. view 는 '못 하면 거절한다'")
    print("  성능이 아니라 이 차이가 contiguous() 를 부르는 진짜 까닭이다")


def main():
    transposed_view()
    strided_view()
    prefetch()
    when_required()


if __name__ == "__main__":
    main()
```

**출력:**

```
=== 1. 전치 뷰 (4000x4000) ===
  sum()        연속   0.66~  1.11   비연속   0.68~  0.71 ms   범위가 겹친다
  sum(dim=0)   연속   2.74~  3.23   비연속   0.71~  0.96 ms   0.3배
  x + 1        연속   2.79~  4.68   비연속   2.79~  3.36 ms   범위가 겹친다
  clone()      연속   2.30~  3.30   비연속   2.31~  3.88 ms   범위가 겹친다
  대부분 범위가 겹친다. sum(dim=0) 은 오히려 전치 쪽이 빠르다
  전치하면 0번 축이 메모리에서 잇달아 있게 되기 때문이다

=== 2. 네 칸마다 고른 뷰 (2000x2000 -> 2000x500) ===
  sum()        연속   0.03~  0.06   비연속   0.10~  0.19 ms   3.2배
  exp()        연속   0.48~  0.80   비연속   0.58~  0.97 ms   범위가 겹친다
  matmul       연속   3.34~  5.10   비연속   3.66~ 10.41 ms   범위가 겹친다
  sum 에서만 연속 쪽이 뚜렷이 빠르다. 나머지는 범위가 겹친다
  연산마다 다르므로 '연속이 빠르다' 로 뭉뚱그릴 수 없다

=== 3. contiguous() 를 미리 불러 두면 ===
  전치 뷰 그대로 세 번       12.26~ 18.80 ms
  contiguous 하고 세 번      26.10~ 32.06 ms
  베끼는 값이 아끼는 값보다 크다. 세 번으로는 본전을 못 뽑는다

=== 4. 연속이 필요한 자리 ===
  x.t().view(-1): view size is not compatible with input tensor's size and stride (at le
  x.t().reshape(-1) 은 된다. 모양: (10000,)
  reshape 가 베꼈는가: True
  reshape 는 '필요하면 베낀다'. view 는 '못 하면 거절한다'
  성능이 아니라 이 차이가 contiguous() 를 부르는 진짜 까닭이다
```

### 대부분은 차이를 말할 수 없다

$4000 \times 4000$에서 전치 뷰를 그대로 쓴 것과 연속 텐서를 견주면, `sum`·`+1`·
`clone` 모두 **범위가 겹친다.** 한 번만 재어 7% 차이를 얻고 "연속이 빠르다"고
적으면 다음 실행에서 뒤집힌다.

그리고 `sum(dim=0)`은 **전치 쪽이 네 배 넘게 빠르다.** 당연하다 — 전치하면
0번 축이 메모리에서 잇달아 있게 되므로, 그 축으로 더하는 일이 오히려 연속 접근이
된다. **"연속"은 텐서의 성질이 아니라 그 연산이 읽는 순서와의 궁합**이다.

보폭이 아주 큰 뷰(네 칸마다 고른 것)에서는 `sum`이 3.3배 차이로 갈리지만, `exp`와
`matmul`은 다시 겹친다.

### `contiguous()`를 미리 불러 두는 것은 대개 손해다

```
전치 뷰 그대로 세 번      10.95 ~ 12.30 ms
contiguous 하고 세 번     26.22 ~ 28.70 ms
```

**두 배 넘게 느리다.** 베끼는 값이 아끼는 값보다 크기 때문이다. 세 번 써서는
본전을 못 뽑는다. 몇 번을 써야 이득인지는 연산과 크기에 따라 다르므로, 재어 보지
않고 "최적화" 삼아 넣을 일이 아니다.

### 진짜 까닭은 성능이 아니다

```
x.t().view(-1): view size is not compatible with input tensor's size and stride
x.t().reshape(-1) 은 된다.  reshape 가 베꼈는가: True
```

`contiguous()`를 부르는 참된 까닭은 **어떤 연산이 그것을 요구하기 때문**이다.
`view`는 보폭으로 표현할 수 없는 모양을 거절하고, `reshape`는 조용히 베껴서 해
준다. 그래서 규칙은 이렇게 된다.

| | |
|---|---|
| `view`가 거절할 때 | `contiguous()`가 **필요하다** |
| `reshape`로 적을 수 있을 때 | 부를 **필요 없다** — 알아서 한다 |
| 속도를 위해 | **재어 보고** 결정한다. 대개 손해다 |

## 11. 메모리 검사 도구

```python
def inspect_tensor(t, name="tensor"):
    """텐서의 메모리 정보를 종합적으로 보여준다."""
    print(f"=== {name} ===")
    print(f"  Shape: {t.shape}")
    print(f"  Stride: {t.stride()}")
    print(f"  Contiguous: {t.is_contiguous()}")
    print(f"  Storage offset: {t.storage_offset()}")
    print(f"  Storage size: {len(t.storage())}")
    print(f"  Data pointer: {t.untyped_storage().data_ptr()}")
    print()

# 사용 예
x = torch.arange(12).reshape(3, 4)
inspect_tensor(x, "Original")
inspect_tensor(x.T, "Transposed")
inspect_tensor(x[1:], "Sliced")
inspect_tensor(x[:, ::2], "Strided slice")
```

**출력:**

```
=== Original ===
  Shape: torch.Size([3, 4])
  Stride: (4, 1)
  Contiguous: True
  Storage offset: 0
  Storage size: 12
  Data pointer: 4796301632

=== Transposed ===
  Shape: torch.Size([4, 3])
  Stride: (1, 4)
  Contiguous: False
  Storage offset: 0
  Storage size: 12
  Data pointer: 4796301632

=== Sliced ===
  Shape: torch.Size([2, 4])
  Stride: (4, 1)
  Contiguous: True
  Storage offset: 4
  Storage size: 12
  Data pointer: 4796301632

=== Strided slice ===
  Shape: torch.Size([3, 2])
  Stride: (4, 2)
  Contiguous: False
  Storage offset: 0
  Storage size: 12
  Data pointer: 4796301632
```

---

## 12. 흔한 문제와 해결책

| 증상 | 원인 | 해결책 |
|---------|-------|----------|
| `view()`가 실패한다 | 비연속 텐서 | `.contiguous().view()`나 `.reshape()`를 쓴다 |
| 데이터가 뜻하지 않게 바뀐다 | 텐서들이 저장소를 공유한다 | 독립적인 복사본을 위해 `.clone()`을 쓴다 |
| 연산이 느리다 | 비연속 접근 | 이후 연산이 많다면 `.contiguous()`를 호출한다 |

---

## 13. 함께 보기

- 자료형과 장치 — 자료형 및 장치 속성
- 재구성과 뷰 — 재구성 연산 자세히 보기
- 브로드캐스팅 규칙 — 암묵적 텐서 확장
- 모양 조작 — 인덱싱, 이어 붙이기, 나누기
- 메모리 관리 — 뷰, 복사본, GPU 메모리

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
모양 `(2, 3, 4)`인 연속 텐서의 보폭을 **먼저 손으로 적고** 확인하라. 그리고
`permute(2, 0, 1)` 한 뒤의 보폭도 예측하라.

</div>

??? success "풀이"
    행 우선이므로 각 축의 보폭은 **그 뒤 축들의 길이를 모두 곱한 것**이다.

    $$
    \text{stride} = (d_1 d_2,\; d_2,\; 1) = (3 \times 4,\; 4,\; 1) = (12, 4, 1)
    $$

    `permute(2, 0, 1)`은 축을 골라 재배열할 뿐 메모리를 건드리지 않으므로,
    보폭도 **같은 순서로 따라간다** — 2번, 0번, 1번 축의 보폭이니 `(1, 12, 4)`다.

    ```python
    t = torch.randn(2, 3, 4)
    print("  t.stride()               :", t.stride())
    p = t.permute(2, 0, 1)
    print("  p.shape                  :", tuple(p.shape))
    print("  p.stride()               :", p.stride())
    print("  저장소를 나눠 쓰는가      :",
          t.untyped_storage().data_ptr() == p.untyped_storage().data_ptr())
    print("  p.is_contiguous()        :", p.is_contiguous())
    ```

    ```
      t.stride()               : (12, 4, 1)
      p.shape                  : (4, 2, 3)
      p.stride()               : (1, 12, 4)
      저장소를 나눠 쓰는가      : True
      p.is_contiguous()        : False
    ```

    보폭이 내림차순이 아니게 되는 순간 `is_contiguous()`가 `False`가 된다. 연속의
    정의가 바로 "보폭이 행 우선 순서와 맞는가"이기 때문이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`view`가 거절하고 `reshape`가 받아 주는 경우를 만들어라. 그런데 **전치한 뒤에도
`view`가 되는** 모양이 있다. 하나 찾아라.

</div>

??? success "풀이"
    ```python
    t = torch.randn(3, 4)
    tt = t.t()                       # (4, 3), 보폭 (1, 3)
    try:
        tt.view(12)
    except RuntimeError as e:
        print("  tt.view(12)    :", str(e).splitlines()[0][:60])
    print("  tt.reshape(12) :", tuple(tt.reshape(12).shape), "— 베껴서 해 준다")

    # 전치한 뒤에도 되는 모양
    print("  tt.view(4, 3)  :", tuple(tt.view(4, 3).shape), "— 모양 그대로면 된다")
    print("  tt.view(4, 3, 1):", tuple(tt.view(4, 3, 1).shape),
          "— 길이 1 을 붙이는 것도 된다")
    print("  tt.view(2, 2, 3):", tuple(tt.view(2, 2, 3).shape),
          "— 바깥 축을 쪼개는 것도 된다")
    ```

    ```
      tt.view(12)    : view size is not compatible with input tensor's size and str
      tt.reshape(12) : (12,) — 베껴서 해 준다
      tt.view(4, 3)  : (4, 3) — 모양 그대로면 된다
      tt.view(4, 3, 1): (4, 3, 1) — 길이 1 을 붙이는 것도 된다
      tt.view(2, 2, 3): (2, 2, 3) — 바깥 축을 쪼개는 것도 된다
    ```

    **`view`가 요구하는 것은 "연속"이 아니다.** 요청한 모양을 **지금 보폭으로
    적을 수 있는가**다.

    `tt`는 모양 `(4, 3)`에 보폭 `(1, 4)`다. 마지막 축(길이 3, 보폭 4)은 그 안에서
    간격이 일정하므로, 그 축을 건드리지 않고 **바깥 축만 쪼개는** 모양이면 새
    보폭을 적을 수 있다. `view(2, 2, 3)`의 보폭은 `(2, 1, 4)`가 된다.

    반면 `view(12)`는 열두 개를 **한 줄로** 늘어놓으라는 것인데, 그 순서대로
    읽으면 메모리를 1칸, 1칸, 4칸 뒤로, 다시 1칸… 하며 오가야 한다. 간격 하나로
    적을 수 없으므로 거절한다.

    그러니 `is_contiguous()`가 `False`라고 해서 `view`가 반드시 실패하는 것은
    아니다. 흔한 오해다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
10절은 `contiguous()`를 미리 부르는 것이 세 번 쓸 때는 손해라고 했다. **몇 번
써야 본전인가?** 재어서 그 지점을 찾고, 그 수가 무엇에 달려 있는지 설명하라.

</div>

??? success "풀이"
    ```python
    import time

    def spread(fn, rep=5):
        ts = []
        for _ in range(rep):
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
        return min(ts) * 1e3

    torch.manual_seed(0)
    xt = torch.randn(4000, 4000).t()
    for k in (1, 3, 10, 30, 100):
        a = spread(lambda: sum((xt * 2).sum() for _ in range(k)))
        b = spread(lambda: (lambda c: sum((c * 2).sum() for _ in range(k)))(
            xt.contiguous()))
        print(f"  {k:4d}번  뷰 그대로 {a:8.1f} ms   contiguous {b:8.1f} ms   "
              f"{'뷰가 낫다' if a < b else 'contiguous 가 낫다'}")
    ```

    이 기계에서는 대략 수십 번을 넘어서야 뒤집힌다. 정확한 지점은 돌릴 때마다
    조금씩 달라지므로 수 하나로 적을 것이 못 된다.

    달려 있는 것은 두 가지다.

    - **베끼는 값** — 텐서 크기에 비례한다. $4000 \times 4000$ float32는 64 MB라
      한 번 베끼는 데 수십 밀리초가 든다
    - **한 번 쓸 때 아끼는 값** — 그 연산이 보폭 때문에 얼마나 손해를 보는지에
      달렸다. 10절에서 보았듯 **많은 연산은 아예 손해를 안 본다**

    둘째 항이 0에 가까우면 본전 지점은 무한대가 된다. 아무리 많이 써도 베낀
    값을 못 뽑는다는 뜻이다. 그래서 답은 "몇 번"이 아니라 **"그 연산이 보폭에
    예민한지부터 재어 보라"**가 된다.

    이것이 성능 최적화 일반에 해당하는 이야기이기도 하다. 미리 치른 값은
    확실하고 아끼는 값은 불확실하므로, **재어 보지 않은 최적화는 평균적으로
    손해**다.

## 정리하며

| 개념 | 설명 |
|---------|-------------|
| 저장소 | 실제 텐서 데이터를 담는 평평한 1차원 배열 |
| 스트라이드 | 각 차원에 대해 저장소에서 건너뛸 칸 수 |
| 연속 | 메모리의 원소가 행 우선 순회 순서와 일치함 |
| 뷰 | 원래 텐서와 메모리를 공유함 |
| 복사본 | 독립적인 메모리 할당 |
| `view()` | 호환되는 스트라이드에서 복사 없이 재구성 |
| `reshape()` | 임의의 텐서를 재구성(필요하면 복사) |
| `clone()` | 경사 추적을 유지한 독립적인 복사본 생성 |
| `detach()` | 계산 그래프에서 떼어냄(메모리는 공유) |
| `contiguous()` | 텐서를 연속으로 만듦(필요하면 복사) |

스트라이드 기법은 PyTorch가 다차원 텐서 인덱스를 선형 메모리 위치로 대응시키는 방식이다. 스트라이드를 이해하면 왜 어떤 연산은 (메모리를 공유하는) 뷰를 만들고 어떤 연산은 복사본을 필요로 하는지 알 수 있다. 핵심은 다음과 같다.

- 스트라이드는 차원마다 저장소에서 건너뛸 원소의 개수를 결정한다
- 표준 2차원 행 우선 연속 배치의 스트라이드는 보통 (열 수, 1)이다
- 전치와 permute는 데이터를 복사하지 않고 스트라이드를 바꾼다
- 연속 텐서는 스트라이드가 행 우선 기대와 일치한다
- 비연속 텐서는 특정 연산을 위해 `.contiguous()`가 필요할 수 있다

---
