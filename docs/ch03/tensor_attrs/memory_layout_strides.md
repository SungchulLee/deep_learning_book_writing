# 메모리 배치와 스트라이드

---

## 1. 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

- 저장소-스트라이드 모형으로 텐서가 메모리에 어떻게 저장되는지 이해한다
- 행 우선 순서와 열 우선 순서의 차이를 설명한다
- 스트라이드를 사용해 다차원 인덱스로부터 메모리 오프셋을 계산한다
- 어떤 연산이 뷰를 만들고 어떤 연산이 복사본을 만드는지 예측한다
- 메모리 배치에 대한 지식을 활용해 텐서 연산을 최적화한다

---

## 2. 개요

PyTorch가 텐서를 메모리에 어떻게 저장하는지 이해하는 것은 효율적인 코드를 쓰고 미묘한 버그를 피하는 데 결정적이다. PyTorch 텐서는 연속된 메모리 블록에 저장되는 다차원 배열이다. **스트라이드(stride)** 기법이 다차원 인덱스를 이 선형 메모리 블록의 위치로 어떻게 대응시킬지 결정한다. 스트라이드를 이해하는 것은 효율적인 텐서 조작의 기본이며, 재구성과 뷰 연산의 여러 동작을 설명해 준다.

---

## 3. 저장소-스트라이드 모형

### 저장소: 원시 데이터

모든 PyTorch 텐서는 **Storage** 객체, 즉 자료형이 정해진 원소들의 평평한 1차원 배열이 뒷받침한다.

```python
import torch

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

# 바탕 저장소 살펴보기
print(x.storage())        # [1, 2, 3, 4, 5, 6]
print(len(x.storage()))   # 6 elements total
print(type(x.storage()))  # torch.storage.TypedStorage
print(x.storage().data_ptr())  # Memory address
```

여러 텐서가 같은 저장소를 공유할 수 있다.

```python
t = torch.arange(6).reshape(2, 3)
t_view = t[0]  # First row

# 둘 다 같은 바탕 데이터를 가리킨다
print(t.storage().data_ptr() == t_view.storage().data_ptr())  # True
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

---

## 5. 뷰와 복사본

### 뷰는 메모리를 공유한다

많은 연산이 바탕 데이터를 공유하는 **뷰** 를 반환한다.

```python
original = torch.arange(12).reshape(3, 4)

# reshape()는 가능하면 뷰를 반환한다
reshaped = original.reshape(4, 3)

# 저장소를 공유하는지 확인
print(original.storage().data_ptr() == reshaped.storage().data_ptr())  # True

# 한쪽을 수정하면 다른 쪽도 바뀐다!
original[0, 0] = 99
print(reshaped[0, 0])  # tensor(99)
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
print(y.storage().data_ptr() == x.storage().data_ptr())  # True

# 재구성(연속일 때)
z = x.reshape(2, 12)
print(z.storage().data_ptr() == x.storage().data_ptr())  # True
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

### 슬라이싱과 저장소 오프셋

슬라이싱은 스트라이드를 보존하고 저장소 오프셋을 조정한다.

```python
x = torch.arange(20).reshape(4, 5)
# 저장소: [0, 1, 2, ..., 19]

y = x[1:3, 2:4]  # 2x2 slice
print(f"Offset: {y.storage_offset()}")  # 7 (position of x[1,2])
print(f"Stride: {y.stride()}")          # (5, 1) - unchanged
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

---

## 7. `view()`와 `reshape()`의 차이

### `view()` - 연속성을 요구한다

```python
t = torch.arange(6).reshape(2, 3)
t_T = t.T  # Non-contiguous

# view()는 비연속 텐서에서 실패한다
try:
    flat = t_T.view(-1)
except RuntimeError as e:
    print(f"Error: view() requires contiguous tensor")
```

### `reshape()` - 언제나 동작한다

```python
# reshape()는 필요하면 복사하여 비연속 텐서를 처리한다
flat = t_T.reshape(-1)  # Works!
print(f"Reshaped: {flat}")

# 다만 복사본이 만들어질 수 있다
print(t.storage().data_ptr() == flat.storage().data_ptr())  # False - a copy was made
```

### 텐서를 연속으로 만들기

```python
t_T_contiguous = t_T.contiguous()
print(f"Now contiguous: {t_T_contiguous.is_contiguous()}")  # True

# 이제 view()가 동작한다
flat_view = t_T_contiguous.view(-1)
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

print(f"Same storage: {original.storage().data_ptr() == cloned.storage().data_ptr()}")
# False - 메모리가 다르다

print(f"Clone requires_grad: {cloned.requires_grad}")  # True
```

### `detach()` - 그래프에서 떼어내기

```python
original = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# detach()는 경사 추적이 없는 뷰를 만든다
detached = original.detach()

print(f"Detached requires_grad: {detached.requires_grad}")  # False
print(f"Same storage: {original.storage().data_ptr() == detached.storage().data_ptr()}")
# True - 같은 메모리이다!
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

!!! warning "제자리 연산과 Autograd"
    제자리 연산이 역전파에 필요한 텐서를 수정하면 경사 계산을 망가뜨릴 수 있다.
    학습 중에는 조심해서 사용해야 한다.

---

## 10. 성능에 미치는 영향

### 연속 접근이 더 빠르다

현대 CPU는 순차적인 메모리 접근에 최적화되어 있다.

```python
import time

def benchmark(x, iterations=1000):
    start = time.time()
    for _ in range(iterations):
        _ = x.sum()
    return time.time() - start

x = torch.randn(1000, 1000)
x_t = x.t()

t_contig = benchmark(x)
t_noncontig = benchmark(x_t)

print(f"Contiguous: {t_contig:.4f}s")
print(f"Non-contiguous: {t_noncontig:.4f}s")
```

### `.contiguous()`를 언제 호출할 것인가

```python
# 필수: 연속된 메모리를 요구하는 연산
x_t = x.t()
x_view = x_t.contiguous().view(-1)

# 선택 사항: 성능 최적화
# 그 텐서에 연산을 많이 수행할 때에만 그렇다
x_fast = x_t.contiguous()

# 불필요하다: reshape가 비연속을 알아서 처리한다
x_flat = x_t.reshape(-1)  # Works without .contiguous()
```

### 효율적인 배치 처리

```python
# 나쁜 예: 작은 텐서를 많이 만든다
batch_bad = [torch.randn(224, 224) for _ in range(32)]

# 좋은 예: 연속된 텐서 하나
batch_good = torch.randn(32, 224, 224)

# 뷰로 개별 이미지에 접근한다(복사 없음)
first_image = batch_good[0]  # View into batch
```

### 불필요한 복사 피하기

```python
# 상황: 데이터 정규화
data = torch.randn(1000, 100)

# 방법 1: 중간 복사본이 생긴다
mean = data.mean(dim=0)
std = data.std(dim=0)
normalized_v1 = (data - mean) / std  # Creates copies

# 방법 2: 제자리(데이터를 수정할 수 있는 경우)
data.sub_(data.mean(dim=0))
data.div_(data.std(dim=0))  # Modifies data in-place
```

---

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
    print(f"  Data pointer: {t.storage().data_ptr()}")
    print()

# 사용 예
x = torch.arange(12).reshape(3, 4)
inspect_tensor(x, "Original")
inspect_tensor(x.T, "Transposed")
inspect_tensor(x[1:], "Sliced")
inspect_tensor(x[:, ::2], "Strided slice")
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

**연습문제 1.**
행 우선(C 연속) 순서로 된 모양 `(2, 3, 4)`의 3차원 텐서가 주어졌을 때 스트라이드를 예측하고 PyTorch로 확인하라.

??? success "연습문제 1 풀이"
    모양 $(d_0, d_1, d_2) = (2, 3, 4)$에 대해 행 우선 순서의 스트라이드는 다음과 같다.

    - 0번 차원의 스트라이드: $d_1 \times d_2 = 3 \times 4 = 12$
    - 1번 차원의 스트라이드: $d_2 = 4$
    - 2번 차원의 스트라이드: $1$

    스트라이드: `(12, 4, 1)`.

    ```python
    import torch
    t = torch.randn(2, 3, 4)
    print(t.stride())  # (12, 4, 1)
    ```

---

**연습문제 2.**
`view()`는 실패하지만 `reshape()`는 성공하는 상황을 만들어라. 메모리 연속성의 관점에서 그 이유를 설명하라.

??? success "연습문제 2 풀이"
    ```python
    import torch
    t = torch.randn(3, 4)
    t_transposed = t.T  # shape (4, 3), strides (1, 4) -- non-contiguous
    # t_transposed.view(12)  # 실행 오류!
    t_reshaped = t_transposed.reshape(12)  # Works -- copies data
    ```
    `view()`는 데이터를 옮기지 않고 모양 메타데이터만 바꾸므로 연속된 메모리를 요구한다. 전치 후에는 스트라이드가 연속인 `(3, 1)`이 아니라 `(1, 4)`이므로 원소들이 행 우선 순서가 아니다. `reshape()`는 이를 감지하여 재구성 전에 연속 복사본을 만든다.

---

**연습문제 3.**
겉보기에는 다르지만 같은 바탕 저장소를 공유하는 두 텐서를 만들어라. 한쪽을 수정하고 다른 쪽에 미치는 영향을 관찰하라.

??? success "연습문제 3 풀이"
    ```python
    import torch
    a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    b = a[::2]   # b = [1, 3, 5], stride = (2,)
    c = a[1:4]   # c = [2, 3, 4], stride = (1,)

    b[0] = 99.0
    print(a)  # tensor([99.,  2.,  3.,  4.,  5.,  6.])
    # 저장소를 공유하므로 b를 수정하면 a도 수정된다
    print(a.data_ptr() == b.data_ptr())  # True
    ```

---

**연습문제 4.**
모양 `(1000, 1000)`인 연속 텐서와 비연속 텐서에서 원소별 곱셈의 성능을 측정하라. 시간 차이를 재고 그 이유를 설명하라.

??? success "연습문제 4 풀이"
    ```python
    import torch, time

    t = torch.randn(1000, 1000)
    t_nc = t.T  # non-contiguous

    # 연속
    start = time.perf_counter()
    for _ in range(100):
        _ = t * t
    print(f"Contiguous: {time.perf_counter() - start:.4f}s")

    # 비연속
    start = time.perf_counter()
    for _ in range(100):
        _ = t_nc * t_nc
    print(f"Non-contiguous: {time.perf_counter() - start:.4f}s")
    ```
    비연속 연산은 순차적인 메모리 접근 패턴을 활용할 수 없으므로 더 느리다. CPU는 연속된 캐시 라인을 미리 가져오는데, 스트라이드 접근은 캐시 미스를 일으킨다.

## 정리하며

| 개념 | 설명 |
|---------|-------------|
| 저장소 | 실제 텐서 데이터를 담는 평평한 1차원 배열 |
| 스트라이드 | 각 차원에 대해 저장소에서 건너뛸 칸 수 |
| 연속 | 메모리의 원소가 행 우선 순회 순서와 일치함 |
| 뷰 | 원래 텐서와 메모리를 공유함 |
| 복사본 | 독립적인 메모리 할당 |
| `view()` | 연속 텐서만 재구성(빠르고 복사 없음) |
| `reshape()` | 임의의 텐서를 재구성(필요하면 복사) |
| `clone()` | 경사 추적을 유지한 독립적인 복사본 생성 |
| `detach()` | 계산 그래프에서 떼어냄(메모리는 공유) |
| `contiguous()` | 텐서를 연속으로 만듦(필요하면 복사) |

스트라이드 기법은 PyTorch가 다차원 텐서 인덱스를 선형 메모리 위치로 대응시키는 방식이다. 스트라이드를 이해하면 왜 어떤 연산은 (메모리를 공유하는) 뷰를 만들고 어떤 연산은 복사본을 필요로 하는지 알 수 있다. 핵심은 다음과 같다.

- 스트라이드는 차원마다 저장소에서 건너뛸 원소의 개수를 결정한다
- 행 우선 순서는 행 스트라이드가 열 스트라이드보다 크다는 뜻이다
- 전치와 permute는 데이터를 복사하지 않고 스트라이드를 바꾼다
- 연속 텐서는 스트라이드가 행 우선 기대와 일치한다
- 비연속 텐서는 특정 연산을 위해 `.contiguous()`가 필요할 수 있다

---
