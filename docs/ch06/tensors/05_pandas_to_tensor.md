# Pandas를 텐서로

Pandas에서 텐서로 가는 길은 늘 NumPy를 지난다. `.values`나 `.to_numpy()`로 배열을 꺼내고 그것을 텐서로 감싼다. 그래서 어려운 자리는 PyTorch 쪽이 아니라 **pandas가 열마다 다른 자료형을 담을 수 있다는 데** 있다. 텐서는 하나만 담으므로 그 사이에서 무엇이 버려지는지 알아야 한다.

## 1. 코드

```python
"""판다스에서 텐서로."""
import torch
import pandas as pd
import numpy as np

# ========================================================================
# 메인
# ========================================================================

def print_info(t):
    # 텐서를 빠르게 살펴보는 함수:
    # - t: dtype을 고려한 형식으로 값을 출력한다
    # - t.shape: 계수/크기. []는 스칼라, [N]은 1차원, [R,C]는 2차원 등이다.
    # - t.dtype: 추론되거나 강제된다. NumPy float64 → 기본적으로 torch.float64
    # - requires_grad: autograd 플래그(실수/복소수에 True로 설정하지 않으면 False)
    print(f"{t = }", f"{t.shape = }", f"{t.dtype = }", f"{t.requires_grad = }", sep="\n", end="\n\n")

def main():
    # --------------------------------------------
    # 1) Pandas Series[int] → 텐서  (**COPY**)
    # --------------------------------------------
    # s.values / s.to_numpy(...) → NumPy 배열. 그다음 torch.tensor(...)는 **복사한다**.
    s1 = pd.Series([1, 2, 3, 4, 5])
    t1 = torch.tensor(s1.values)   # COPY (independent storage)
    print_info(t1)
    # 기댓값: tensor([1, 2, 3, 4, 5])   dtype=torch.int64

    # --------------------------------------------
    # 2) Pandas Series[float] → 텐서  (**COPY**)
    # --------------------------------------------
    # Pandas/NumPy의 기본 실수는 float64이므로 따로 지정하지 않으면 torch.float64가 된다.
    s2 = pd.Series([0.1, 0.2, 0.3])
    t2 = torch.tensor(s2.values)   # COPY (dtype follows NumPy, likely float64)
    print_info(t2)

    # --------------------------------------------
    # 3) 명시적 dtype 변환  (**COPY**)
    # --------------------------------------------
    # 정밀도/성능이 중요하다면 명시적으로 쓰는 것이 모범 사례이다.
    t3 = torch.tensor(s2.values, dtype=torch.float32)  # COPY (float32)
    print_info(t3)

    # --------------------------------------------
    # 4) 불리언 Series → torch.bool 텐서  (**COPY**)
    # --------------------------------------------
    s4 = pd.Series([True, False, True])
    t4 = torch.tensor(s4.values)   # COPY
    print_info(t4)
    # 기댓값: tensor([ True, False,  True])   dtype=torch.bool

    # --------------------------------------------
    # 5) torch.from_numpy를 통한 메모리 공유  (**SHARE**)
    # --------------------------------------------
    # torch.from_numpy(ndarray)는 ndarray와 저장소를 **공유한다**(복사 없음).
    # 어느 쪽을 바꾸어도 다른 쪽에 반영된다(요구조건: 수치형, 쓰기 가능, 지원되는 배치).
    arr = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    s5 = pd.Series(arr)                    # wraps the SAME ndarray (no copy)
    t5 = torch.from_numpy(s5.values)       # SHARE (no copy)
    print_info(t5)

    arr[0] = 99.0   # mutate underlying NumPy array
    print("   NumPy arr after:", arr)
    print("   Tensor after    :", t5)  # reflects change (shared memory)

    # 요령:
    # - 독립성이 필요한가?  t5_ind = torch.from_numpy(s5.values).clone()  # 공유 후 COPY
    # - Series가 쓰기 가능/연속이 아니면 .from_numpy가 오류를 낼 수 있다 → s.to_numpy(..., copy=True)를 쓴다.

    # --------------------------------------------
    # 6) 수치형이 아닌 Series(object dtype) → 오류
    # --------------------------------------------
    try:
        s6 = pd.Series(["a", "b", "c"])
        torch.tensor(s6.values)  # object dtype → ValueError / TypeError
    except Exception as e:
        print("Non-numeric Series error:", e)

    # ---------------------- 참고(COPY / SHARE / 공유 시도) ----------------------
    # • 판다스 → 넘파이:
    #     s.to_numpy(dtype=..., copy=False)     # 바탕 데이터와 SHARE하거나(복사 없음) 뷰를 만들 수 있다
    #     s.values                               # to_numpy()와 같은 개념. 명시적 제어를 원하면 to_numpy를 쓴다
    #
    # • 넘파이 → 토치:
    #     torch.tensor(ndarray)        → **COPY**(항상 새로운 독립 저장소)
    #     torch.from_numpy(ndarray)    → **SHARE**(복사 없음. 변경이 양쪽에 반영된다)
    #     torch.as_tensor(ndarray)     → **공유 시도**(호환되면 공유: 수치형, 쓰기 가능,
    #                                        스트라이드가 지원되면 공유, 아니면 COPY로 되돌아간다)
    #
    # • 파이썬 리스트/튜플을 쓸 때(NumPy가 아닐 때):
    #     torch.tensor(list_like)      → **베낌**
    #     torch.as_tensor(list_like)   → **COPY**(공유할 것이 없다)
    #
    # • Autograd:
    #     새로 만든 텐서는 requires_grad=False이다.
    #     역전파를 할 것이라면 실수/복소수 텐서에 requires_grad=True를 설정한다.
    #
    # • 기기/데이터 클래스:
    #     dtype를 명시하는 편이 좋다(예: 학습에는 float32). 필요하면 장치를 옮긴다:
    #         t = torch.from_numpy(arr).to("cuda")   # CPU에서 먼저 공유한 뒤 GPU로 COPY
    #         t = torch.tensor(df.to_numpy(np.float32), device="cuda")  # GPU로 바로 **COPY**

if __name__ == "__main__":
    main()
```

**출력:**

```
t = tensor([1, 2, 3, 4, 5])
t.shape = torch.Size([5])
t.dtype = torch.int64
t.requires_grad = False

t = tensor([0.1000, 0.2000, 0.3000], dtype=torch.float64)
t.shape = torch.Size([3])
t.dtype = torch.float64
t.requires_grad = False

t = tensor([0.1000, 0.2000, 0.3000])
t.shape = torch.Size([3])
t.dtype = torch.float32
t.requires_grad = False

t = tensor([ True, False,  True])
t.shape = torch.Size([3])
t.dtype = torch.bool
t.requires_grad = False

t = tensor([10., 20., 30.])
t.shape = torch.Size([3])
t.dtype = torch.float32
t.requires_grad = False

   NumPy arr after: [99. 20. 30.]
   Tensor after    : tensor([99., 20., 30.])
Non-numeric Series error: can't convert np.ndarray of type numpy.object_. The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint64, uint32, uint16, uint8, and bool.
```

## 2. 논의

**`DataFrame`은 열마다 자료형을 따로 갖고, 텐서는 하나만 갖는다.** 그래서 `.values`는 프레임 전체를 담을 수 있는 **한 가지** 자료형을 골라야 하고, 거기서 두 가지 일이 일어난다.

| 프레임의 열 | `.values`의 자료형 | 결과 |
|---|---|---|
| `int64` + `float64` | `float64` | 정수 열이 조용히 실수가 된다 |
| `int64` + 문자열 | `object` | 텐서로 만들 수 없다 — `TypeError` |

둘째 줄은 오류가 나므로 금방 안다. **위험한 것은 첫째 줄이다.** 라벨로 쓰려던 정수 열이 실수가 되어도 아무 말이 없다. 열을 따로 꺼내 각각 텐서로 만들면 피할 수 있다.

**pandas의 실수는 `float64`다.** `Series[float]`를 텐서로 만들면 `torch.float64`가 나온다. 그런데 PyTorch의 층들은 `float32`로 만들어지므로, 그대로 넣으면 자료형이 어긋났다는 오류를 만난다. 둘 중 하나로 미리 맞춘다.

```python
torch.tensor(s.values, dtype=torch.float32)      # 텐서로 만들 때 맞춘다
torch.tensor(s.to_numpy(dtype="float32"))        # 넘파이에서 맞춘다
```

**빠진 값은 조용히 지나간다.** `NaN`이 섞인 `Series`는 오류 없이 텐서가 되고, `nan`이 그대로 들어앉는다. 그 뒤로는 `nan`과 더한 모든 값이 `nan`이 되므로 손실이 `nan`으로 변하고, 원인은 한참 뒤에서 드러난다. **텐서로 바꾸기 전에** `isna().any()`로 확인하거나 `fillna()`/`dropna()`로 처리해야 한다. 바꾼 뒤에는 어느 열에서 왔는지 알 수 없다.

!!! warning "대문자 `Int64`는 `int64`와 다르다"
    pandas에는 빠진 값을 담을 수 있는 자료형이 따로 있다. `dtype="Int64"`(대문자 I)가 그것인데, 이때 `.values`는 NumPy 배열이 아니라 `IntegerArray`를 돌려주고 `torch.tensor`는 `RuntimeError: Could not infer dtype of NAType`을 낸다. 소문자 `int64`와 글자 하나 차이라 알아채기 어렵다. `.to_numpy(dtype="float64", na_value=np.nan)`처럼 빠진 값을 무엇으로 바꿀지 밝혀 주어야 넘어간다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`pd.Series([1, 2, 3])`과 `pd.Series([1.0, 2.0, 3.0])`을 각각 텐서로 만들어 `dtype`을 찍어라. 둘째 것의 자료형이 PyTorch 모델에 그대로 쓰기 어려운 까닭을 밝히고, 고치는 방법을 두 가지 보여라.

</div>

??? success "연습문제 1 풀이"
    ```python
    import pandas as pd, torch

    print(torch.tensor(pd.Series([1, 2, 3]).values).dtype)
    print(torch.tensor(pd.Series([1.0, 2.0, 3.0]).values).dtype)
    ```

    ```
    torch.int64
    torch.float64
    ```

    pandas의 실수는 `float64`이고 그것이 그대로 넘어온다. 그런데 `nn.Linear` 같은
    층은 `float32` 가중치로 만들어지므로, `float64` 입력을 넣으면 자료형이 맞지
    않는다는 오류가 난다. 고치는 두 길은 어디서 맞추느냐의 차이뿐이다.

    ```python
    s = pd.Series([1.0, 2.0, 3.0])
    a = torch.tensor(s.values, dtype=torch.float32)     # 텐서로 만들 때
    b = torch.tensor(s.to_numpy(dtype="float32"))       # 넘파이에서
    print(a.dtype, b.dtype)
    ```

    ```
    torch.float32 torch.float32
    ```

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
아래 두 `DataFrame`의 `.values`가 어떤 자료형이 되는지 확인하고, 각각을 텐서로 만들어 보라.

```python
df1 = pd.DataFrame({"a": [1, 2, 3],   "b": [1.5, 2.5, 3.5]})
df2 = pd.DataFrame({"a": [1, 2, 3],   "b": ["x", "y", "z"]})
```

둘 중 하나는 오류가 나고 하나는 조용히 넘어간다. **조용히 넘어가는 쪽이 왜 더 위험한지** 설명하고, 피하는 방법을 보여라.

</div>

??? success "연습문제 2 풀이"
    ```python
    import pandas as pd, torch

    df1 = pd.DataFrame({"a": [1, 2, 3], "b": [1.5, 2.5, 3.5]})
    df2 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    print("df1.values:", df1.values.dtype)
    print("df2.values:", df2.values.dtype)
    try:
        torch.tensor(df2.values)
    except TypeError as e:
        print("df2 ->", type(e).__name__, "-", str(e)[:52])
    ```

    ```
    df1.values: float64
    df2.values: object
    df2 -> TypeError - can't convert np.ndarray of type numpy.object_
    ```

    `.values`는 프레임 전체를 담을 **한 가지** 자료형을 골라야 한다. `df2`는 정수와
    문자열을 함께 담을 수치형이 없으니 `object`가 되고, 텐서는 `object`를 받지 못해
    `TypeError`가 난다.

    `df1`은 오류가 없다. 정수 열이 실수로 올라가 `float64` 하나로 담긴다. **이쪽이 더
    위험하다.** `a`를 부류 라벨로 쓰려 했다면 지금 실수가 되어 있는데 아무도 알려
    주지 않는다. 라벨을 실수로 넘기면 손실 함수가 자료형 오류를 내거나, 더 나쁘게는
    회귀 손실이 조용히 계산된다.

    열마다 자료형이 다르면 **열을 따로 꺼낸다.**

    ```python
    x = torch.tensor(df1["b"].values, dtype=torch.float32)   # 특징
    y = torch.tensor(df1["a"].values)                        # 라벨 — int64로 남는다
    print(x.dtype, y.dtype)
    ```

    ```
    torch.float32 torch.int64
    ```

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
빠진 값이 있는 두 `Series`를 텐서로 만들어 보라.

```python
s1 = pd.Series([1.0, np.nan, 3.0])              # 보통의 float64
s2 = pd.Series([1, 2, None], dtype="Int64")      # 대문자 I
```

하나는 넘어가고 하나는 오류가 난다. 어느 쪽이 더 위험한지 밝히고, `s2`를 텐서로 만드는 방법을 보여라.

</div>

??? success "연습문제 3 풀이"
    ```python
    import pandas as pd, numpy as np, torch

    s1 = pd.Series([1.0, np.nan, 3.0])
    t1 = torch.tensor(s1.values)
    print("s1 ->", t1, torch.isnan(t1).tolist())

    s2 = pd.Series([1, 2, None], dtype="Int64")
    print("s2.values 갈래:", type(s2.values).__name__)
    try:
        torch.tensor(s2.values)
    except RuntimeError as e:
        print("s2 ->", type(e).__name__, "-", e)
    ```

    ```
    s1 -> tensor([1., nan, 3.], dtype=torch.float64) [False, True, False]
    s2.values 갈래: IntegerArray
    s2 -> RuntimeError - Could not infer dtype of NAType
    ```

    `s2`는 오류를 내므로 안전하다. 대문자 `Int64`는 빠진 값을 담을 수 있는 pandas
    고유의 자료형이고, `.values`가 NumPy 배열이 아니라 `IntegerArray`를 돌려주므로
    PyTorch가 `NAType`을 보고 멈춘다. 소문자 `int64`와 글자 하나 차이지만 다른
    자료형이다. 빠진 값을 무엇으로 바꿀지 밝혀 주면 넘어간다.

    ```python
    a = s2.to_numpy(dtype="float64", na_value=np.nan)
    print(a, "->", torch.tensor(a))
    ```

    ```
    [ 1.  2. nan] -> tensor([1., 2., nan], dtype=torch.float64)
    ```

    **`s1`이 더 위험하다.** 아무 말 없이 `nan`을 담은 텐서를 내놓기 때문이다. 그 뒤로
    `nan`과 더하거나 곱한 모든 값이 `nan`이 되므로 손실이 `nan`으로 바뀌는데, 그때는
    이미 어느 열에서 비롯했는지 알 수 없다. 텐서가 된 뒤에는 열 이름이 남아 있지
    않다.

    그러므로 확인은 **바꾸기 전에** pandas 쪽에서 한다.

    ```python
    print("빠진 값이 있는가:", s1.isna().any())
    ```

    ```
    빠진 값이 있는가: True
    ```

## 정리하며

pandas에서 텐서로 가는 길은 NumPy를 지난다. `.values`로 배열을 꺼내고 그것을 감싸므로, 앞 쪽의 복사·공유 규칙이 그대로 적용된다. pandas 쪽에서 새로 생기는 어려움은 셋이다.

- **열마다 자료형이 다르다.** `.values`는 프레임 전체를 담을 하나를 골라야 한다. 정수와 실수가 섞이면 `float64`로 올라가고(조용히), 문자열이 섞이면 `object`가 되어 `TypeError`가 난다(시끄럽게). 자료형이 다른 열은 **따로 꺼낸다.**
- **pandas의 실수는 `float64`다.** 층들은 `float32`이므로 `dtype=torch.float32`나 `to_numpy(dtype="float32")`로 미리 맞춘다.
- **빠진 값은 조용히 지나간다.** `NaN`은 오류 없이 텐서에 들어앉아 뒤에 가서 손실을 `nan`으로 바꾼다. 확인은 텐서가 되기 전, 열 이름이 아직 남아 있을 때 한다.

대문자 `Int64`는 소문자 `int64`와 다른 자료형이고, 이쪽은 오류를 내 준다.

규칙으로 적으면 하나다. **조용히 넘어가는 변환이 오류를 내는 변환보다 위험하다.**
