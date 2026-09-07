# Pandas를 텐서로

이 스크립트는 Pandas 데이터를 텐서로 바꾸는 방법을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

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
    main()```

## 논의

CPU 텐서에서 PyTorch와 NumPy의 상호 운용은 매끄럽다. `torch.from_numpy()`는 배열과 메모리를 공유하는 텐서를 만들고, `torch.tensor()`는 항상 복사한다. 어떤 연산이 저장소를 공유하고 어떤 연산이 독립적인 복사본을 만드는지 이해하는 것이 미묘한 버그를 피하는 데 결정적이다.

GPU 가속은 텐서 연산, 특히 신경망 계산을 지배하는 행렬 곱에 대해 몇 자릿수의 속도 향상을 제공한다. `.to(device)`로 텐서와 모델을 GPU로 옮기는 것은 간단하지만, 성능을 유지하려면 CPU-GPU 사이의 데이터 전송을 최소화하는 것이 결정적이다.

## 연습문제

**연습문제 1.**
NumPy 배열을 만들고 `torch.from_numpy()`로 PyTorch 텐서로 변환한 뒤, 원래 배열을 수정하여 텐서도 함께 바뀌는지 확인하라.

??? success "연습문제 1 풀이"
    ```python
    import numpy as np
    arr = np.array([1.0, 2.0, 3.0])
    t = torch.from_numpy(arr)
    arr[0] = 99.0
    print(t)  # tensor([99.,  2.,  3.]) -- shared memory
    ```

---


**연습문제 2.**
`torch.as_tensor()`가 언제 데이터를 복사하고 언제 메모리를 공유하는지 설명하라. 어떤 조건에서 복사가 일어나는가?

??? success "연습문제 2 풀이"
    `torch.as_tensor()`는 입력이 스트라이드가 호환되는 쓰기 가능한 NumPy 배열이고 요청한 dtype/device가 일치할 때 메모리를 공유한다. 배열이 읽기 전용이거나, 스트라이드가 음수이거나, dtype 또는 device 변환이 필요할 때는 복사한다.

---


**연습문제 3.**
`requires_grad=True`인 텐서에 `.numpy()`를 호출하면 오류가 나는 이유는 무엇인가? 올바른 변환 방법을 보여라.

??? success "연습문제 3 풀이"
    ```python
    x = torch.randn(3, requires_grad=True)
    # x.numpy()  # 오류: 경사가 필요한 텐서에는 numpy()를 호출할 수 없다
    x_np = x.detach().cpu().numpy()  # Correct: detach from graph first
    ```

    NumPy에는 autograd 체계가 없으므로, 추적 중인 텐서의 뷰를 노출하면 경사 계산을 망가뜨리는 변경이 일어날 수 있다. `.detach()`는 텐서를 계산 그래프에서 떼어낸다.
