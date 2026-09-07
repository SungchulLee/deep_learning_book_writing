# NumPy 배열을 텐서로

이 스크립트는 NumPy 배열을 텐서로 바꾸는 방법을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
"""넘파이 배열에서 텐서로."""
import numpy as np
import torch

# ========================================================================
# 메인
# ========================================================================

def header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def ptr_numpy(a: np.ndarray) -> int:
    """넘파이 배열의 밑 자료 가리개를 (파이썬 정수로) 돌려준다.
    Notes:
      • 보기라면 밑 버퍼의 가운데를 가리킬 수도 있다.
      • 이 주소와 걸음/꼴이 원소가 어디에 있는지를 온전히 밝힌다.
    """
    return a.__array_interface__['data'][0]

def ptr_torch(t: torch.Tensor) -> int:
    """PyTorch 텐서 저장소의 밑 자료 가리개를 (파이썬 정수로) 돌려준다.
    Notes:
      • C++의 Tensor.storage().data_ptr()과 같다.
      • 두 객체가 같은 밑 버퍼를 가리키면 "기억 자리를 나눠 쓴다"고 하지만,
        논리상 첫 원소는 서로 다른 자리(걸음)에서 비롯할 수 있다.
    """
    return t.untyped_storage().data_ptr()


def main():
    # ------------------------------------------------------------------------------
    # 1) from_numpy: **SHARE**(변경이 양쪽으로 전파된다)
    # ------------------------------------------------------------------------------
    header("1) torch.from_numpy(np_array) → SHARE (no copy)")
    # 참고(포트란 순서 NumPy):
    # `arr`이 포트란 순서라면(예: arr = np.asfortranarray(arr_2d)),
    # torch.from_numpy(arr)는 여전히 메모리를 **공유한다**. 결과 텐서는
    # 포트란 방식(열 우선) 스트라이드를 가지며 대개
    # PyTorch의 행 우선 기준으로는 비연속적이다:
    #     t_shared.is_contiguous()  # 아마 False
    #     t_shared.stride()         # 열 우선 방식의 스트라이드를 보여준다
    #
    # 많은 연산이 비연속 텐서에서도 잘 동작하지만, 연속성을 요구하는 연산은
    # 다음 중 하나를 한다:
    #   • 내부적으로 연속 복사본을 만들거나,
    #   • 다음 호출을 요구한다:  t_shared = t_shared.contiguous()   # **COPY**(공유가 끊긴다)
    #
    # 처음부터 행 우선으로 공유하고 싶다면:
    #     arr_c = np.ascontiguousarray(arr)  # 필요하면 **COPY**
    #     t_shared = torch.from_numpy(arr_c)  # 행 우선 배치로 SHARE
    #
    # 참고: 음수/특이한 스트라이드를 가진 배치(예: arr[::-1])는 지원되지 않는다
    # 공유에는 쓸 수 없다. torch.as_tensor(arr)를 쓰거나(**COPY**할 수 있다) 먼저 연속 뷰를 만든다.

    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)   # CPU, dense, writable, contiguous
    t_shared = torch.from_numpy(arr)                    # **SHARE** storage with arr

    print("arr (before):", arr)
    print("t_shared (before):", t_shared)
    print("ptr(arr)    =", ptr_numpy(arr))
    print("ptr(tensor) =", ptr_torch(t_shared), "(same → shared)")

    # 어느 쪽을 바꾸어도 다른 쪽이 갱신된다(같은 저장소를 가리킨다):
    arr[0] = 99.0
    print("arr (after arr[0]=99):      ", arr)
    print("t_shared (after arr change):", t_shared)

    t_shared[1] = -7.0
    print("arr (after t_shared[1]=-7): ", arr)
    print("t_shared (after):           ", t_shared)

    # ------------------------------------------------------------------------------
    # 2) as_tensor(np_array): **공유 시도**(가능하면 공유, 아니면 복사)
    # ------------------------------------------------------------------------------
    header("2) torch.as_tensor(np_array) → TRY TO SHARE (fallback COPY)")
    # as_tensor의 공유 규칙:
    # • C 순서(행 우선) ndarray이고 수치형이며 쓰기 가능하면 → **SHARE**(복사 없음).
    # • 포트란 순서(열 우선) ndarray이고 수치형이며 쓰기 가능하면 → **SHARE**(복사 없음),
    #   그러나 결과 텐서는 대개 PyTorch의 행 우선 기준으로 비연속적이다
    #   (스트라이드가 열 우선 배치를 반영한다). 많은 연산이 잘 동작하지만, 연속성을
    #   요구하는 연산은 내부적으로 복사하거나 t_as = t_as.contiguous()를 요구한다  # **COPY**
    # • 다음 경우에는 **COPY**한다:
    #     - ndarray가 읽기 전용이거나,
    #     - 스트라이드/배치가 지원되지 않거나(예: arr[::-1] 같은 음수 스트라이드),
    #     - dtype/device 변경이 요청된 경우:
    #         · as_tensor(ndarray, dtype=...)는 dtype을 맞추려고 COPY할 수 있다
    #         · as_tensor(..., device=...)는 해당 장치에 생성한다 → COPY
    #
    # from_numpy(ndarray)에 대한 참고:
    #   - from_numpy는 주어진 CPU ndarray와 **항상 공유한다**(수치형, 쓰기 가능, 호환 스트라이드).
    #   - from_numpy에는 dtype/device를 넘길 수 없다.
    #   - 다른 dtype이 필요하면 먼저: arr2 = arr.astype(np.float32, copy=True/False)
    #       그다음: t = torch.from_numpy(arr2)  # arr2와 공유한다(arr2 자체는 복사본일 수 있다)
    #   - GPU/MPS가 필요하면: t_cpu = torch.from_numpy(arr); t = t_cpu.to('cuda')  # 장치 이동 = **COPY**

    arr3 = np.array([1.1, 2.2, 3.3], dtype=np.float64)
    # as_tensor는 가능하면 복사를 피한다. CPU에서 수치형, 쓰기 가능, 호환되는 스트라이드/배치일 때이다.
    t_as = torch.as_tensor(arr3)  # usually **SHARE**; may **COPY** if incompatible
    print("arr3 (before):", arr3)
    print("t_as (before): ", t_as)

    arr3[1] = 222.0
    print("arr3 (after arr3[1]=222):", arr3)
    print("t_as (after):            ", t_as)

    print("ptr(arr3)  =", ptr_numpy(arr3))
    print("ptr(t_as)  =", ptr_torch(t_as), "(same → shared; different → copied)")

    # ------------------------------------------------------------------------------
    # 3) tensor(np_array): **COPY**(독립적인 메모리)
    # ------------------------------------------------------------------------------
    header("3) torch.tensor(np_array) → COPY (independent)")
    # ---------------------------------------------------------------------------
    # NumPy ndarray → PyTorch 텐서: 어떤 생성자를 쓸 것인가?
    #
    # 1) torch.tensor(ndarray)  → 베낌
    #    • 가장 안전하고 방어적이다. 항상 새로운 독립 텐서를 할당한다.
    #    • ndarray의 공유/스트라이드를 무시한다. 이후 NumPy 변경으로 놀랄 일이 없다.
    #    • dtype/device를 직접 지정할 수 있다(예: device="cuda").
    #    • 비용: 추가 할당과 데이터 복사.
    #
    # 2) torch.from_numpy(ndarray)  → SHARE(복사 없음)
    #    • 무복사: 텐서가 CPU NumPy 배열과 저장소를 공유한다.
    #    • 요구조건: 수치형 dtype, 쓰기 가능, 호환되는(보통 양수) 스트라이드.
    #    • 공유를 끊기 전까지는(예: .clone(), .contiguous(), .to('cuda')) 변경이 양쪽에 반영된다.
    #    • dtype/device를 넘길 수 없다. dtype은 ndarray에서 얻고 device는 CPU이다.
    #
    # 3) torch.as_tensor(ndarray)  → 공유 시도(안 되면 COPY)
    #    • from_numpy처럼 무복사를 선호한다. 호환되지 않으면(읽기 전용, 음수 스트라이드,
    #      dtype/device 변경이 필요하면) 조용히 COPY를 만든다.
    #    • 수동 확인 없이 "가능하면 공유"를 해 주는 편리한 선택.
    #
    # 어림 규칙:
    #   • 안전성/독립성이 필요하면 → torch.tensor(...)를 쓴다
    #   • 속도/무복사가 필요하고 공유 메모리의 주의사항을 감수할 수 있으면 → torch.from_numpy(...)
    #   • 번거로움 없이 최대한 공유하고 싶으면 → torch.as_tensor(...)
    #
    # 요령:
    #   • 많은 연산이 비연속 텐서에서 동작한다. 연속성을 요구하는 연산은 내부적으로 복사하거나
    #     또는 t = t.contiguous()가 필요하다  # COPY(공유가 끊긴다)
    #   • 장치 이동(CPU→CUDA/MPS)은 항상 복사하며 공유를 끊는다.
    #   • 공유 전에 dtype을 바꿔야 한다면: arr2 = arr.astype(np.float32, copy=True/False);
    #     t = torch.from_numpy(arr2)  # arr2와 공유한다(arr2 자체는 복사본일 수 있다)
    # ---------------------------------------------------------------------------

    arr2 = np.array([10, 20, 30], dtype=np.int64)
    t_copy = torch.tensor(arr2)  # **COPY** from NumPy → independent buffer
    print("arr2 (before):", arr2)
    print("t_copy (before):", t_copy)

    arr2[0] = 123
    print("arr2 (after arr2[0]=123):", arr2)
    print("t_copy (unchanged):       ", t_copy)  # separate storage

    print("ptr(arr2)   =", ptr_numpy(arr2))
    print("ptr(t_copy) =", ptr_torch(t_copy), "(different → copy)")

    # ------------------------------------------------------------------------------
    # 4) from_numpy/as_tensor가 흔히 지원하는 dtype 대응
    # ------------------------------------------------------------------------------
    header("4) Dtype mappings (float32, float64, int64, int32, uint8, bool)")
    # CPU에서의 흔한 NumPy→Torch 대응:
    #   float32 → torch.float32     float64 → torch.float64
    #   int64   → torch.int64       int32   → torch.int32
    #   uint8   → torch.uint8       bool_   → torch.bool
    for np_dtype in [np.float32, np.float64, np.int64, np.int32, np.uint8, np.bool_]:
        a = np.array([0, 1, 2], dtype=np_dtype)
        t = torch.from_numpy(a)
        # 다음 중 어느 쪽이든 동작한다:
        print(f"NumPy dtype {a.dtype.name:>8} → Torch dtype {t.dtype}")
        # 또는
        # print(f"넘파이 dtype {str(a.dtype):>8} → 토치 dtype {t.dtype}")

    # ------------------------------------------------------------------------------
    # 5) 비연속/스트라이드 뷰(양수 보폭)도 여전히 **공유한다**
    # ------------------------------------------------------------------------------
    header("5) Strided NumPy views (positive step) → SHARE")

    base = np.arange(10, dtype=np.float32)     # [0,1,2,3,4,5,6,7,8,9]
    view = base[::2]                           # [0,2,4,6,8] (non-contiguous view)
    t_view = torch.from_numpy(view)            # **SHARE** with view (and base)

    print("base:", base)
    print("view:", view)
    print("t_view:", t_view)
    print("ptr(base)  =", ptr_numpy(base))
    print("ptr(view)  =", ptr_numpy(view))     # pointer may point into base’s buffer
    print("ptr(t_view) =", ptr_torch(t_view), "(same as view → shared)")

    # 변경이 모든 별칭에 반영된다:
    view[0] = 999.0
    print("After view[0]=999 → base:", base)
    print("After view[0]=999 → t_view:", t_view)

    # ------------------------------------------------------------------------------
    # 6) 읽기 전용 NumPy 배열: from_numpy는 쓰기 가능한 배열을 필요로 한다
    # ------------------------------------------------------------------------------
    header("6) Read-only NumPy arrays → from_numpy may error")

    ro = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ro.setflags(write=False)  # make array read-only
    try:
        _ = torch.from_numpy(ro)  # often errors: cannot write to read-only array
        print("from_numpy(readonly) succeeded (behavior may vary)")
    except Exception as e:
        print("from_numpy(readonly) error:", repr(e))

    # 안전한 대안: 먼저 쓰기 가능한 복사본을 만든다(공유가 끊긴다):
    t_ro_copy = torch.from_numpy(np.array(ro, copy=True))  # **COPY**
    print("Fallback via copy:", t_ro_copy)

    # ------------------------------------------------------------------------------
    # 7) 지원되지 않거나 까다로운 dtype의 예: 복소수
    # ------------------------------------------------------------------------------
    header("7) Complex dtype example: may need explicit conversion")
    cplx = np.array([1+2j, 3+4j], dtype=np.complex128)
    try:
        # 버전/빌드에 따라 from_numpy(complex128)을 바로 쓰면 오류가 날 수 있다.
        torch.from_numpy(cplx)  # if unsupported → exception
        print("from_numpy(complex128) succeeded on this setup")
    except Exception as e:
        print("from_numpy(complex128) error:", repr(e))
        # 흔한 우회책: 실수부/허수부로 나누거나 직접 변환한다.
        t_real = torch.from_numpy(np.real(cplx).astype(np.float64))  # **SHARE** after astype copy
        t_imag = torch.from_numpy(np.imag(cplx).astype(np.float64))  # **SHARE** after astype copy
        print("Real part tensor:", t_real)
        print("Imag part tensor:", t_imag)

    # ------------------------------------------------------------------------------
    # 8) 간단 요약 도우미: 어떤 것이 메모리를 공유하는가?
    # ------------------------------------------------------------------------------
    header("8) Summary: SHARE → TRY-TO-SHARE → COPY")
    print("from_numpy(np_array)   → **SHARE** (no copy; requires numeric, writable, compatible strides)")
    print("as_tensor(np_array)    → **TRY TO SHARE** (shares if possible; else **COPY**)")
    print("tensor(np_array)       → **COPY** (always independent)")

    # ------------------------------ 참고 / 요령 ------------------------------
    # • Autograd: NumPy에서 만든 텐서는 기본적으로 requires_grad=False이다.
    #   경사가 필요하면 (실수/복소수 dtype에) requires_grad=True를 설정한다.
    # • Device: from_numpy/as_tensor는 CPU 텐서를 만든다. CUDA/MPS로 옮기면 **COPY**가 일어난다:
    #       t_cpu = torch.from_numpy(arr)   # CPU에서 SHARE
    #       t_gpu = t_cpu.to('cuda')        # GPU로 COPY(프레임워크/장치를 넘어선 공유는 없다)
    # • 음수/특이한 스트라이드: 일부 NumPy 뷰(예: 뒤집힌 배열 a[::-1])는 호환되지 않아
    #   from_numpy와 함께 쓴다. 그러면 as_tensor는 공유 대신 **COPY**한다.
    # • 공유한 뒤에도 독립성이 필요한가? 텐서에 .clone()을 쓴다.

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
