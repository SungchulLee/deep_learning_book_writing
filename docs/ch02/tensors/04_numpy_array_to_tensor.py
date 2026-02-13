import numpy as np
import torch

def header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def ptr_numpy(a: np.ndarray) -> int:
    """Return the base data pointer of a NumPy array (as a Python int).
    Notes:
      • For a view, this may point into the middle of the base buffer.
      • This address + strides/shape fully describe where elements live.
    """
    return a.__array_interface__['data'][0]

def ptr_torch(t: torch.Tensor) -> int:
    """Return the base data pointer of a PyTorch tensor's storage (as a Python int).
    Notes:
      • Equivalent to C++ Tensor.storage().data_ptr().
      • Two objects “share memory” if they reference the same underlying buffer,
        but their logical first element may start at different offsets (strides).
    """
    return t.untyped_storage().data_ptr()


def main():
    # ------------------------------------------------------------------------------
    # 1) from_numpy: **SHARE** (changes propagate both ways)
    # ------------------------------------------------------------------------------
    header("1) torch.from_numpy(np_array) → SHARE (no copy)")
    # NOTE (Fortran-ordered NumPy):
    # If `arr` is Fortran-ordered (e.g., arr = np.asfortranarray(arr_2d)),
    # torch.from_numpy(arr) STILL **SHARES** memory. The resulting tensor
    # will have Fortran-like (column-major) strides and is typically
    # non-contiguous in PyTorch’s row-major sense:
    #     t_shared.is_contiguous()  # likely False
    #     t_shared.stride()         # shows column-major-style strides
    #
    # Many ops work fine on non-contiguous tensors, but ops that REQUIRE
    # contiguous memory will either:
    #   • internally make a contiguous copy, or
    #   • require you to call:  t_shared = t_shared.contiguous()   # **COPY** (breaks sharing)
    #
    # If you want row-major sharing from the start:
    #     arr_c = np.ascontiguousarray(arr)  # **COPY** if needed
    #     t_shared = torch.from_numpy(arr_c)  # SHARE with row-major layout
    #
    # Reminder: layouts with negative/odd strides (e.g., arr[::-1]) are not supported
    # for sharing; use torch.as_tensor(arr) (may **COPY**) or make a contiguous view first.

    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)   # CPU, dense, writable, contiguous
    t_shared = torch.from_numpy(arr)                    # **SHARE** storage with arr

    print("arr (before):", arr)
    print("t_shared (before):", t_shared)
    print("ptr(arr)    =", ptr_numpy(arr))
    print("ptr(tensor) =", ptr_torch(t_shared), "(same → shared)")

    # Mutating either updates the other (aliasing the same storage):
    arr[0] = 99.0
    print("arr (after arr[0]=99):      ", arr)
    print("t_shared (after arr change):", t_shared)

    t_shared[1] = -7.0
    print("arr (after t_shared[1]=-7): ", arr)
    print("t_shared (after):           ", t_shared)

    # ------------------------------------------------------------------------------
    # 2) as_tensor(np_array): **TRY TO SHARE** (shares if possible; else copies)
    # ------------------------------------------------------------------------------
    header("2) torch.as_tensor(np_array) → TRY TO SHARE (fallback COPY)")
    # as_tensor sharing rules:
    # • C-order (row-major) ndarray, numeric, writable → **SHARE** (no copy).
    # • Fortran-order (column-major) ndarray, numeric, writable → **SHARE** (no copy),
    #   but the resulting tensor is usually non-contiguous in PyTorch’s row-major sense
    #   (strides reflect column-major layout). Many ops work fine; ops that REQUIRE
    #   contiguity will either make an internal copy or require: t_as = t_as.contiguous()  # **COPY**
    # • Will **COPY** if:
    #     - ndarray is read-only, or
    #     - strides/layout are unsupported (e.g., negative strides like arr[::-1]), or
    #     - a dtype/device change is requested:
    #         · as_tensor(ndarray, dtype=...) may COPY to satisfy dtype
    #         · as_tensor(..., device=...) will create on that device → COPY
    #
    # NOTE on from_numpy(ndarray):
    #   - from_numpy **always shares** with the given CPU ndarray (numeric, writable, compatible strides).
    #   - You cannot pass dtype/device to from_numpy.
    #   - If you need a different dtype, first do: arr2 = arr.astype(np.float32, copy=True/False)
    #       then: t = torch.from_numpy(arr2)  # shares with arr2 (arr2 itself may be a copy)
    #   - If you need GPU/MPS: t_cpu = torch.from_numpy(arr); t = t_cpu.to('cuda')  # device move = **COPY**

    arr3 = np.array([1.1, 2.2, 3.3], dtype=np.float64)
    # as_tensor avoids copying when it can: numeric, writable, compatible strides/layout on CPU.
    t_as = torch.as_tensor(arr3)  # usually **SHARE**; may **COPY** if incompatible
    print("arr3 (before):", arr3)
    print("t_as (before): ", t_as)

    arr3[1] = 222.0
    print("arr3 (after arr3[1]=222):", arr3)
    print("t_as (after):            ", t_as)

    print("ptr(arr3)  =", ptr_numpy(arr3))
    print("ptr(t_as)  =", ptr_torch(t_as), "(same → shared; different → copied)")

    # ------------------------------------------------------------------------------
    # 3) tensor(np_array): **COPY** (independent memory)
    # ------------------------------------------------------------------------------
    header("3) torch.tensor(np_array) → COPY (independent)")
    # ---------------------------------------------------------------------------
    # NumPy ndarray → PyTorch Tensor: which constructor to use?
    #
    # 1) torch.tensor(ndarray)  → COPY
    #    • Safest/most defensive: always allocates a new, independent tensor.
    #    • Ignores ndarray’s sharing/strides; no surprises from later NumPy mutations.
    #    • Can set dtype/device directly (e.g., device="cuda").
    #    • Cost: extra allocation and data copy.
    #
    # 2) torch.from_numpy(ndarray)  → SHARE (no copy)
    #    • Zero-copy: tensor shares storage with the CPU NumPy array.
    #    • Requirements: numeric dtype, writable, compatible (usually positive) strides.
    #    • Mutations reflect both ways until you break sharing (e.g., .clone(), .contiguous(), .to('cuda')).
    #    • Cannot pass dtype/device; dtype is derived from ndarray, device is CPU.
    #
    # 3) torch.as_tensor(ndarray)  → TRY TO SHARE (fallback COPY)
    #    • Prefers zero-copy like from_numpy; if incompatible (read-only, negative strides,
    #      dtype/device change needed), it silently makes a COPY.
    #    • Good “share-if-you-can” convenience without manual checks.
    #
    # Rule of thumb:
    #   • Need safety/independence → use torch.tensor(...)
    #   • Need speed/zero-copy and you’re OK with shared-memory caveats → torch.from_numpy(...)
    #   • Want best-effort sharing without fuss → torch.as_tensor(...)
    #
    # Tips:
    #   • Many ops work on non-contiguous tensors; ops requiring contiguity may internally copy
    #     or require: t = t.contiguous()  # COPY (breaks sharing)
    #   • Device move (CPU→CUDA/MPS) always copies, breaking sharing.
    #   • If you must change dtype before sharing: arr2 = arr.astype(np.float32, copy=True/False);
    #     t = torch.from_numpy(arr2)  # shares with arr2 (arr2 itself may be a copy)
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
    # 4) Dtype mappings commonly supported by from_numpy/as_tensor
    # ------------------------------------------------------------------------------
    header("4) Dtype mappings (float32, float64, int64, int32, uint8, bool)")
    # Common NumPy→Torch correspondences on CPU:
    #   float32 → torch.float32     float64 → torch.float64
    #   int64   → torch.int64       int32   → torch.int32
    #   uint8   → torch.uint8       bool_   → torch.bool
    for np_dtype in [np.float32, np.float64, np.int64, np.int32, np.uint8, np.bool_]:
        a = np.array([0, 1, 2], dtype=np_dtype)
        t = torch.from_numpy(a)
        # either of these works:
        print(f"NumPy dtype {a.dtype.name:>8} → Torch dtype {t.dtype}")
        # or
        # print(f"NumPy dtype {str(a.dtype):>8} → Torch dtype {t.dtype}")

    # ------------------------------------------------------------------------------
    # 5) Non-contiguous / strided views (positive step) still **SHARE**
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

    # Mutations reflect across all aliases:
    view[0] = 999.0
    print("After view[0]=999 → base:", base)
    print("After view[0]=999 → t_view:", t_view)

    # ------------------------------------------------------------------------------
    # 6) Read-only NumPy arrays: from_numpy needs writable arrays
    # ------------------------------------------------------------------------------
    header("6) Read-only NumPy arrays → from_numpy may error")

    ro = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ro.setflags(write=False)  # make array read-only
    try:
        _ = torch.from_numpy(ro)  # often errors: cannot write to read-only array
        print("from_numpy(readonly) succeeded (behavior may vary)")
    except Exception as e:
        print("from_numpy(readonly) error:", repr(e))

    # Safe fallback: make a writable copy first (breaking sharing):
    t_ro_copy = torch.from_numpy(np.array(ro, copy=True))  # **COPY**
    print("Fallback via copy:", t_ro_copy)

    # ------------------------------------------------------------------------------
    # 7) Unsupported / tricky dtypes example: complex numbers
    # ------------------------------------------------------------------------------
    header("7) Complex dtype example: may need explicit conversion")
    cplx = np.array([1+2j, 3+4j], dtype=np.complex128)
    try:
        # Depending on versions/builds, direct from_numpy(complex128) may error.
        torch.from_numpy(cplx)  # if unsupported → exception
        print("from_numpy(complex128) succeeded on this setup")
    except Exception as e:
        print("from_numpy(complex128) error:", repr(e))
        # Typical workaround: split into real/imag or cast manually.
        t_real = torch.from_numpy(np.real(cplx).astype(np.float64))  # **SHARE** after astype copy
        t_imag = torch.from_numpy(np.imag(cplx).astype(np.float64))  # **SHARE** after astype copy
        print("Real part tensor:", t_real)
        print("Imag part tensor:", t_imag)

    # ------------------------------------------------------------------------------
    # 8) Quick summary helper: which ones share memory?
    # ------------------------------------------------------------------------------
    header("8) Summary: SHARE → TRY-TO-SHARE → COPY")
    print("from_numpy(np_array)   → **SHARE** (no copy; requires numeric, writable, compatible strides)")
    print("as_tensor(np_array)    → **TRY TO SHARE** (shares if possible; else **COPY**)")
    print("tensor(np_array)       → **COPY** (always independent)")

    # ------------------------------ FYI / Tips ------------------------------
    # • Autograd: Tensors created from NumPy have requires_grad=False by default.
    #   Set requires_grad=True (on float/complex dtypes) if you want gradients.
    # • Device: from_numpy/as_tensor create CPU tensors. Moving to CUDA/MPS causes a **COPY**:
    #       t_cpu = torch.from_numpy(arr)   # SHARE on CPU
    #       t_gpu = t_cpu.to('cuda')        # COPY to GPU (no sharing across frameworks/devices)
    # • Negative/odd strides: Some NumPy views (e.g., reversed arrays a[::-1]) are incompatible
    #   with from_numpy; as_tensor will then **COPY** instead of sharing.
    # • Need independence even after sharing? Use .clone() on the tensor.

if __name__ == "__main__":
    main()