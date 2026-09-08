# 재구성과 차원 제거

이 스크립트는 텐서의 재구성과 차원 제거을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
꼴 바꾸기와 차원 다루기 셈.

Covers:
- reshape, view, contiguous 견주기
- squeeze과 unsqueeze
- flatten과 ravel
- transpose과 permute
- movedim과 swapdims
- 차원 더하기와 없애기
- 손에 잡히는 꼴 바꾸기 무늬
"""

import torch

# ========================================================================
# 메인
# ========================================================================

def header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def main():
    torch.manual_seed(42)

    # -------------------------------------------------------------------------
    header("reshape: change shape (may copy if needed)")
    x = torch.arange(12)
    print("x:", x)
    print("x.shape:", x.shape)
    
    # 2차원으로 재구성
    x2d = x.reshape(3, 4)
    print("reshape(3, 4):\n", x2d)
    
    # 3차원으로 재구성
    x3d = x.reshape(2, 2, 3)
    print("reshape(2, 2, 3).shape:", x3d.shape)
    
    # 자동 차원 추론을 위해 -1을 쓴다
    auto = x.reshape(3, -1)  # -1 infers 4
    print("reshape(3, -1).shape:", auto.shape)

    # -------------------------------------------------------------------------
    header("view vs reshape: view requires contiguous memory")
    x = torch.arange(12).reshape(3, 4)
    
    # view는 연속 텐서에서 동작한다
    v = x.view(4, 3)
    print("view(4, 3) works:", v.shape)
    
    # 전치 후에는 연속이 아니다
    xt = x.t()
    print("After transpose, is_contiguous:", xt.is_contiguous())
    
    try:
        v = xt.view(4, 3)  # This will fail!
    except RuntimeError as e:
        print("view() failed (expected):", str(e)[:60] + "...")
    
    # reshape는 동작한다(필요하면 복사본을 만든다)
    r = xt.reshape(4, 3)
    print("reshape() works even when non-contiguous")
    
    # 먼저 연속으로 만들면 그다음에는 view가 동작한다
    v = xt.contiguous().view(4, 3)
    print("After .contiguous(), view works")

    # -------------------------------------------------------------------------
    header("flatten: collapse to 1D")
    x = torch.arange(24).reshape(2, 3, 4)
    print("x.shape:", x.shape)
    
    # 전부 펼치기
    flat = x.flatten()
    print("flatten():", flat.shape)
    
    # 특정 차원만 펼치기
    flat_last = x.flatten(start_dim=1)  # Keep dim 0, flatten rest
    print("flatten(start_dim=1):", flat_last.shape)  # (2, 12)
    
    flat_middle = x.flatten(start_dim=1, end_dim=1)  # Only dim 1
    print("flatten(1, 1):", flat_middle.shape)  # (2, 3, 4) - no change

    # -------------------------------------------------------------------------
    header("squeeze: remove dimensions of size 1")
    x = torch.randn(1, 3, 1, 4, 1)
    print("Original shape:", x.shape)
    
    # 크기 1인 차원을 모두 제거
    squeezed = x.squeeze()
    print("squeeze():", squeezed.shape)  # (3, 4)
    
    # 특정 차원 제거(크기가 1일 때만)
    sq_dim0 = x.squeeze(0)  # Remove dim 0 (size 1)
    print("squeeze(0):", sq_dim0.shape)  # (3, 1, 4, 1)
    
    sq_dim2 = x.squeeze(2)  # Remove dim 2 (size 1)
    print("squeeze(2):", sq_dim2.shape)  # (1, 3, 4, 1)
    
    # 크기가 1보다 크면 제거되지 않는다
    sq_dim1 = x.squeeze(1)  # Dim 1 has size 3
    print("squeeze(1) (size > 1, no change):", sq_dim1.shape)  # (1, 3, 1, 4, 1)

    # -------------------------------------------------------------------------
    header("unsqueeze: add dimension of size 1")
    x = torch.randn(3, 4)
    print("Original shape:", x.shape)
    
    # 0 위치에 차원 추가
    unsq_0 = x.unsqueeze(0)
    print("unsqueeze(0):", unsq_0.shape)  # (1, 3, 4)
    
    # 1 위치에 차원 추가
    unsq_1 = x.unsqueeze(1)
    print("unsqueeze(1):", unsq_1.shape)  # (3, 1, 4)
    
    # 끝에 차원 추가
    unsq_end = x.unsqueeze(-1)
    print("unsqueeze(-1):", unsq_end.shape)  # (3, 4, 1)
    
    # unsqueeze를 연달아 적용
    multi = x.unsqueeze(0).unsqueeze(-1)
    print("unsqueeze(0).unsqueeze(-1):", multi.shape)  # (1, 3, 4, 1)

    # -------------------------------------------------------------------------
    header("Indexing with None adds dimension (same as unsqueeze)")
    x = torch.randn(3, 4)
    print("x.shape:", x.shape)
    
    # 인덱싱에서 None으로 차원 추가
    x_none = x[None, :, :]     # Same as unsqueeze(0)
    print("x[None, :, :]:", x_none.shape)
    
    x_none2 = x[:, None, :]    # Same as unsqueeze(1)
    print("x[:, None, :]:", x_none2.shape)
    
    x_none3 = x[:, :, None]    # Same as unsqueeze(2)
    print("x[:, :, None]:", x_none3.shape)

    # -------------------------------------------------------------------------
    header("transpose: swap two dimensions")
    x = torch.arange(12).reshape(3, 4)
    print("x (3, 4):\n", x)
    
    # 0번과 1번 차원 전치
    xt = x.transpose(0, 1)
    print("transpose(0, 1) (4, 3):\n", xt)
    
    # .T는 2차원 행렬의 축약형
    print("x.T (same as transpose):\n", x.T)
    
    # 참고: transpose는 뷰를 반환한다
    print("Is view:", id(x.storage()) == id(xt.storage()))

    # -------------------------------------------------------------------------
    header("permute: reorder dimensions")
    x = torch.randn(2, 3, 4, 5)
    print("Original shape:", x.shape)
    
    # (4, 2, 5, 3)으로 permute
    perm = x.permute(2, 0, 3, 1)
    print("permute(2, 0, 3, 1):", perm.shape)
    
    # 흔한 패턴: (배치, 높이, 너비, 채널)에서 (배치, 채널, 높이, 너비)로
    img = torch.randn(10, 224, 224, 3)  # HWC format
    img_chw = img.permute(0, 3, 1, 2)    # Convert to CHW
    print("\nImage format conversion:")
    print("  HWC:", img.shape, "→ CHW:", img_chw.shape)

    # -------------------------------------------------------------------------
    header("movedim: move dimensions to new positions")
    x = torch.randn(2, 3, 4, 5)
    print("Original shape:", x.shape)
    
    # 1번 차원을 3번 위치로 옮기기
    moved = torch.movedim(x, 1, 3)
    print("movedim(source=1, destination=3):", moved.shape)
    
    # 여러 차원 옮기기
    moved2 = torch.movedim(x, [0, 1], [2, 3])
    print("movedim([0,1], [2,3]):", moved2.shape)

    # -------------------------------------------------------------------------
    header("swapdims: swap two dimensions (alias for transpose)")
    x = torch.randn(2, 3, 4)
    swapped = torch.swapdims(x, 0, 2)
    print("Original:", x.shape)
    print("swapdims(0, 2):", swapped.shape)

    # -------------------------------------------------------------------------
    header("Reshaping with -1 for auto-inference")
    x = torch.arange(24)
    
    # -1 하나가 남은 크기를 추론한다
    r1 = x.reshape(3, -1)
    print("reshape(3, -1):", r1.shape)  # (3, 8)
    
    r2 = x.reshape(-1, 6)
    print("reshape(-1, 6):", r2.shape)  # (4, 6)
    
    r3 = x.reshape(2, 3, -1)
    print("reshape(2, 3, -1):", r3.shape)  # (2, 3, 4)
    
    # -1을 여러 개 쓸 수는 없다
    try:
        bad = x.reshape(-1, -1)
    except RuntimeError as e:
        print("Multiple -1 fails (expected):", str(e)[:50] + "...")

    # -------------------------------------------------------------------------
    header("Practical pattern: batch processing")
    # 표본 하나에 배치 차원 추가
    sample = torch.randn(3, 224, 224)  # Single image (C, H, W)
    batch = sample.unsqueeze(0)         # Add batch dim
    print("Single sample:", sample.shape)
    print("As batch:", batch.shape)      # (1, 3, 224, 224)
    
    # 처리 후 배치 차원 제거
    result = batch.squeeze(0)
    print("Remove batch:", result.shape)  # (3, 224, 224)

    # -------------------------------------------------------------------------
    header("Practical pattern: sequence to batch")
    # 임베딩 수열: (시퀀스 길이, 임베딩 차원)
    seq = torch.randn(10, 512)
    print("Sequence:", seq.shape)
    
    # 모델 처리를 위해 배치 차원 추가
    batched = seq.unsqueeze(0)  # (1, seq_len, embed_dim)
    print("Batched:", batched.shape)
    
    # 또는 각 시퀀스 원소를 "배치"로 만든다
    batch_of_items = seq.unsqueeze(1)  # (seq_len, 1, embed_dim)
    print("Each item as batch:", batch_of_items.shape)

    # -------------------------------------------------------------------------
    header("Practical pattern: flattening for linear layer")
    # 완전연결 층 앞의 CNN 특징 맵
    features = torch.randn(32, 128, 7, 7)  # (batch, channels, H, W)
    print("CNN features:", features.shape)
    
    # 공간 차원 펼치기
    flat = features.flatten(start_dim=1)  # Keep batch, flatten rest
    print("Flattened:", flat.shape)  # (32, 6272)
    
    # 대안: reshape
    flat2 = features.reshape(32, -1)
    print("Using reshape:", flat2.shape)

    # -------------------------------------------------------------------------
    header("View must preserve total elements")
    x = torch.randn(12)
    
    # 유효한 재구성
    print("12 elements can reshape to:")
    for shape in [(12,), (1, 12), (12, 1), (3, 4), (4, 3), (2, 6), (2, 2, 3)]:
        r = x.reshape(shape)
        print(f"  {shape}: {r.numel()} elements")
    
    # 잘못된 재구성
    try:
        bad = x.reshape(5, 5)  # 25 ≠ 12
    except RuntimeError as e:
        print("reshape(5, 5) fails:", str(e)[:50] + "...")

    # -------------------------------------------------------------------------
    header("Understanding contiguity with strides")
    x = torch.arange(12).reshape(3, 4)
    print("x (contiguous):\n", x)
    print("  stride:", x.stride())
    print("  is_contiguous:", x.is_contiguous())
    
    # 전치는 스트라이드를 바꾸지만 메모리는 공유한다
    xt = x.t()
    print("\nx.t() (non-contiguous):\n", xt)
    print("  stride:", xt.stride())
    print("  is_contiguous:", xt.is_contiguous())
    
    # 연속으로 만들기(복사본이 생긴다)
    xt_contig = xt.contiguous()
    print("\nAfter .contiguous():")
    print("  stride:", xt_contig.stride())
    print("  is_contiguous:", xt_contig.is_contiguous())

    # -------------------------------------------------------------------------
    header("Combining operations: common workflow")
    # (배치, 시퀀스 길이, 은닉 차원)으로 시작
    x = torch.randn(8, 20, 256)
    print("Input:", x.shape)
    
    # 어텐션을 위한 전치: (시퀀스 길이, 배치, 은닉 차원)
    x = x.transpose(0, 1)
    print("After transpose:", x.shape)
    
    # 멀티헤드를 위한 재구성: (시퀀스 길이, 배치, 헤드 수, 헤드 차원)
    n_heads = 8
    head_dim = 256 // n_heads
    x = x.reshape(20, 8, n_heads, head_dim)
    print("After reshape for heads:", x.shape)
    
    # 계산을 위한 permute: (배치, 헤드 수, 시퀀스 길이, 헤드 차원)
    x = x.permute(1, 2, 0, 3)
    print("After permute:", x.shape)

    # -------------------------------------------------------------------------
    header("Quick reference: reshaping operations")
    print("\nShape changes:")
    print("  .reshape(shape)   - Change shape (may copy)")
    print("  .view(shape)      - Change shape (must be contiguous)")
    print("  .flatten()        - Collapse to 1D")
    print("  .flatten(start, end) - Collapse specific dims")
    
    print("\nDimension manipulation:")
    print("  .squeeze()        - Remove size-1 dimensions")
    print("  .squeeze(dim)     - Remove specific size-1 dim")
    print("  .unsqueeze(dim)   - Add size-1 dimension")
    print("  x[None, ...]      - Add dimension via indexing")
    
    print("\nReordering dimensions:")
    print("  .transpose(d1, d2) - Swap two dimensions")
    print("  .T                 - Transpose (2D only)")
    print("  .permute(dims)     - Arbitrary reordering")
    print("  .movedim(src, dst) - Move dimension(s)")
    print("  .swapdims(d1, d2)  - Swap dimensions")
    
    print("\nMemory layout:")
    print("  .contiguous()     - Create contiguous copy if needed")
    print("  .is_contiguous()  - Check if contiguous")
    
    print("\nTips:")
    print("  - Use -1 in reshape to infer dimension")
    print("  - view requires contiguous, reshape doesn't")
    print("  - Most reshaping ops return views (no copy)")
    print("  - transpose/permute change strides, may need .contiguous()")

if __name__ == "__main__":
    main()
```

## 2. 논의

재구성 연산은 데이터를 반드시 복사하지 않으면서 텐서의 논리적 배치를 바꾼다. `view()` 메서드는 연속된 메모리를 요구하며 항상 뷰를 반환하고, `reshape()`는 (필요하면 복사하여) 어떤 텐서에서도 동작한다. `transpose()`나 `permute()` 같은 연산은 데이터 배치가 아니라 스트라이드를 바꾸므로 결과가 연속적이지 않을 수 있다.

PyTorch는 (저장소를 공유하는 뷰를 반환하는) 기본 슬라이싱과 (복사본을 반환하는) 불리언 마스크나 정수 배열을 이용한 고급 인덱싱을 모두 지원한다. 이 구분을 이해하는 것은 메모리 효율을 위해서도, 인덱싱한 결과를 수정할 때 의도치 않은 부작용을 피하기 위해서도 중요하다.

텐서 생성 함수는 데이터를 초기화하는 유연한 방법을 제공한다. `torch.zeros`, `torch.randn`, `torch.arange` 같은 팩토리 함수는 `dtype`, `device`, `requires_grad` 매개변수를 받으므로 불필요한 복사 없이 목표 장치에 곧바로 할당할 수 있다.

## 연습문제

**연습문제 1.**
모양이 $(24,)$인 텐서를 만들어 $(2, 3, 4)$로 재구성하라. 그런 다음 차원을 $(4, 2, 3)$으로 바꾸고 전체 원소 개수가 변하지 않았음을 확인하라.

??? success "연습문제 1 풀이"
    ```python
    x = torch.arange(24)
    x_3d = x.reshape(2, 3, 4)
    x_perm = x_3d.permute(2, 0, 1)
    assert x_perm.shape == (4, 2, 3)
    assert x_perm.numel() == 24
    ```

---


**연습문제 2.**
`transpose()` 후에 `view()`는 실패하지만 `reshape()`는 성공하는 이유를 설명하라. `contiguous()`는 무엇을 하는가?

??? success "연습문제 2 풀이"
    `transpose()`는 스트라이드는 다르지만 바탕 저장소는 같은 뷰를 반환한다. 데이터가 더 이상 행 우선 순서가 아니므로 (연속된 메모리를 요구하는) `view()`가 실패한다. `reshape()`는 비연속성을 감지하여 복사본을 만든다. `.contiguous()`를 호출하면 행 우선 배치를 가진 새 텐서가 명시적으로 만들어지며, 그 뒤에는 `view()`가 동작한다.

---


**연습문제 3.**
NCHW 형식의 모양 $(32, 3, 224, 224)$인 이미지 배치가 주어졌을 때 `permute()`로 NHWC 형식으로 바꾸고, 공간 차원을 펼쳐 모양 $(32, 3, 50176)$을 얻어라.

??? success "연습문제 3 풀이"
    ```python
    images = torch.randn(32, 3, 224, 224)
    nhwc = images.permute(0, 2, 3, 1)  # (32, 224, 224, 3)
    # (32, 3, 50176)을 얻으려면 원본에서 H와 W를 펼친다:
    flat = images.flatten(start_dim=2)  # (32, 3, 50176)
    print(flat.shape)
    ```

## 정리하며

**다룬 것** — 재구성과 차원 제거

재구성 연산은 데이터를 반드시 복사하지 않으면서 텐서의 논리적 배치를 바꾼다.

앞의 연습문제 3개로 직접 확인할 수 있다.
