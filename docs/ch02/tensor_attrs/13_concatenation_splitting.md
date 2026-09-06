# 이어 붙이기와 나누기

이 스크립트는 텐서를 이어 붙이고 나누는 방법을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
#!/usr/bin/env python3
"""
Concatenation, stacking, and splitting operations.

Covers:
- torch.cat: concatenate along existing dimension
- torch.stack: stack along new dimension
- torch.split: split into chunks
- torch.chunk: split into equal pieces
- torch.unbind: unpack along dimension
- torch.hstack, vstack, dstack helpers
- Practical patterns for combining tensors
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
    header("torch.cat: concatenate along existing dimension")
    a = torch.tensor([[1, 2], [3, 4]])
    b = torch.tensor([[5, 6], [7, 8]])
    print("a:\n", a)
    print("b:\n", b)
    
    # 0번 차원(행)으로 이어 붙이기
    cat_dim0 = torch.cat([a, b], dim=0)
    print("cat([a, b], dim=0):\n", cat_dim0)
    print("  Shape:", cat_dim0.shape)  # (4, 2)
    
    # 1번 차원(열)으로 이어 붙이기
    cat_dim1 = torch.cat([a, b], dim=1)
    print("cat([a, b], dim=1):\n", cat_dim1)
    print("  Shape:", cat_dim1.shape)  # (2, 4)

    # -------------------------------------------------------------------------
    header("torch.cat with multiple tensors")
    t1 = torch.tensor([[1], [2]])
    t2 = torch.tensor([[3], [4]])
    t3 = torch.tensor([[5], [6]])
    
    result = torch.cat([t1, t2, t3], dim=1)
    print("cat([t1, t2, t3], dim=1):\n", result)

    # -------------------------------------------------------------------------
    header("torch.cat requires matching dimensions (except cat dim)")
    a = torch.randn(2, 3, 4)
    b = torch.randn(2, 5, 4)  # Different size in dim 1
    
    # 0번이나 2번 차원으로는 이어 붙일 수 있지만(크기가 맞는다) 1번 차원으로는 안 된다
    c = torch.cat([a, b], dim=1)
    print("a.shape:", a.shape)
    print("b.shape:", b.shape)
    print("cat(dim=1).shape:", c.shape)  # (2, 8, 4)
    
    try:
        bad = torch.cat([a, b], dim=0)  # Fails: dim 1 doesn't match
    except RuntimeError as e:
        print("cat(dim=0) fails:", str(e)[:60] + "...")

    # -------------------------------------------------------------------------
    header("torch.stack: stack along NEW dimension")
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])
    c = torch.tensor([7, 8, 9])
    
    # stack은 새 차원을 만든다
    stacked_dim0 = torch.stack([a, b, c], dim=0)
    print("stack([a,b,c], dim=0):\n", stacked_dim0)
    print("  Shape:", stacked_dim0.shape)  # (3, 3)
    
    stacked_dim1 = torch.stack([a, b, c], dim=1)
    print("stack([a,b,c], dim=1):\n", stacked_dim1)
    print("  Shape:", stacked_dim1.shape)  # (3, 3)

    # -------------------------------------------------------------------------
    header("stack vs cat comparison")
    a = torch.randn(3, 4)
    b = torch.randn(3, 4)
    c = torch.randn(3, 4)
    
    # cat: 기존 차원을 따라 이어 붙인다
    cat_result = torch.cat([a, b, c], dim=0)
    print("cat shapes: (3,4) + (3,4) + (3,4) → ", cat_result.shape)
    
    # stack: 새 차원을 추가한다
    stack_result = torch.stack([a, b, c], dim=0)
    print("stack shapes: (3,4) + (3,4) + (3,4) →", stack_result.shape)
    
    # stack은 모든 텐서의 모양이 같을 것을 요구한다
    # cat은 이어 붙이는 차원을 뺀 나머지만 크기가 맞으면 된다

    # -------------------------------------------------------------------------
    header("torch.split: split into specific sizes")
    x = torch.arange(12).reshape(4, 3)
    print("x:\n", x)
    
    # 0번 차원을 따라 크기 2인 덩어리로 나누기
    chunks = torch.split(x, split_size_or_sections=2, dim=0)
    print("split(x, 2, dim=0):")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}:\n{chunk}")
    
    # 서로 다른 크기로 나누기
    chunks2 = torch.split(x, split_size_or_sections=[1, 2, 1], dim=0)
    print("split(x, [1,2,1], dim=0):")
    for i, chunk in enumerate(chunks2):
        print(f"  Chunk {i} shape:", chunk.shape)

    # -------------------------------------------------------------------------
    header("torch.chunk: split into equal chunks")
    x = torch.arange(12)
    print("x:", x)
    
    # 같은 크기의 덩어리 3개로 나누기
    chunks = torch.chunk(x, chunks=3, dim=0)
    print("chunk(x, 3):")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}:", chunk)
    
    # 딱 나누어떨어지지 않으면 마지막 덩어리가 더 작다
    chunks2 = torch.chunk(x, chunks=5, dim=0)
    print("chunk(x, 5):")
    for i, chunk in enumerate(chunks2):
        print(f"  Chunk {i} size:", chunk.shape)

    # -------------------------------------------------------------------------
    header("torch.unbind: unpack along dimension")
    x = torch.tensor([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
    print("x:\n", x)
    
    # 0번 차원을 따라 unbind(1차원 텐서들의 튜플을 반환한다)
    rows = torch.unbind(x, dim=0)
    print("unbind(dim=0):")
    for i, row in enumerate(rows):
        print(f"  Row {i}:", row)
    
    # 1번 차원을 따라 unbind
    cols = torch.unbind(x, dim=1)
    print("unbind(dim=1):")
    for i, col in enumerate(cols):
        print(f"  Col {i}:", col)

    # -------------------------------------------------------------------------
    header("torch.hstack, vstack, dstack helpers")
    a = torch.tensor([[1], [2], [3]])
    b = torch.tensor([[4], [5], [6]])
    
    # hstack: 가로 방향 쌓기(열을 따라)
    h = torch.hstack([a, b])
    print("hstack([a, b]):\n", h)
    print("  Equivalent to cat(dim=1)")
    
    # vstack: 세로 방향 쌓기(행을 따라)
    v = torch.vstack([a.t(), b.t()])
    print("vstack:\n", v)
    print("  Equivalent to cat(dim=0)")
    
    # dstack: 깊이 방향 쌓기(3번째 차원을 따라)
    d = torch.dstack([a, b])
    print("dstack shape:", d.shape)
    print("  Stacks along new dimension 2")

    # -------------------------------------------------------------------------
    header("Practical pattern: batching samples")
    # 개별 표본
    sample1 = torch.randn(3, 224, 224)
    sample2 = torch.randn(3, 224, 224)
    sample3 = torch.randn(3, 224, 224)
    
    # stack으로 배치 만들기
    batch = torch.stack([sample1, sample2, sample3], dim=0)
    print("Individual samples:", sample1.shape)
    print("Stacked batch:", batch.shape)  # (3, 3, 224, 224)

    # -------------------------------------------------------------------------
    header("Practical pattern: sequence concatenation")
    # 길이가 다른 시퀀스들(같은 길이로 채워 넣음)
    seq1 = torch.randn(10, 512)  # 10 timesteps
    seq2 = torch.randn(15, 512)  # 15 timesteps
    
    # 시간 차원을 따라 시퀀스 이어 붙이기
    combined = torch.cat([seq1, seq2], dim=0)
    print("seq1:", seq1.shape)
    print("seq2:", seq2.shape)
    print("Combined sequence:", combined.shape)  # (25, 512)

    # -------------------------------------------------------------------------
    header("Practical pattern: multi-GPU gathering")
    # 여러 GPU의 출력 흉내 내기
    gpu1_out = torch.randn(8, 10)   # Batch of 8
    gpu2_out = torch.randn(8, 10)   # Batch of 8
    gpu3_out = torch.randn(8, 10)   # Batch of 8
    
    # 모든 출력을 모은다
    all_outputs = torch.cat([gpu1_out, gpu2_out, gpu3_out], dim=0)
    print("Per-GPU outputs:", gpu1_out.shape)
    print("All outputs:", all_outputs.shape)  # (24, 10)

    # -------------------------------------------------------------------------
    header("Practical pattern: feature concatenation")
    # 서로 다른 특징 추출기
    text_features = torch.randn(32, 512)   # Text embeddings
    image_features = torch.randn(32, 2048) # Image features
    meta_features = torch.randn(32, 64)    # Metadata
    
    # 모든 특징을 이어 붙인다
    combined = torch.cat([text_features, image_features, meta_features], dim=1)
    print("Text features:", text_features.shape)
    print("Image features:", image_features.shape)
    print("Meta features:", meta_features.shape)
    print("Combined features:", combined.shape)  # (32, 2624)

    # -------------------------------------------------------------------------
    header("split for train/val/test splits")
    data = torch.randn(100, 10)  # 100 samples
    
    # 학습(70%), 검증(15%), 테스트(15%)로 나누기
    train, val, test = torch.split(data, [70, 15, 15], dim=0)
    print("Data:", data.shape)
    print("Train:", train.shape)
    print("Val:", val.shape)
    print("Test:", test.shape)

    # -------------------------------------------------------------------------
    header("Combining cat and split for dynamic batching")
    # 길이가 다른 시퀀스들(이미 채워 넣음)
    seqs = [
        torch.randn(5, 128),   # Length 5
        torch.randn(8, 128),   # Length 8
        torch.randn(3, 128),   # Length 3
        torch.randn(10, 128),  # Length 10
    ]
    lengths = [len(s) for s in seqs]
    
    # 모든 시퀀스를 이어 붙인다
    all_seqs = torch.cat(seqs, dim=0)
    print("Concatenated shape:", all_seqs.shape)  # (26, 128)
    
    # 원래 시퀀스로 다시 나누기
    recovered = torch.split(all_seqs, lengths, dim=0)
    print("Recovered sequences:")
    for i, seq in enumerate(recovered):
        print(f"  Seq {i}: {seq.shape}")

    # -------------------------------------------------------------------------
    header("Stack for time series data")
    # 7일간의 일별 측정값
    measurements = []
    for day in range(7):
        # 하루당: 24시간 × 특징
        measurements.append(torch.randn(24, 5))
    
    # (일, 시간, 특징)으로 쌓기
    time_series = torch.stack(measurements, dim=0)
    print("Daily measurements:", measurements[0].shape)
    print("Time series:", time_series.shape)  # (7, 24, 5)

    # -------------------------------------------------------------------------
    header("meshgrid for creating coordinate grids")
    # 2차원 좌표 격자 만들기
    x = torch.linspace(-1, 1, 5)
    y = torch.linspace(-1, 1, 5)
    
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    print("x:", x)
    print("grid_x:\n", grid_x)
    print("grid_y:\n", grid_y)
    
    # 위치 부호화를 위해 좌표 쌓기
    coords = torch.stack([grid_x, grid_y], dim=-1)
    print("Coordinate pairs shape:", coords.shape)  # (5, 5, 2)

    # -------------------------------------------------------------------------
    header("Efficient batching with list comprehension")
    # cat으로 반복하는 대신
    def slow_batching(tensors):
        result = tensors[0].unsqueeze(0)
        for t in tensors[1:]:
            result = torch.cat([result, t.unsqueeze(0)], dim=0)
        return result
    
    # 더 나은 방법: stack을 바로 쓴다
    def fast_batching(tensors):
        return torch.stack(tensors, dim=0)
    
    tensors = [torch.randn(10) for _ in range(100)]
    
    import time
    start = time.time()
    _ = slow_batching(tensors)
    slow_time = time.time() - start
    
    start = time.time()
    _ = fast_batching(tensors)
    fast_time = time.time() - start
    
    print(f"Slow (iterative cat): {slow_time:.4f}s")
    print(f"Fast (single stack): {fast_time:.4f}s")
    print(f"Speedup: {slow_time/fast_time:.1f}x")

    # -------------------------------------------------------------------------
    header("Quick reference: combining and splitting")
    print("\nCombining tensors:")
    print("  torch.cat(tensors, dim)   - Concatenate along existing dim")
    print("  torch.stack(tensors, dim) - Stack along NEW dimension")
    print("  torch.hstack(tensors)     - Horizontal stack (columns)")
    print("  torch.vstack(tensors)     - Vertical stack (rows)")
    print("  torch.dstack(tensors)     - Depth stack (3rd dimension)")
    
    print("\nSplitting tensors:")
    print("  torch.split(x, size, dim) - Split into chunks of size")
    print("  torch.chunk(x, n, dim)    - Split into n equal parts")
    print("  torch.unbind(x, dim)      - Unpack along dimension")
    
    print("\nKey differences:")
    print("  cat:   Concatenate along existing dim (dims must match)")
    print("  stack: Stack along new dim (ALL shapes must match)")
    print("  split: Specify chunk sizes")
    print("  chunk: Specify number of chunks")
    
    print("\nPerformance tips:")
    print("  - Use stack() instead of iterative cat()")
    print("  - Pre-allocate when possible")
    print("  - unbind() returns tuple (faster than loop + indexing)")

if __name__ == "__main__":
    main()```

## 논의

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
