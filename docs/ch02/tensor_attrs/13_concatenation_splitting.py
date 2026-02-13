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
    
    # Concatenate along dim 0 (rows)
    cat_dim0 = torch.cat([a, b], dim=0)
    print("cat([a, b], dim=0):\n", cat_dim0)
    print("  Shape:", cat_dim0.shape)  # (4, 2)
    
    # Concatenate along dim 1 (columns)
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
    
    # Can cat along dim 0 or 2 (matching), not dim 1
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
    
    # Stack creates new dimension
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
    
    # cat: concatenate along existing dimension
    cat_result = torch.cat([a, b, c], dim=0)
    print("cat shapes: (3,4) + (3,4) + (3,4) → ", cat_result.shape)
    
    # stack: add new dimension
    stack_result = torch.stack([a, b, c], dim=0)
    print("stack shapes: (3,4) + (3,4) + (3,4) →", stack_result.shape)
    
    # stack requires ALL tensors to have SAME shape
    # cat only requires matching except in cat dimension

    # -------------------------------------------------------------------------
    header("torch.split: split into specific sizes")
    x = torch.arange(12).reshape(4, 3)
    print("x:\n", x)
    
    # Split into chunks of size 2 along dim 0
    chunks = torch.split(x, split_size_or_sections=2, dim=0)
    print("split(x, 2, dim=0):")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}:\n{chunk}")
    
    # Split into different sizes
    chunks2 = torch.split(x, split_size_or_sections=[1, 2, 1], dim=0)
    print("split(x, [1,2,1], dim=0):")
    for i, chunk in enumerate(chunks2):
        print(f"  Chunk {i} shape:", chunk.shape)

    # -------------------------------------------------------------------------
    header("torch.chunk: split into equal chunks")
    x = torch.arange(12)
    print("x:", x)
    
    # Split into 3 equal chunks
    chunks = torch.chunk(x, chunks=3, dim=0)
    print("chunk(x, 3):")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}:", chunk)
    
    # If not evenly divisible, last chunk is smaller
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
    
    # Unbind along dim 0 (returns tuple of 1D tensors)
    rows = torch.unbind(x, dim=0)
    print("unbind(dim=0):")
    for i, row in enumerate(rows):
        print(f"  Row {i}:", row)
    
    # Unbind along dim 1
    cols = torch.unbind(x, dim=1)
    print("unbind(dim=1):")
    for i, col in enumerate(cols):
        print(f"  Col {i}:", col)

    # -------------------------------------------------------------------------
    header("torch.hstack, vstack, dstack helpers")
    a = torch.tensor([[1], [2], [3]])
    b = torch.tensor([[4], [5], [6]])
    
    # hstack: horizontal stack (along columns)
    h = torch.hstack([a, b])
    print("hstack([a, b]):\n", h)
    print("  Equivalent to cat(dim=1)")
    
    # vstack: vertical stack (along rows)
    v = torch.vstack([a.t(), b.t()])
    print("vstack:\n", v)
    print("  Equivalent to cat(dim=0)")
    
    # dstack: depth stack (along 3rd dimension)
    d = torch.dstack([a, b])
    print("dstack shape:", d.shape)
    print("  Stacks along new dimension 2")

    # -------------------------------------------------------------------------
    header("Practical pattern: batching samples")
    # Individual samples
    sample1 = torch.randn(3, 224, 224)
    sample2 = torch.randn(3, 224, 224)
    sample3 = torch.randn(3, 224, 224)
    
    # Create batch using stack
    batch = torch.stack([sample1, sample2, sample3], dim=0)
    print("Individual samples:", sample1.shape)
    print("Stacked batch:", batch.shape)  # (3, 3, 224, 224)

    # -------------------------------------------------------------------------
    header("Practical pattern: sequence concatenation")
    # Sequences of different lengths (padded to same length)
    seq1 = torch.randn(10, 512)  # 10 timesteps
    seq2 = torch.randn(15, 512)  # 15 timesteps
    
    # Concatenate sequences along time dimension
    combined = torch.cat([seq1, seq2], dim=0)
    print("seq1:", seq1.shape)
    print("seq2:", seq2.shape)
    print("Combined sequence:", combined.shape)  # (25, 512)

    # -------------------------------------------------------------------------
    header("Practical pattern: multi-GPU gathering")
    # Simulate outputs from multiple GPUs
    gpu1_out = torch.randn(8, 10)   # Batch of 8
    gpu2_out = torch.randn(8, 10)   # Batch of 8
    gpu3_out = torch.randn(8, 10)   # Batch of 8
    
    # Gather all outputs
    all_outputs = torch.cat([gpu1_out, gpu2_out, gpu3_out], dim=0)
    print("Per-GPU outputs:", gpu1_out.shape)
    print("All outputs:", all_outputs.shape)  # (24, 10)

    # -------------------------------------------------------------------------
    header("Practical pattern: feature concatenation")
    # Different feature extractors
    text_features = torch.randn(32, 512)   # Text embeddings
    image_features = torch.randn(32, 2048) # Image features
    meta_features = torch.randn(32, 64)    # Metadata
    
    # Concatenate all features
    combined = torch.cat([text_features, image_features, meta_features], dim=1)
    print("Text features:", text_features.shape)
    print("Image features:", image_features.shape)
    print("Meta features:", meta_features.shape)
    print("Combined features:", combined.shape)  # (32, 2624)

    # -------------------------------------------------------------------------
    header("split for train/val/test splits")
    data = torch.randn(100, 10)  # 100 samples
    
    # Split into train (70%), val (15%), test (15%)
    train, val, test = torch.split(data, [70, 15, 15], dim=0)
    print("Data:", data.shape)
    print("Train:", train.shape)
    print("Val:", val.shape)
    print("Test:", test.shape)

    # -------------------------------------------------------------------------
    header("Combining cat and split for dynamic batching")
    # Variable-length sequences (already padded)
    seqs = [
        torch.randn(5, 128),   # Length 5
        torch.randn(8, 128),   # Length 8
        torch.randn(3, 128),   # Length 3
        torch.randn(10, 128),  # Length 10
    ]
    lengths = [len(s) for s in seqs]
    
    # Concatenate all sequences
    all_seqs = torch.cat(seqs, dim=0)
    print("Concatenated shape:", all_seqs.shape)  # (26, 128)
    
    # Split back into original sequences
    recovered = torch.split(all_seqs, lengths, dim=0)
    print("Recovered sequences:")
    for i, seq in enumerate(recovered):
        print(f"  Seq {i}: {seq.shape}")

    # -------------------------------------------------------------------------
    header("Stack for time series data")
    # Daily measurements for 7 days
    measurements = []
    for day in range(7):
        # Each day: 24 hours × features
        measurements.append(torch.randn(24, 5))
    
    # Stack into (days, hours, features)
    time_series = torch.stack(measurements, dim=0)
    print("Daily measurements:", measurements[0].shape)
    print("Time series:", time_series.shape)  # (7, 24, 5)

    # -------------------------------------------------------------------------
    header("meshgrid for creating coordinate grids")
    # Create 2D coordinate grid
    x = torch.linspace(-1, 1, 5)
    y = torch.linspace(-1, 1, 5)
    
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    print("x:", x)
    print("grid_x:\n", grid_x)
    print("grid_y:\n", grid_y)
    
    # Stack coordinates for position encoding
    coords = torch.stack([grid_x, grid_y], dim=-1)
    print("Coordinate pairs shape:", coords.shape)  # (5, 5, 2)

    # -------------------------------------------------------------------------
    header("Efficient batching with list comprehension")
    # Instead of looping with cat
    def slow_batching(tensors):
        result = tensors[0].unsqueeze(0)
        for t in tensors[1:]:
            result = torch.cat([result, t.unsqueeze(0)], dim=0)
        return result
    
    # Better: use stack directly
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
    main()
