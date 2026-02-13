#!/usr/bin/env python3
"""
Reshaping and dimension manipulation operations.

Covers:
- reshape vs view vs contiguous
- squeeze and unsqueeze
- flatten and ravel
- transpose and permute
- movedim and swapdims
- Adding and removing dimensions
- Practical reshaping patterns
"""

import torch

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
    
    # Reshape to 2D
    x2d = x.reshape(3, 4)
    print("reshape(3, 4):\n", x2d)
    
    # Reshape to 3D
    x3d = x.reshape(2, 2, 3)
    print("reshape(2, 2, 3).shape:", x3d.shape)
    
    # Use -1 for automatic dimension inference
    auto = x.reshape(3, -1)  # -1 infers 4
    print("reshape(3, -1).shape:", auto.shape)

    # -------------------------------------------------------------------------
    header("view vs reshape: view requires contiguous memory")
    x = torch.arange(12).reshape(3, 4)
    
    # view works on contiguous tensor
    v = x.view(4, 3)
    print("view(4, 3) works:", v.shape)
    
    # After transpose, not contiguous
    xt = x.t()
    print("After transpose, is_contiguous:", xt.is_contiguous())
    
    try:
        v = xt.view(4, 3)  # This will fail!
    except RuntimeError as e:
        print("view() failed (expected):", str(e)[:60] + "...")
    
    # reshape works (makes copy if needed)
    r = xt.reshape(4, 3)
    print("reshape() works even when non-contiguous")
    
    # Make contiguous first, then view works
    v = xt.contiguous().view(4, 3)
    print("After .contiguous(), view works")

    # -------------------------------------------------------------------------
    header("flatten: collapse to 1D")
    x = torch.arange(24).reshape(2, 3, 4)
    print("x.shape:", x.shape)
    
    # Flatten everything
    flat = x.flatten()
    print("flatten():", flat.shape)
    
    # Flatten only specific dimensions
    flat_last = x.flatten(start_dim=1)  # Keep dim 0, flatten rest
    print("flatten(start_dim=1):", flat_last.shape)  # (2, 12)
    
    flat_middle = x.flatten(start_dim=1, end_dim=1)  # Only dim 1
    print("flatten(1, 1):", flat_middle.shape)  # (2, 3, 4) - no change

    # -------------------------------------------------------------------------
    header("squeeze: remove dimensions of size 1")
    x = torch.randn(1, 3, 1, 4, 1)
    print("Original shape:", x.shape)
    
    # Remove all size-1 dimensions
    squeezed = x.squeeze()
    print("squeeze():", squeezed.shape)  # (3, 4)
    
    # Remove specific dimension (only if size 1)
    sq_dim0 = x.squeeze(0)  # Remove dim 0 (size 1)
    print("squeeze(0):", sq_dim0.shape)  # (3, 1, 4, 1)
    
    sq_dim2 = x.squeeze(2)  # Remove dim 2 (size 1)
    print("squeeze(2):", sq_dim2.shape)  # (1, 3, 4, 1)
    
    # Won't remove if size > 1
    sq_dim1 = x.squeeze(1)  # Dim 1 has size 3
    print("squeeze(1) (size > 1, no change):", sq_dim1.shape)  # (1, 3, 1, 4, 1)

    # -------------------------------------------------------------------------
    header("unsqueeze: add dimension of size 1")
    x = torch.randn(3, 4)
    print("Original shape:", x.shape)
    
    # Add dimension at position 0
    unsq_0 = x.unsqueeze(0)
    print("unsqueeze(0):", unsq_0.shape)  # (1, 3, 4)
    
    # Add dimension at position 1
    unsq_1 = x.unsqueeze(1)
    print("unsqueeze(1):", unsq_1.shape)  # (3, 1, 4)
    
    # Add dimension at end
    unsq_end = x.unsqueeze(-1)
    print("unsqueeze(-1):", unsq_end.shape)  # (3, 4, 1)
    
    # Chain multiple unsqueezes
    multi = x.unsqueeze(0).unsqueeze(-1)
    print("unsqueeze(0).unsqueeze(-1):", multi.shape)  # (1, 3, 4, 1)

    # -------------------------------------------------------------------------
    header("Indexing with None adds dimension (same as unsqueeze)")
    x = torch.randn(3, 4)
    print("x.shape:", x.shape)
    
    # Add dimensions using None in indexing
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
    
    # Transpose dims 0 and 1
    xt = x.transpose(0, 1)
    print("transpose(0, 1) (4, 3):\n", xt)
    
    # .T shorthand for 2D matrices
    print("x.T (same as transpose):\n", x.T)
    
    # Note: transpose returns a view
    print("Is view:", id(x.storage()) == id(xt.storage()))

    # -------------------------------------------------------------------------
    header("permute: reorder dimensions")
    x = torch.randn(2, 3, 4, 5)
    print("Original shape:", x.shape)
    
    # Permute to (4, 2, 5, 3)
    perm = x.permute(2, 0, 3, 1)
    print("permute(2, 0, 3, 1):", perm.shape)
    
    # Common pattern: (batch, height, width, channels) to (batch, channels, height, width)
    img = torch.randn(10, 224, 224, 3)  # HWC format
    img_chw = img.permute(0, 3, 1, 2)    # Convert to CHW
    print("\nImage format conversion:")
    print("  HWC:", img.shape, "→ CHW:", img_chw.shape)

    # -------------------------------------------------------------------------
    header("movedim: move dimensions to new positions")
    x = torch.randn(2, 3, 4, 5)
    print("Original shape:", x.shape)
    
    # Move dim 1 to position 3
    moved = torch.movedim(x, 1, 3)
    print("movedim(source=1, destination=3):", moved.shape)
    
    # Move multiple dimensions
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
    
    # Single -1 infers remaining size
    r1 = x.reshape(3, -1)
    print("reshape(3, -1):", r1.shape)  # (3, 8)
    
    r2 = x.reshape(-1, 6)
    print("reshape(-1, 6):", r2.shape)  # (4, 6)
    
    r3 = x.reshape(2, 3, -1)
    print("reshape(2, 3, -1):", r3.shape)  # (2, 3, 4)
    
    # Can't have multiple -1s
    try:
        bad = x.reshape(-1, -1)
    except RuntimeError as e:
        print("Multiple -1 fails (expected):", str(e)[:50] + "...")

    # -------------------------------------------------------------------------
    header("Practical pattern: batch processing")
    # Add batch dimension to single sample
    sample = torch.randn(3, 224, 224)  # Single image (C, H, W)
    batch = sample.unsqueeze(0)         # Add batch dim
    print("Single sample:", sample.shape)
    print("As batch:", batch.shape)      # (1, 3, 224, 224)
    
    # Remove batch dimension after processing
    result = batch.squeeze(0)
    print("Remove batch:", result.shape)  # (3, 224, 224)

    # -------------------------------------------------------------------------
    header("Practical pattern: sequence to batch")
    # Sequence of embeddings: (seq_len, embed_dim)
    seq = torch.randn(10, 512)
    print("Sequence:", seq.shape)
    
    # Add batch dimension for model processing
    batched = seq.unsqueeze(0)  # (1, seq_len, embed_dim)
    print("Batched:", batched.shape)
    
    # Or make each sequence element a "batch"
    batch_of_items = seq.unsqueeze(1)  # (seq_len, 1, embed_dim)
    print("Each item as batch:", batch_of_items.shape)

    # -------------------------------------------------------------------------
    header("Practical pattern: flattening for linear layer")
    # CNN feature maps before FC layer
    features = torch.randn(32, 128, 7, 7)  # (batch, channels, H, W)
    print("CNN features:", features.shape)
    
    # Flatten spatial dimensions
    flat = features.flatten(start_dim=1)  # Keep batch, flatten rest
    print("Flattened:", flat.shape)  # (32, 6272)
    
    # Alternative: reshape
    flat2 = features.reshape(32, -1)
    print("Using reshape:", flat2.shape)

    # -------------------------------------------------------------------------
    header("View must preserve total elements")
    x = torch.randn(12)
    
    # Valid reshapes
    print("12 elements can reshape to:")
    for shape in [(12,), (1, 12), (12, 1), (3, 4), (4, 3), (2, 6), (2, 2, 3)]:
        r = x.reshape(shape)
        print(f"  {shape}: {r.numel()} elements")
    
    # Invalid reshape
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
    
    # Transpose changes stride but shares memory
    xt = x.t()
    print("\nx.t() (non-contiguous):\n", xt)
    print("  stride:", xt.stride())
    print("  is_contiguous:", xt.is_contiguous())
    
    # Make contiguous (creates copy)
    xt_contig = xt.contiguous()
    print("\nAfter .contiguous():")
    print("  stride:", xt_contig.stride())
    print("  is_contiguous:", xt_contig.is_contiguous())

    # -------------------------------------------------------------------------
    header("Combining operations: common workflow")
    # Start with (batch, seq_len, hidden_dim)
    x = torch.randn(8, 20, 256)
    print("Input:", x.shape)
    
    # Transpose for attention: (seq_len, batch, hidden_dim)
    x = x.transpose(0, 1)
    print("After transpose:", x.shape)
    
    # Reshape for multi-head: (seq_len, batch, n_heads, head_dim)
    n_heads = 8
    head_dim = 256 // n_heads
    x = x.reshape(20, 8, n_heads, head_dim)
    print("After reshape for heads:", x.shape)
    
    # Permute for computation: (batch, n_heads, seq_len, head_dim)
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
