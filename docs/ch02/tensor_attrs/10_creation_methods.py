#!/usr/bin/env python3
"""
Tensor creation and initialization methods.

Covers:
- Basic constructors: torch.tensor, torch.as_tensor, torch.from_numpy
- Constant tensors: zeros, ones, full, empty
- Identity and diagonal: eye, diag
- Random tensors: rand, randn, randint, randperm
- Range tensors: arange, linspace, logspace
- Like constructors: zeros_like, ones_like, etc.
- Device and dtype specification
"""

import torch
import numpy as np

def header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def main():
    torch.manual_seed(42)

    # -------------------------------------------------------------------------
    header("torch.tensor: creates copy from data")
    data = [[1, 2], [3, 4]]
    t = torch.tensor(data)
    print("torch.tensor(data):\n", t)
    print("dtype:", t.dtype, "| device:", t.device)
    
    # Specify dtype
    t_float = torch.tensor(data, dtype=torch.float64)
    print("With dtype=float64:", t_float.dtype)

    # -------------------------------------------------------------------------
    header("torch.as_tensor: may share memory with input")
    np_array = np.array([[1, 2], [3, 4]])
    t = torch.as_tensor(np_array)  # Shares memory if possible
    print("torch.as_tensor(numpy):\n", t)
    
    # Modification affects both
    np_array[0, 0] = 999
    print("After modifying numpy, torch tensor:", t[0, 0].item())

    # -------------------------------------------------------------------------
    header("torch.from_numpy: shares memory with numpy")
    np_array = np.array([1.0, 2.0, 3.0])
    t = torch.from_numpy(np_array)
    print("torch.from_numpy:", t)
    print("Shares memory:", np.shares_memory(np_array, t.numpy()))

    # -------------------------------------------------------------------------
    header("torch.zeros: all zeros")
    z = torch.zeros(3, 4)
    print("zeros(3, 4):\n", z)
    
    # Specify dtype and device
    z_int = torch.zeros(2, 3, dtype=torch.int64)
    print("zeros with dtype=int64:\n", z_int)

    # -------------------------------------------------------------------------
    header("torch.ones: all ones")
    o = torch.ones(2, 3)
    print("ones(2, 3):\n", o)

    # -------------------------------------------------------------------------
    header("torch.full: fill with specific value")
    f = torch.full((3, 3), fill_value=7.5)
    print("full((3,3), 7.5):\n", f)

    # -------------------------------------------------------------------------
    header("torch.empty: uninitialized memory (fast but random values)")
    e = torch.empty(2, 3)
    print("empty(2, 3) (uninitialized):\n", e)
    print("⚠️  Values are random - don't rely on them!")

    # -------------------------------------------------------------------------
    header("torch.eye: identity matrix")
    eye = torch.eye(4)
    print("eye(4):\n", eye)
    
    # Non-square identity
    eye_rect = torch.eye(3, 5)
    print("eye(3, 5):\n", eye_rect)

    # -------------------------------------------------------------------------
    header("torch.diag: create diagonal matrix or extract diagonal")
    # Create diagonal matrix from vector
    v = torch.tensor([1., 2., 3., 4.])
    diag_mat = torch.diag(v)
    print("diag(vector):\n", diag_mat)
    
    # Extract diagonal from matrix
    mat = torch.randn(4, 4)
    diagonal = torch.diag(mat)
    print("diag(matrix):", diagonal)
    
    # Offset diagonals
    upper_diag = torch.diag(mat, diagonal=1)  # One above main diagonal
    print("Upper diagonal:", upper_diag)

    # -------------------------------------------------------------------------
    header("torch.arange: sequence like Python range")
    seq = torch.arange(10)
    print("arange(10):", seq)
    
    seq2 = torch.arange(2, 10, 2)  # start, end, step
    print("arange(2, 10, 2):", seq2)
    
    seq3 = torch.arange(0, 1, 0.1)  # Float steps
    print("arange(0, 1, 0.1):", seq3)

    # -------------------------------------------------------------------------
    header("torch.linspace: linearly spaced values")
    lin = torch.linspace(0, 10, steps=11)
    print("linspace(0, 10, 11):", lin)
    
    # Useful for plotting ranges
    x = torch.linspace(-3.14, 3.14, steps=7)
    print("linspace(-π, π, 7):", x)

    # -------------------------------------------------------------------------
    header("torch.logspace: logarithmically spaced values")
    log = torch.logspace(0, 3, steps=4)  # 10^0 to 10^3
    print("logspace(0, 3, 4):", log)
    print("Exponentially increasing: 10^0, 10^1, 10^2, 10^3")

    # -------------------------------------------------------------------------
    header("torch.rand: uniform [0, 1)")
    r = torch.rand(3, 4)
    print("rand(3, 4):\n", r)
    print("Range: [0, 1)")

    # -------------------------------------------------------------------------
    header("torch.randn: standard normal N(0, 1)")
    rn = torch.randn(3, 4)
    print("randn(3, 4):\n", rn)
    print("Distribution: N(0, 1)")
    
    # Custom mean and std
    custom = torch.randn(1000) * 2.5 + 10  # N(10, 2.5^2)
    print(f"Custom N(10, 2.5): mean={custom.mean():.2f}, std={custom.std():.2f}")

    # -------------------------------------------------------------------------
    header("torch.randint: random integers")
    ri = torch.randint(low=0, high=10, size=(3, 4))
    print("randint(0, 10, (3,4)):\n", ri)
    print("Range: [0, 10)")

    # -------------------------------------------------------------------------
    header("torch.randperm: random permutation")
    perm = torch.randperm(10)
    print("randperm(10):", perm)
    print("Useful for shuffling indices")

    # -------------------------------------------------------------------------
    header("torch.multinomial: sample from multinomial distribution")
    # Probability weights (don't need to sum to 1)
    weights = torch.tensor([1., 2., 3., 4.])  # Higher numbers = more likely
    samples = torch.multinomial(weights, num_samples=10, replacement=True)
    print("Weights:", weights)
    print("Samples:", samples)
    print("Higher indices (3) should appear more often")

    # -------------------------------------------------------------------------
    header("_like constructors: same shape as another tensor")
    template = torch.randn(3, 4)
    print("Template shape:", template.shape)
    
    z = torch.zeros_like(template)
    print("zeros_like:", z.shape, z.dtype)
    
    o = torch.ones_like(template)
    print("ones_like:", o.shape, o.dtype)
    
    r = torch.rand_like(template)
    print("rand_like:", r.shape, r.dtype)
    
    # Can override dtype
    z_int = torch.zeros_like(template, dtype=torch.int32)
    print("zeros_like with dtype override:", z_int.dtype)

    # -------------------------------------------------------------------------
    header("Device specification")
    cpu_tensor = torch.randn(3, 3, device='cpu')
    print("CPU tensor device:", cpu_tensor.device)
    
    if torch.cuda.is_available():
        gpu_tensor = torch.randn(3, 3, device='cuda')
        print("GPU tensor device:", gpu_tensor.device)
        
        # Transfer between devices
        moved = cpu_tensor.to('cuda')
        print("Moved to GPU:", moved.device)
    else:
        print("CUDA not available, skipping GPU examples")

    # -------------------------------------------------------------------------
    header("requires_grad specification")
    # Create tensor with autograd enabled
    x = torch.randn(3, 4, requires_grad=True)
    print("requires_grad:", x.requires_grad)
    print("is_leaf:", x.is_leaf)
    
    # Can also set after creation
    y = torch.randn(3, 4)
    y.requires_grad_(True)
    print("Set requires_grad after:", y.requires_grad)

    # -------------------------------------------------------------------------
    header("torch.empty_like vs torch.zeros_like performance")
    import time
    
    large_template = torch.randn(1000, 1000)
    
    # empty_like is faster (no initialization)
    start = time.time()
    for _ in range(1000):
        _ = torch.empty_like(large_template)
    empty_time = time.time() - start
    
    start = time.time()
    for _ in range(1000):
        _ = torch.zeros_like(large_template)
    zeros_time = time.time() - start
    
    print(f"empty_like: {empty_time:.4f}s")
    print(f"zeros_like: {zeros_time:.4f}s")
    print(f"Speedup: {zeros_time/empty_time:.2f}x")
    print("⚠️  Use empty only when you'll immediately overwrite values")

    # -------------------------------------------------------------------------
    header("Complex number tensors")
    # Create complex tensor
    real = torch.tensor([1., 2., 3.])
    imag = torch.tensor([4., 5., 6.])
    c = torch.complex(real, imag)
    print("Complex tensor:", c)
    print("dtype:", c.dtype)
    
    # Direct creation
    c2 = torch.tensor([1+2j, 3+4j, 5+6j])
    print("Direct complex:", c2)

    # -------------------------------------------------------------------------
    header("Sparse tensors (briefly)")
    # Create sparse COO tensor
    indices = torch.tensor([[0, 1, 2], [1, 0, 2]])  # (row, col) indices
    values = torch.tensor([3., 4., 5.])
    sparse = torch.sparse_coo_tensor(indices, values, (3, 3))
    print("Sparse tensor:\n", sparse)
    print("Dense representation:\n", sparse.to_dense())

    # -------------------------------------------------------------------------
    header("Cloning and copying")
    original = torch.randn(3, 3)
    
    # clone() creates a copy
    copy = original.clone()
    print("Shares storage (clone):", id(original.storage()) == id(copy.storage()))
    
    # detach().clone() for gradient-free copy
    x = torch.randn(3, requires_grad=True)
    y = x.detach().clone()
    print("Detached clone requires_grad:", y.requires_grad)

    # -------------------------------------------------------------------------
    header("Quick reference: creation functions")
    print("\nZero-value tensors:")
    print("  torch.zeros(shape)           - All zeros")
    print("  torch.zeros_like(tensor)     - Zeros with same shape")
    
    print("\nOne-value tensors:")
    print("  torch.ones(shape)            - All ones")
    print("  torch.ones_like(tensor)      - Ones with same shape")
    print("  torch.full(shape, value)     - All same value")
    
    print("\nRandom tensors:")
    print("  torch.rand(shape)            - Uniform [0, 1)")
    print("  torch.randn(shape)           - Normal N(0, 1)")
    print("  torch.randint(low, high, sz) - Random integers")
    print("  torch.randperm(n)            - Random permutation")
    
    print("\nSequential tensors:")
    print("  torch.arange(start, end, step) - Like Python range")
    print("  torch.linspace(start, end, n)  - n evenly spaced")
    print("  torch.logspace(start, end, n)  - n log-spaced")
    
    print("\nStructured tensors:")
    print("  torch.eye(n)                 - Identity matrix")
    print("  torch.diag(vector)           - Diagonal matrix")
    
    print("\nFrom data:")
    print("  torch.tensor(data)           - Copy from list/array")
    print("  torch.from_numpy(array)      - Share memory with numpy")
    print("  torch.as_tensor(data)        - Share memory if possible")

if __name__ == "__main__":
    main()
