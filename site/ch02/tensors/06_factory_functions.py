import torch


def header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    # -------------------------------------------------------------------------
    # Seed for reproducibility
    # -------------------------------------------------------------------------
    # Controls the RNG for CPU/CUDA in this process (rand/randn/normal/etc.).
    # NOTE: CUDA has its own RNG stream but is seeded here as well.
    # Determinism across different PyTorch/BLAS versions or devices is not guaranteed.
    torch.manual_seed(123)  # controls rand/randn/normal etc.

    # -------------------------------------------------------------------------
    # Device selection (CPU by default; use CUDA if available)
    # -------------------------------------------------------------------------
    # On Apple Silicon you could also check for 'mps' (Metal) separately.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Common dtype we’ll reuse below (single precision is a good default)
    fp = torch.float32

    # -------------------------------------------------------------------------
    # Basic “fill” factories
    # -------------------------------------------------------------------------
    header("1) Basic fills: zeros / ones / full / empty / eye")

    t_zeros = torch.zeros((2, 3), dtype=fp, device=device, requires_grad=False)
    t_ones  = torch.ones((2, 3), dtype=fp, device=device)
    t_full  = torch.full((2, 3), fill_value=7.7, dtype=fp, device=device)
    t_empty = torch.empty((2, 3), dtype=fp, device=device)  # ⚠️ uninitialized memory (values are garbage)
    t_eye   = torch.eye(4, dtype=fp, device=device)         # 4x4 identity (float because dtype=fp)

    print("zeros:\n", t_zeros)
    print("ones:\n",  t_ones)
    print("full(7.7):\n", t_full)
    print("empty (uninitialized):\n", t_empty)
    print("eye(4):\n", t_eye)

    # -------------------------------------------------------------------------
    # Ranges & spaced values
    # -------------------------------------------------------------------------
    header("2) Ranges: arange / linspace / logspace / randint / randperm")

    # arange: half-open interval [start, end). If any argument is float → float output.
    t_arange_i = torch.arange(0, 10, 2, device=device)          # ints by default when step is int
    t_arange_f = torch.arange(0.0, 1.0, 0.2, device=device)     # floats when any arg is float

    # linspace: N evenly spaced points INCLUDING the end (closed interval)
    t_lin = torch.linspace(0, 1, steps=5, device=device)        # [0., .25, .5, .75, 1.]

    # logspace: geometrically spaced points between base**start and base**end (inclusive)
    t_log = torch.logspace(start=0, end=3, steps=4, base=10.0, device=device)  # [1, 10, 100, 1000]

    # randint: integers in [low, high)
    t_randi = torch.randint(low=0, high=10, size=(3, 4), device=device)

    # randperm: a random permutation of 0..n-1 (no repeats)
    t_perm = torch.randperm(10, device=device)

    print("arange int:", t_arange_i)
    print("arange float:", t_arange_f)
    print("linspace(0,1,5):", t_lin)
    print("logspace(0,3,4):", t_log)
    print("randint[0,10):\n", t_randi)
    print("randperm(10):", t_perm)

    # -------------------------------------------------------------------------
    # Random continuous distributions
    # -------------------------------------------------------------------------
    header("3) Random: rand / randn / normal")

    # rand: U(0,1) i.i.d. on the chosen device/dtype
    t_rand  = torch.rand((2, 3), dtype=fp, device=device)

    # randn: standard normal N(0,1)
    t_randn = torch.randn((2, 3), dtype=fp, device=device)

    # normal: N(mean, std); supports broadcasting if mean/std are tensors
    t_norm  = torch.normal(mean=5.0, std=2.0, size=(2, 3), dtype=fp, device=device)

    print("rand U(0,1):\n", t_rand)
    print("randn N(0,1):\n", t_randn)
    print("normal N(5,2):\n", t_norm)

    # -------------------------------------------------------------------------
    # *_like: create tensors matching shape/dtype/device of another tensor
    # -------------------------------------------------------------------------
    header("4) *_like variants: zeros_like / ones_like / full_like")

    base = torch.randn((3, 2), dtype=torch.float64, device=device)
    # *_like copies shape/dtype/device by default; you can override with kwargs.
    z_like = torch.zeros_like(base)                 # dtype=float64 because base is float64
    o_like = torch.ones_like(base)
    f_like = torch.full_like(base, fill_value=3.14)

    print("base (float64):\n", base)
    print("zeros_like(base):\n", z_like)
    print("ones_like(base):\n",  o_like)
    print("full_like(base, 3.14):\n", f_like)

    # -------------------------------------------------------------------------
    # Triangular / diagonal helpers
    # -------------------------------------------------------------------------
    header("5) Triangular / diagonal: triu / tril / diag / diagonal")

    M = torch.arange(1, 10, device=device, dtype=fp).reshape(3, 3)
    print("M:\n", M)

    M_triu = torch.triu(M)         # upper triangular (copies lower part to zero)
    M_tril = torch.tril(M)         # lower triangular
    d_main = torch.diagonal(M)     # view of the main diagonal (shares storage)
    D = torch.diag(torch.tensor([9., 8., 7.], device=device))  # 1-D → diag matrix (new tensor)

    print("triu(M):\n", M_triu)
    print("tril(M):\n", M_tril)
    print("diagonal(M):", d_main)
    print("diag([9,8,7]):\n", D)

    # -------------------------------------------------------------------------
    # requires_grad: track computations for autograd
    # -------------------------------------------------------------------------
    header("6) requires_grad example")

    # When requires_grad=True on floating tensors, PyTorch builds a graph and accumulates grads.
    w = torch.ones((2, 2), dtype=fp, device=device, requires_grad=True)
    b = torch.zeros((2, 2), dtype=fp, device=device, requires_grad=True)
    x = torch.rand((2, 2), dtype=fp, device=device)  # input (no grad)

    # y = sum(w * x + b) → dy/dw = x, dy/db = 1 (same shape as b)
    y = (w * x + b).sum()
    y.backward()  # populates w.grad and b.grad

    print("w:\n", w)
    print("x:\n", x)
    print("b:\n", b)
    print("y (sum):", y.item())
    print("w.grad:\n", w.grad)  # ≈ x
    print("b.grad:\n", b.grad)  # all ones

    # -------------------------------------------------------------------------
    # Tip: portable device creation
    # -------------------------------------------------------------------------
    header("7) Portable device tip")

    # Preferred pattern: pass `device=device` at creation → avoids extra copies/moves.
    t_portable = torch.ones((2, 2), device=device)
    print("Portable tensor on chosen device:\n", t_portable)

    # If you already made it on CPU, move with .to(device) (creates a new tensor on target device).
    t_moved = torch.ones((2, 2)).to(device)
    print("Moved tensor to device:\n", t_moved)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    header("8) Summary")
    print(
        "• Use zeros/ones/full/empty/eye for basic shapes\n"
        "• Use arange/linspace/logspace/randint/randperm for sequences\n"
        "• Use rand/randn/normal for random continuous values\n"
        "• Use *_like to mirror another tensor's shape/dtype/device\n"
        "• Use triu/tril/diag/diagonal for structured matrices\n"
        "• Always set dtype/device/requires_grad explicitly when it matters\n"
    )


if __name__ == "__main__":
    main()
