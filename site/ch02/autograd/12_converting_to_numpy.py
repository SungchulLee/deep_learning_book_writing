import torch

def main():
    t = torch.randn(4, requires_grad=True)

    # ----------------------------------------------------------
    # Case 1: requires_grad=True
    # ----------------------------------------------------------
    # .numpy() is DISALLOWED on tensors that require grad.
    # Reason: NumPy has no autograd; exposing a view would let you mutate
    # values behind autograd’s back. PyTorch blocks this to prevent bugs.
    #
    # NOTE: .numpy() also requires the tensor to be on CPU. If you're on CUDA/MPS,
    # call .cpu() first (see below).
    try:
        _ = t.numpy()
    except Exception as e:
        print("t.numpy() fails (requires_grad=True):", e)

    # Correct workflow to get a NumPy VIEW of the SAME storage:
    #   1) .detach()  → drop autograd history (requires_grad=False, grad_fn=None)
    #   2) (can skip if tensor is in CPU) .cpu()  → move to CPU; NumPy arrays are CPU-only
    #   3) .numpy()   → zero-copy view (shared memory with the tensor)
    t_cpu_np = t.detach().clone().cpu().numpy()
    print("Detached .cpu().numpy() shape:", t_cpu_np.shape, "| dtype:", t_cpu_np.dtype)

    # ----------------------------------------------------------
    # Case 2: requires_grad=False
    # ----------------------------------------------------------
    # If the tensor does NOT require gradients and is already on CPU,
    # .numpy() works directly and returns a VIEW (shared memory).
    t2 = torch.randn(4, requires_grad=False)  # CPU by default

    # Correct workflow to get a NumPy VIEW of the SAME storage:
    #   1) (can skip if requires_grad=False) .detach()  → drop autograd history (requires_grad=False, grad_fn=None)
    #   2) (can skip if tensor is in CPU) .cpu()  → move to CPU; NumPy arrays are CPU-only
    #   3) .numpy()   → zero-copy view (shared memory with the tensor)
    t2_np = t2.numpy()
    print("t2.requires_grad:", t2.requires_grad,
          "| numpy() works directly, shape:", t2_np.shape, "| dtype:", t2_np.dtype)

    # ----------------------------------------------------------
    # Notes (important!)
    # ----------------------------------------------------------
    # clone() → copy data and metadata
    #   Data: clone data at new storage (no sharing).
    #   Grad: clone requires_grad and grad_fn if the source has requires_grad=True, the clone keeps grad tracking and is connected to the graph (it has a grad_fn=CloneBackward).
    #   Use when: you need an independent tensor but still want gradients to flow back to the original.
    # detach() → share data but reset metadata
    #   Data: shares the same storage with the source (no new allocation).
    #   Grad: set requires_grad=False and grad_fn=None and stops autograd; returned tensor has requires_grad=False and grad_fn=None.
    #   Use when: you want a view of the data for non-grad work (NumPy, logging, etc.) without breaking the source’s graph.
    # detach_() → reset metadata in-place
    #   Change requires_grad=False, grad_fn=None so it no longer tracks grad.
    #   Allowed on many tensors (leaf and non-leaf), but be careful: doing this on a tensor that’s still used in the graph can silently cut gradient flow.

if __name__ == "__main__":
    main()