"""Conv from scratch."""
# ---
# title: "Convolution Operations from Scratch"
# description: "1D and 2D convolution forward/backward passes in NumPy,
#               multi-channel support, gradient checking, and PyTorch validation"
# ---
#
# Understanding convolution at the implementation level is essential for
# designing custom architectures and debugging gradient issues.  This script
# builds convolution operations from scratch:
#
#   Part 1 – 1D convolution: forward, input grad, param grad
#   Part 2 – 1D convolution with batches
#   Part 3 – 2D convolution: forward and backward
#   Part 4 – Multi-channel 2D convolution (batch × channels × H × W)
#   Part 5 – Numerical gradient checking
#   Part 6 – Validate against PyTorch nn.functional.conv2d
#
# Adapted from: O'Reilly "Deep Learning from Scratch" Ch 5

import numpy as np
from numpy import ndarray


# =====================================================================
# Helpers
# =====================================================================
def assert_same_shape(a: ndarray, b: ndarray):
    assert a.shape == b.shape, (
        f"Shape mismatch: {a.shape} vs {b.shape}"
    )


# =====================================================================
# Part 1 – 1D Convolution (single sample)
# =====================================================================
print("=" * 60)
print("Part 1: 1D Convolution from Scratch")
print("=" * 60)


def _pad_1d(inp: ndarray, num: int) -> ndarray:
    """Zero-pad a 1D array on both sides."""
    z = np.zeros(num)
    return np.concatenate([z, inp, z])


def conv_1d(inp: ndarray, param: ndarray) -> ndarray:
    """1D convolution with 'same' padding.

    Args:
        inp:   1D input array  [input_length]
        param: 1D filter array [filter_length] (must be odd)

    Returns:
        out:   1D output array [input_length] (same size as input)
    """
    param_len = param.shape[0]
    param_mid = param_len // 2
    inp_pad = _pad_1d(inp, param_mid)

    out = np.zeros(inp.shape)
    for o in range(out.shape[0]):
        for p in range(param_len):
            out[o] += param[p] * inp_pad[o + p]
    return out


def _input_grad_1d(inp: ndarray, param: ndarray,
                   output_grad: ndarray = None) -> ndarray:
    """Gradient of loss w.r.t. the input of a 1D convolution.

    Key insight: the input gradient is a convolution of the output gradient
    with the *flipped* filter.
    """
    param_len = param.shape[0]
    param_mid = param_len // 2

    if output_grad is None:
        output_grad = np.ones_like(inp)
    assert_same_shape(inp, output_grad)

    output_pad = _pad_1d(output_grad, param_mid)
    input_grad = np.zeros_like(inp)

    for o in range(inp.shape[0]):
        for f in range(param_len):
            # Note the reversal: param_len - f - 1  (flipped filter)
            input_grad[o] += output_pad[o + param_len - f - 1] * param[f]
    return input_grad


def _param_grad_1d(inp: ndarray, param: ndarray,
                   output_grad: ndarray = None) -> ndarray:
    """Gradient of loss w.r.t. the filter of a 1D convolution.

    Key insight: the param gradient is a correlation of the padded input
    with the output gradient.
    """
    param_len = param.shape[0]
    param_mid = param_len // 2
    input_pad = _pad_1d(inp, param_mid)

    if output_grad is None:
        output_grad = np.ones_like(inp)
    assert_same_shape(inp, output_grad)

    param_grad = np.zeros_like(param)
    for o in range(inp.shape[0]):
        for p in range(param_len):
            param_grad[p] += input_pad[o + p] * output_grad[o]
    return param_grad


# Demo
input_1d = np.array([1, 2, 3, 4, 5], dtype=float)
param_1d = np.array([1, 1, 1], dtype=float)

out_1d = conv_1d(input_1d, param_1d)
print(f"  Input:  {input_1d}")
print(f"  Filter: {param_1d}")
print(f"  Output: {out_1d}")
print(f"  Input grad:  {_input_grad_1d(input_1d, param_1d)}")
print(f"  Param grad:  {_param_grad_1d(input_1d, param_1d)}")
print()


# =====================================================================
# Part 2 – 1D Convolution with Batches
# =====================================================================
print("=" * 60)
print("Part 2: Batched 1D Convolution")
print("=" * 60)


def _pad_1d_batch(inp: ndarray, num: int) -> ndarray:
    """Pad each sample in a batch [batch, length]."""
    return np.stack([_pad_1d(obs, num) for obs in inp])


def conv_1d_batch(inp: ndarray, param: ndarray) -> ndarray:
    """1D convolution over a batch [batch, length]."""
    return np.stack([conv_1d(obs, param) for obs in inp])


def input_grad_1d_batch(inp: ndarray, param: ndarray) -> ndarray:
    """Input gradient for batched 1D convolution."""
    out = conv_1d_batch(inp, param)
    out_grad = np.ones_like(out)
    grads = [_input_grad_1d(inp[i], param, out_grad[i])
             for i in range(inp.shape[0])]
    return np.stack(grads)


def param_grad_1d_batch(inp: ndarray, param: ndarray) -> ndarray:
    """Parameter gradient for batched 1D convolution (sum over batch)."""
    output_grad = np.ones_like(inp)
    inp_pad = _pad_1d_batch(inp, 1)
    param_grad = np.zeros_like(param)

    for i in range(inp.shape[0]):
        for o in range(inp.shape[1]):
            for p in range(param.shape[0]):
                param_grad[p] += inp_pad[i][o + p] * output_grad[i][o]
    return param_grad


batch_input = np.array([[0, 1, 2, 3, 4, 5, 6],
                         [1, 2, 3, 4, 5, 6, 7]], dtype=float)
print(f"  Batch input shape: {batch_input.shape}")
print(f"  Batch output:\n{conv_1d_batch(batch_input, param_1d)}")
print(f"  Input grad:\n{input_grad_1d_batch(batch_input, param_1d)}")
print(f"  Param grad: {param_grad_1d_batch(batch_input, param_1d)}")
print()


# =====================================================================
# Part 3 – 2D Convolution (single channel, batched)
# =====================================================================
print("=" * 60)
print("Part 3: 2D Convolution from Scratch")
print("=" * 60)


def _pad_2d_obs(inp: ndarray, num: int) -> ndarray:
    """Zero-pad a 2D array on all four sides."""
    inp_pad = _pad_1d_batch(inp, num)  # pad left/right on each row
    pad_row = np.zeros((num, inp.shape[1] + num * 2))
    return np.concatenate([pad_row, inp_pad, pad_row])


def _pad_2d(inp: ndarray, num: int) -> ndarray:
    """Pad a batch of 2D arrays [batch, H, W]."""
    return np.stack([_pad_2d_obs(obs, num) for obs in inp])


def _compute_output_obs_2d(obs: ndarray, param: ndarray) -> ndarray:
    """2D convolution forward for a single observation.

    Args:
        obs:   [H, W]
        param: [fH, fW] (square filter, odd size)

    Returns:
        out:   [H, W] (same spatial size)
    """
    param_mid = param.shape[0] // 2
    obs_pad = _pad_2d_obs(obs, param_mid)
    out = np.zeros_like(obs)

    for o_h in range(out.shape[0]):
        for o_w in range(out.shape[1]):
            for p_h in range(param.shape[0]):
                for p_w in range(param.shape[1]):
                    out[o_h][o_w] += param[p_h][p_w] * obs_pad[o_h + p_h][o_w + p_w]
    return out


def _compute_output_2d(img_batch: ndarray, param: ndarray) -> ndarray:
    """2D convolution forward for a batch [batch, H, W]."""
    return np.stack([_compute_output_obs_2d(obs, param) for obs in img_batch])


def _compute_grads_obs_2d(input_obs: ndarray, output_grad_obs: ndarray,
                          param: ndarray) -> ndarray:
    """Input gradient for a single 2D observation.

    The input gradient is the convolution of the output gradient with the
    180°-rotated filter.
    """
    param_size = param.shape[0]
    output_obs_pad = _pad_2d_obs(output_grad_obs, param_size // 2)
    input_grad = np.zeros_like(input_obs)

    for i_h in range(input_obs.shape[0]):
        for i_w in range(input_obs.shape[1]):
            for p_h in range(param_size):
                for p_w in range(param_size):
                    input_grad[i_h][i_w] += (
                        output_obs_pad[i_h + param_size - p_h - 1]
                                      [i_w + param_size - p_w - 1]
                        * param[p_h][p_w]
                    )
    return input_grad


def _compute_grads_2d(inp: ndarray, output_grad: ndarray,
                      param: ndarray) -> ndarray:
    """Input gradient for a batch of 2D observations."""
    return np.stack([
        _compute_grads_obs_2d(inp[i], output_grad[i], param)
        for i in range(output_grad.shape[0])
    ])


def _param_grad_2d(inp: ndarray, output_grad: ndarray,
                   param: ndarray) -> ndarray:
    """Parameter (filter) gradient for batched 2D convolution."""
    param_size = param.shape[0]
    inp_pad = _pad_2d(inp, param_size // 2)
    param_grad = np.zeros_like(param)
    img_shape = output_grad.shape[1:]

    for i in range(inp.shape[0]):
        for o_h in range(img_shape[0]):
            for o_w in range(img_shape[1]):
                for p_h in range(param_size):
                    for p_w in range(param_size):
                        param_grad[p_h][p_w] += (
                            inp_pad[i][o_h + p_h][o_w + p_w]
                            * output_grad[i][o_h][o_w]
                        )
    return param_grad


np.random.seed(42)
imgs_2d = np.random.randn(3, 8, 8)   # batch of 3, 8×8 images
filter_2d = np.random.randn(3, 3)     # 3×3 filter

out_2d = _compute_output_2d(imgs_2d, filter_2d)
print(f"  Input shape:  {imgs_2d.shape} (batch, H, W)")
print(f"  Filter shape: {filter_2d.shape}")
print(f"  Output shape: {out_2d.shape}")
print()


# =====================================================================
# Part 4 – Multi-channel 2D Convolution
# =====================================================================
print("=" * 60)
print("Part 4: Multi-channel 2D Convolution")
print("=" * 60)


def _pad_2d_channel(inp: ndarray, num: int) -> ndarray:
    """Pad each channel of [C, H, W]."""
    return np.stack([_pad_2d_obs(ch, num) for ch in inp])


def _pad_conv_input(inp: ndarray, num: int) -> ndarray:
    """Pad [batch, C, H, W]."""
    return np.stack([_pad_2d_channel(obs, num) for obs in inp])


def conv2d_forward(inp: ndarray, param: ndarray) -> ndarray:
    """Full multi-channel 2D convolution forward pass.

    Args:
        inp:   [batch, in_channels, H, W]
        param: [in_channels, out_channels, fH, fW]

    Returns:
        out:   [batch, out_channels, H, W]
    """
    batch_size = inp.shape[0]
    in_channels = param.shape[0]
    out_channels = param.shape[1]
    param_size = param.shape[2]
    param_mid = param_size // 2
    img_size = inp.shape[2]

    inp_pad = _pad_conv_input(inp, param_mid)
    out = np.zeros((batch_size, out_channels, img_size, img_size))

    for b in range(batch_size):
        for c_in in range(in_channels):
            for c_out in range(out_channels):
                for o_h in range(img_size):
                    for o_w in range(img_size):
                        for p_h in range(param_size):
                            for p_w in range(param_size):
                                out[b][c_out][o_h][o_w] += (
                                    param[c_in][c_out][p_h][p_w]
                                    * inp_pad[b][c_in][o_h + p_h][o_w + p_w]
                                )
    return out


def conv2d_input_grad(inp: ndarray, output_grad: ndarray,
                      param: ndarray) -> ndarray:
    """Input gradient for multi-channel 2D convolution.

    Args:
        inp:         [batch, in_channels, H, W]
        output_grad: [batch, out_channels, H, W]
        param:       [in_channels, out_channels, fH, fW]

    Returns:
        input_grad:  [batch, in_channels, H, W]
    """
    batch_size = inp.shape[0]
    in_channels = inp.shape[1]
    out_channels = param.shape[1]
    param_size = param.shape[2]
    param_mid = param_size // 2
    img_size = inp.shape[2]

    input_grad = np.zeros_like(inp)
    output_grad_pad = _pad_conv_input(
        output_grad.reshape(batch_size, out_channels, img_size, img_size),
        param_mid,
    )

    for b in range(batch_size):
        for c_in in range(in_channels):
            for c_out in range(out_channels):
                for i_h in range(img_size):
                    for i_w in range(img_size):
                        for p_h in range(param_size):
                            for p_w in range(param_size):
                                input_grad[b][c_in][i_h][i_w] += (
                                    output_grad_pad[b][c_out]
                                    [i_h + param_size - p_h - 1]
                                    [i_w + param_size - p_w - 1]
                                    * param[c_in][c_out][p_h][p_w]
                                )
    return input_grad


def conv2d_param_grad(inp: ndarray, output_grad: ndarray,
                      param: ndarray) -> ndarray:
    """Parameter gradient for multi-channel 2D convolution.

    Args:
        inp:         [batch, in_channels, H, W]
        output_grad: [batch, out_channels, H, W]
        param:       [in_channels, out_channels, fH, fW]

    Returns:
        param_grad:  [in_channels, out_channels, fH, fW]
    """
    batch_size = inp.shape[0]
    in_channels = param.shape[0]
    out_channels = param.shape[1]
    param_size = param.shape[2]
    param_mid = param_size // 2
    img_size = inp.shape[2]

    inp_pad = _pad_conv_input(inp, param_mid)
    param_grad = np.zeros_like(param)

    for b in range(batch_size):
        for c_in in range(in_channels):
            for c_out in range(out_channels):
                for o_h in range(img_size):
                    for o_w in range(img_size):
                        for p_h in range(param_size):
                            for p_w in range(param_size):
                                param_grad[c_in][c_out][p_h][p_w] += (
                                    inp_pad[b][c_in][o_h + p_h][o_w + p_w]
                                    * output_grad[b][c_out][o_h][o_w]
                                )
    return param_grad


# Demo with CIFAR-like dimensions (small)
np.random.seed(42)
imgs_mc = np.random.randn(2, 3, 8, 8)    # 2 images, 3 channels, 8×8
param_mc = np.random.randn(3, 4, 3, 3)   # 3→4 channels, 3×3 filter

out_mc = conv2d_forward(imgs_mc, param_mc)
print(f"  Input shape:  {imgs_mc.shape}  (batch, in_ch, H, W)")
print(f"  Filter shape: {param_mc.shape} (in_ch, out_ch, fH, fW)")
print(f"  Output shape: {out_mc.shape}  (batch, out_ch, H, W)")
print()


# =====================================================================
# Part 5 – Numerical Gradient Checking
# =====================================================================
print("=" * 60)
print("Part 5: Numerical Gradient Checking")
print("=" * 60)


def numerical_grad_check(forward_fn, x, idx, eps=1e-5):
    """Check gradient using finite differences at a specific index."""
    x_plus = x.copy()
    x_plus.flat[idx] += eps
    x_minus = x.copy()
    x_minus.flat[idx] -= eps
    return (forward_fn(x_plus) - forward_fn(x_minus)) / (2 * eps)


# --- 1D gradient check ---
np.random.seed(42)
inp_1d = np.random.randn(5)
par_1d = np.random.randn(3)

# Check input grad
for idx in range(5):
    numerical = numerical_grad_check(
        lambda x: conv_1d(x, par_1d).sum(), inp_1d, idx
    )
    analytical = _input_grad_1d(inp_1d, par_1d)[idx]
    assert abs(numerical - analytical) < 1e-7, f"1D input grad mismatch at {idx}"

# Check param grad
for idx in range(3):
    numerical = numerical_grad_check(
        lambda p: conv_1d(inp_1d, p).sum(), par_1d, idx
    )
    analytical = _param_grad_1d(inp_1d, par_1d)[idx]
    assert abs(numerical - analytical) < 1e-7, f"1D param grad mismatch at {idx}"

print("  ✓ 1D convolution gradients pass numerical check")

# --- 2D gradient check (single channel) ---
np.random.seed(42)
imgs = np.random.randn(2, 6, 6)
filt = np.random.randn(3, 3)

# Check input grad at random position
test_idx = 25  # flat index into imgs
numerical = numerical_grad_check(
    lambda x: _compute_output_2d(x.reshape(2, 6, 6), filt).sum(),
    imgs.ravel(), test_idx,
)
analytical = _compute_grads_2d(imgs, np.ones_like(imgs), filt).ravel()[test_idx]
assert abs(numerical - analytical) < 1e-6, "2D input grad mismatch"

# Check param grad at random position
test_idx_p = 4
numerical = numerical_grad_check(
    lambda p: _compute_output_2d(imgs, p.reshape(3, 3)).sum(),
    filt.ravel(), test_idx_p,
)
analytical = _param_grad_2d(imgs, np.ones_like(imgs), filt).ravel()[test_idx_p]
assert abs(numerical - analytical) < 1e-6, "2D param grad mismatch"

print("  ✓ 2D convolution gradients pass numerical check")

# --- Multi-channel gradient check ---
np.random.seed(42)
imgs_small = np.random.randn(2, 2, 6, 6)
param_small = np.random.randn(2, 3, 3, 3)

# Input grad check
test_idx = 50
numerical = numerical_grad_check(
    lambda x: conv2d_forward(x.reshape(2, 2, 6, 6), param_small).sum(),
    imgs_small.ravel(), test_idx,
)
analytical = conv2d_input_grad(
    imgs_small, np.ones((2, 3, 6, 6)), param_small
).ravel()[test_idx]
assert abs(numerical - analytical) < 1e-5, "Multi-channel input grad mismatch"

# Param grad check
test_idx_p = 20
numerical = numerical_grad_check(
    lambda p: conv2d_forward(imgs_small, p.reshape(2, 3, 3, 3)).sum(),
    param_small.ravel(), test_idx_p,
)
analytical = conv2d_param_grad(
    imgs_small, np.ones((2, 3, 6, 6)), param_small
).ravel()[test_idx_p]
assert abs(numerical - analytical) < 1e-5, "Multi-channel param grad mismatch"

print("  ✓ Multi-channel 2D convolution gradients pass numerical check")
print()


# =====================================================================
# Part 6 – Validate Against PyTorch
# =====================================================================
print("=" * 60)
print("Part 6: Validation against PyTorch conv2d")
print("=" * 60)

try:
    import torch
    import torch.nn.functional as F

    np.random.seed(42)
    X_np = np.random.randn(2, 2, 8, 8).astype(np.float64)
    # PyTorch Conv2d uses [out_ch, in_ch, fH, fW] convention
    # Our code uses [in_ch, out_ch, fH, fW], so we need to transpose
    W_np = np.random.randn(2, 3, 3, 3).astype(np.float64)  # in, out, fH, fW
    W_pt_format = W_np.transpose(1, 0, 2, 3)  # out, in, fH, fW

    # PyTorch forward
    X_pt = torch.tensor(X_np, requires_grad=True)
    W_pt = torch.tensor(W_pt_format, requires_grad=True)
    out_pt = F.conv2d(X_pt, W_pt, padding=1)
    loss_pt = out_pt.sum()
    loss_pt.backward()

    # Our forward
    out_ours = conv2d_forward(X_np, W_np)

    # Compare outputs
    out_diff = np.abs(out_pt.detach().numpy() - out_ours).max()
    print(f"  Forward pass max |diff|: {out_diff:.2e}")

    # Compare input gradients
    in_grad_ours = conv2d_input_grad(X_np, np.ones_like(out_ours), W_np)
    in_grad_diff = np.abs(X_pt.grad.numpy() - in_grad_ours).max()
    print(f"  Input grad max |diff|:   {in_grad_diff:.2e}")

    # Compare param gradients
    param_grad_ours = conv2d_param_grad(X_np, np.ones_like(out_ours), W_np)
    # Transpose back for comparison
    param_grad_ours_pt = param_grad_ours.transpose(1, 0, 2, 3)
    param_grad_diff = np.abs(W_pt.grad.numpy() - param_grad_ours_pt).max()
    print(f"  Param grad max |diff|:   {param_grad_diff:.2e}")

    all_match = out_diff < 1e-10 and in_grad_diff < 1e-10 and param_grad_diff < 1e-10
    print(f"  All match: {all_match}")

except ImportError:
    print("  PyTorch not available — skipping validation")

print("\nDone.")


if __name__ == "__main__":
    pass
