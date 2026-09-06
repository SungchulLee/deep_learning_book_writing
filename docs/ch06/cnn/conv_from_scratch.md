# 밑바닥부터 만드는 합성곱

밑바닥부터 만드는 합성곱.

합성곱 구조는 요즘 컴퓨터 비전 시스템의 뼈대를 이룬다. 이 구현은 PyTorch로 합성곱 신경망 설계의 핵심 개념을 보이며, 이미지 데이터에서 공간적인 특징의 위계가 어떻게 학습되는지 드러낸다.

## 코드

```python
"""밑바닥부터 만드는 합성곱."""
# ---
# title: "밑바닥부터 만드는 합성곱 연산"
# description: "NumPy로 구현한 1차원·2차원 합성곱의 순전파와 역전파,
#               다채널 지원, 기울기 확인, PyTorch로 검증"
# ---
#
# 구현 수준에서 합성곱을 이해하는 일은 직접 구조를 설계하고
# 기울기 문제의 벌레를 잡는 데 꼭 필요하다. 이 스크립트는
# 합성곱 연산을 밑바닥부터 만든다:
#
#   1부 – 1차원 합성곱: 순전파, 입력 기울기, 매개변수 기울기
#   2부 – 배치를 쓰는 1차원 합성곱
#   3부 – 2차원 합성곱: 순전파와 역전파
#   4부 – 다채널 2차원 합성곱 (배치 × 채널 × H × W)
#   5부 – 수치적 기울기 확인
#   6부 – PyTorch nn.functional.conv2d와 견주어 검증
#
# 출처: O'Reilly "Deep Learning from Scratch" 5장에서 고쳐 씀

import numpy as np
from numpy import ndarray


# =====================================================================
# 도우미 함수
# =====================================================================
def assert_same_shape(a: ndarray, b: ndarray):
    assert a.shape == b.shape, (
        f"Shape mismatch: {a.shape} vs {b.shape}"
    )


# =====================================================================
# 1부 – 1차원 합성곱 (표본 하나)
# =====================================================================
print("=" * 60)
print("Part 1: 1D Convolution from Scratch")
print("=" * 60)


def _pad_1d(inp: ndarray, num: int) -> ndarray:
    """1차원 배열의 양쪽에 0을 덧댄다."""
    z = np.zeros(num)
    return np.concatenate([z, inp, z])


def conv_1d(inp: ndarray, param: ndarray) -> ndarray:
    """출력 크기가 같아지도록 덧댄 1차원 합성곱.

    인수:
        inp:   1차원 입력 배열  [input_length]
        param: 1차원 필터 배열 [filter_length] (홀수여야 한다)

    반환값:
        out:   1차원 출력 배열 [input_length] (입력과 크기가 같다)
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
    """1차원 합성곱의 입력에 대한 손실의 기울기.

    핵심: 입력 기울기는 출력 기울기와 *뒤집은* 필터의
    합성곱이다.
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
            # 거꾸로 놓는 것에 주의: param_len - f - 1  (뒤집은 필터)
            input_grad[o] += output_pad[o + param_len - f - 1] * param[f]
    return input_grad


def _param_grad_1d(inp: ndarray, param: ndarray,
                   output_grad: ndarray = None) -> ndarray:
    """1차원 합성곱의 필터에 대한 손실의 기울기.

    핵심: 매개변수 기울기는 덧댄 입력과 출력 기울기의
    상관이다.
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


# 시연
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
# 2부 – 배치를 쓰는 1차원 합성곱
# =====================================================================
print("=" * 60)
print("Part 2: Batched 1D Convolution")
print("=" * 60)


def _pad_1d_batch(inp: ndarray, num: int) -> ndarray:
    """배치 [batch, length]의 표본마다 덧댄다."""
    return np.stack([_pad_1d(obs, num) for obs in inp])


def conv_1d_batch(inp: ndarray, param: ndarray) -> ndarray:
    """배치 [batch, length]에 대한 1차원 합성곱."""
    return np.stack([conv_1d(obs, param) for obs in inp])


def input_grad_1d_batch(inp: ndarray, param: ndarray) -> ndarray:
    """배치 1차원 합성곱의 입력 기울기."""
    out = conv_1d_batch(inp, param)
    out_grad = np.ones_like(out)
    grads = [_input_grad_1d(inp[i], param, out_grad[i])
             for i in range(inp.shape[0])]
    return np.stack(grads)


def param_grad_1d_batch(inp: ndarray, param: ndarray) -> ndarray:
    """배치 1차원 합성곱의 매개변수 기울기 (배치에 대해 합)."""
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
# 3부 – 2차원 합성곱 (단일 채널, 배치)
# =====================================================================
print("=" * 60)
print("Part 3: 2D Convolution from Scratch")
print("=" * 60)


def _pad_2d_obs(inp: ndarray, num: int) -> ndarray:
    """2차원 배열의 네 면에 0을 덧댄다."""
    inp_pad = _pad_1d_batch(inp, num)  # 행마다 좌우로 덧대기
    pad_row = np.zeros((num, inp.shape[1] + num * 2))
    return np.concatenate([pad_row, inp_pad, pad_row])


def _pad_2d(inp: ndarray, num: int) -> ndarray:
    """2차원 배열의 배치 [batch, H, W]에 덧댄다."""
    return np.stack([_pad_2d_obs(obs, num) for obs in inp])


def _compute_output_obs_2d(obs: ndarray, param: ndarray) -> ndarray:
    """관측값 하나에 대한 2차원 합성곱 순전파.

    인수:
        obs:   [H, W]
        param: [fH, fW] (정사각 필터, 홀수 크기)

    반환값:
        out:   [H, W] (공간 크기가 같다)
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
    """배치 [batch, H, W]에 대한 2차원 합성곱 순전파."""
    return np.stack([_compute_output_obs_2d(obs, param) for obs in img_batch])


def _compute_grads_obs_2d(input_obs: ndarray, output_grad_obs: ndarray,
                          param: ndarray) -> ndarray:
    """2차원 관측값 하나의 입력 기울기.

    입력 기울기는 출력 기울기와 180° 돌린 필터의
    합성곱이다.
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
    """2차원 관측값 배치의 입력 기울기."""
    return np.stack([
        _compute_grads_obs_2d(inp[i], output_grad[i], param)
        for i in range(output_grad.shape[0])
    ])


def _param_grad_2d(inp: ndarray, output_grad: ndarray,
                   param: ndarray) -> ndarray:
    """배치 2차원 합성곱의 매개변수(필터) 기울기."""
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
imgs_2d = np.random.randn(3, 8, 8)   # 8×8 이미지 3장의 배치
filter_2d = np.random.randn(3, 3)     # 3×3 필터

out_2d = _compute_output_2d(imgs_2d, filter_2d)
print(f"  Input shape:  {imgs_2d.shape} (batch, H, W)")
print(f"  Filter shape: {filter_2d.shape}")
print(f"  Output shape: {out_2d.shape}")
print()


# =====================================================================
# 4부 – 다채널 2차원 합성곱
# =====================================================================
print("=" * 60)
print("Part 4: Multi-channel 2D Convolution")
print("=" * 60)


def _pad_2d_channel(inp: ndarray, num: int) -> ndarray:
    """[C, H, W]의 채널마다 덧댄다."""
    return np.stack([_pad_2d_obs(ch, num) for ch in inp])


def _pad_conv_input(inp: ndarray, num: int) -> ndarray:
    """[batch, C, H, W]에 덧댄다."""
    return np.stack([_pad_2d_channel(obs, num) for obs in inp])


def conv2d_forward(inp: ndarray, param: ndarray) -> ndarray:
    """온전한 다채널 2차원 합성곱 순전파.

    인수:
        inp:   [batch, in_channels, H, W]
        param: [in_channels, out_channels, fH, fW]

    반환값:
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
    """다채널 2차원 합성곱의 입력 기울기.

    인수:
        inp:         [batch, in_channels, H, W]
        output_grad: [batch, out_channels, H, W]
        param:       [in_channels, out_channels, fH, fW]

    반환값:
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
    """다채널 2차원 합성곱의 매개변수 기울기.

    인수:
        inp:         [batch, in_channels, H, W]
        output_grad: [batch, out_channels, H, W]
        param:       [in_channels, out_channels, fH, fW]

    반환값:
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


# CIFAR 비슷한 차원(작게)으로 시연
np.random.seed(42)
imgs_mc = np.random.randn(2, 3, 8, 8)    # 이미지 2장, 채널 3개, 8×8
param_mc = np.random.randn(3, 4, 3, 3)   # 채널 3→4개, 3×3 필터

out_mc = conv2d_forward(imgs_mc, param_mc)
print(f"  Input shape:  {imgs_mc.shape}  (batch, in_ch, H, W)")
print(f"  Filter shape: {param_mc.shape} (in_ch, out_ch, fH, fW)")
print(f"  Output shape: {out_mc.shape}  (batch, out_ch, H, W)")
print()


# =====================================================================
# 5부 – 수치적 기울기 확인
# =====================================================================
print("=" * 60)
print("Part 5: Numerical Gradient Checking")
print("=" * 60)


def numerical_grad_check(forward_fn, x, idx, eps=1e-5):
    """특정 색인에서 유한 차분으로 기울기를 확인한다."""
    x_plus = x.copy()
    x_plus.flat[idx] += eps
    x_minus = x.copy()
    x_minus.flat[idx] -= eps
    return (forward_fn(x_plus) - forward_fn(x_minus)) / (2 * eps)


# --- 1차원 기울기 확인 ---
np.random.seed(42)
inp_1d = np.random.randn(5)
par_1d = np.random.randn(3)

# 입력 기울기 확인
for idx in range(5):
    numerical = numerical_grad_check(
        lambda x: conv_1d(x, par_1d).sum(), inp_1d, idx
    )
    analytical = _input_grad_1d(inp_1d, par_1d)[idx]
    assert abs(numerical - analytical) < 1e-7, f"1D input grad mismatch at {idx}"

# 매개변수 기울기 확인
for idx in range(3):
    numerical = numerical_grad_check(
        lambda p: conv_1d(inp_1d, p).sum(), par_1d, idx
    )
    analytical = _param_grad_1d(inp_1d, par_1d)[idx]
    assert abs(numerical - analytical) < 1e-7, f"1D param grad mismatch at {idx}"

print("  ✓ 1D convolution gradients pass numerical check")

# --- 2차원 기울기 확인 (단일 채널) ---
np.random.seed(42)
imgs = np.random.randn(2, 6, 6)
filt = np.random.randn(3, 3)

# 무작위 자리에서 입력 기울기 확인
test_idx = 25  # imgs에 대한 평평한 색인
numerical = numerical_grad_check(
    lambda x: _compute_output_2d(x.reshape(2, 6, 6), filt).sum(),
    imgs.ravel(), test_idx,
)
analytical = _compute_grads_2d(imgs, np.ones_like(imgs), filt).ravel()[test_idx]
assert abs(numerical - analytical) < 1e-6, "2D input grad mismatch"

# 무작위 자리에서 매개변수 기울기 확인
test_idx_p = 4
numerical = numerical_grad_check(
    lambda p: _compute_output_2d(imgs, p.reshape(3, 3)).sum(),
    filt.ravel(), test_idx_p,
)
analytical = _param_grad_2d(imgs, np.ones_like(imgs), filt).ravel()[test_idx_p]
assert abs(numerical - analytical) < 1e-6, "2D param grad mismatch"

print("  ✓ 2D convolution gradients pass numerical check")

# --- 다채널 기울기 확인 ---
np.random.seed(42)
imgs_small = np.random.randn(2, 2, 6, 6)
param_small = np.random.randn(2, 3, 3, 3)

# 입력 기울기 확인
test_idx = 50
numerical = numerical_grad_check(
    lambda x: conv2d_forward(x.reshape(2, 2, 6, 6), param_small).sum(),
    imgs_small.ravel(), test_idx,
)
analytical = conv2d_input_grad(
    imgs_small, np.ones((2, 3, 6, 6)), param_small
).ravel()[test_idx]
assert abs(numerical - analytical) < 1e-5, "Multi-channel input grad mismatch"

# 매개변수 기울기 확인
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
# 6부 – PyTorch와 견주어 검증
# =====================================================================
print("=" * 60)
print("Part 6: Validation against PyTorch conv2d")
print("=" * 60)

try:
    import torch
    import torch.nn.functional as F

    np.random.seed(42)
    X_np = np.random.randn(2, 2, 8, 8).astype(np.float64)
    # PyTorch Conv2d는 [out_ch, in_ch, fH, fW] 규약을 쓴다
    # 우리 코드는 [in_ch, out_ch, fH, fW]를 쓰므로 전치해야 한다
    W_np = np.random.randn(2, 3, 3, 3).astype(np.float64)  # in, out, fH, fW
    W_pt_format = W_np.transpose(1, 0, 2, 3)  # out, in, fH, fW

    # PyTorch 순전파
    X_pt = torch.tensor(X_np, requires_grad=True)
    W_pt = torch.tensor(W_pt_format, requires_grad=True)
    out_pt = F.conv2d(X_pt, W_pt, padding=1)
    loss_pt = out_pt.sum()
    loss_pt.backward()

    # 우리 순전파
    out_ours = conv2d_forward(X_np, W_np)

    # 출력 견주기
    out_diff = np.abs(out_pt.detach().numpy() - out_ours).max()
    print(f"  Forward pass max |diff|: {out_diff:.2e}")

    # 입력 기울기 견주기
    in_grad_ours = conv2d_input_grad(X_np, np.ones_like(out_ours), W_np)
    in_grad_diff = np.abs(X_pt.grad.numpy() - in_grad_ours).max()
    print(f"  Input grad max |diff|:   {in_grad_diff:.2e}")

    # 매개변수 기울기 견주기
    param_grad_ours = conv2d_param_grad(X_np, np.ones_like(out_ours), W_np)
    # 견주려고 다시 전치
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
```

## 논의

손실 계산은 모델의 출력을 최적화 목표와 이어 준다. 알맞은 손실 함수를 고르는 일은 결정적으로 중요하다. 손실 함수가 모델이 무엇을 최적화하도록 배울지를 정하며, 학습된 표현과 결정 경계를 직접 빚어내기 때문이다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 컴퓨터 비전 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
코드를 훑으며 핵심 설계 결정을 찾아라. 구체적인 구현 선택 세 가지를 열거하고 각각이 합성곱 신경망에 알맞은 까닭을 설명하라.

??? success "연습문제 1 풀이"
    설계 결정은 구현마다 다르지만 흔히 다음이 포함된다. (1) 활성화 함수의 선택 — ReLU 계열은 포화되지 않는 경사를 주어 학습을 빠르게 한다. (2) 정규화 전략 — 배치 정규화는 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 — 있을 경우 건너뛰는 경로를 제공하여 깊은 신경망에서도 경사가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 절충을 반영한다.

---

**연습문제 2.**
$64 \times 64$ 크기의 RGB 이미지(입력 모양 $3 \times 64 \times 64$)를 받도록 구조를 수정하라. 모든 층의 차원을 그에 맞게 고치고 모델이 오류 없이 실행되는지 확인하라.

??? success "연습문제 2 풀이"
    첫 합성곱 층의 `in_channels`을 지금 값에서 3으로 바꾸어라. 식 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$으로 합성곱과 풀링 층마다의 공간 차원을 다시 계산하라. 마지막 합성곱/풀링 층의 펼친 출력에 맞게 첫 선형층의 `in_features`을 고쳐라. `model = Conv From Scratch(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`으로 확인하라.

---

**연습문제 3.**
같은 입출력 차원에서 표준 합성곱과 깊이별 분리 합성곱의 매개변수 개수와 FLOPs를 비교하라. 계산 절감이 가장 큰 것은 언제인가?

??? success "연습문제 3 풀이"
    표준 `Conv2d(C_in, C_out, k)`은 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개의 매개변수를 갖는다. 깊이별 분리 합성곱은 이를 둘로 나눈다. (1) 깊이별: $C_{{\text{{in}}}} \times k^2$개(입력 채널마다 필터 하나), (2) 점별: $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 합성곱)이다. 매개변수의 비는 대략 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 매개변수가 약 $8{-}9\times$ 적어진다. 절감은 $C_{{\text{{out}}}}$과 $k$가 모두 클 때 가장 크다.

---

**연습문제 4.**
밑바닥부터 만든 합성곱 구현을 검증하는 종합 시험 함수를 작성하라. 빈 입력, 원소가 하나뿐인 입력, 아주 큰 입력, 극단적인 값(0, 아주 큰 수)을 가진 입력 같은 경계 상황을 시험하라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_conv from scratch():
        model = Conv From Scratch(...)
        # 보통의 입력
        assert model(normal_input).shape == expected_shape
        # 원소가 하나인 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값 (넘침을 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 경사의 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    경사의 흐름을 시험하는 것은 그 구조가 처음부터 끝까지 이어지는 학습을 지원하는지 확인하는 데 특히 중요하다.
