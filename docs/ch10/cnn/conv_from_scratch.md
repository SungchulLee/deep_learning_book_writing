# 밑바닥부터 만드는 컨볼루션

NumPy 반복문으로 상호상관의 순전파와 입력·필터 그래디언트를 구현하고 PyTorch 결과와 비교한다. 코드의 범위는 스트라이드·팽창률 1, 홀수 길이의 1차원 필터와 홀수 크기의 정사각 2차원 필터를 사용한 같은 크기 패딩이다. 바이어스나 그룹 컨볼루션 등은 구현하지 않는다.

딥러닝에서 컨볼루션이라고 부르는 상호상관은 필터를 뒤집지 않는다. 입력 그래디언트는 이에 대응하는 전치 연산이며, 같은 상호상관 반복문을 재사용할 때는 필터 뒤집기와 경계 인덱스를 맞춰야 한다.

## 1. 코드

??? note "코드 (629줄)"

    ```python
    """밑바닥부터 만드는 컨볼루션."""
    # ---
    # title: "밑바닥부터 만드는 컨볼루션 연산"
    # description: "NumPy로 구현한 1차원·2차원 컨볼루션의 순전파와 역전파,
    #               다채널 지원, 그래디언트 확인, PyTorch로 검증"
    # ---
    #
    # 구현 수준에서 컨볼루션을 이해하는 일은 직접 구조를 설계하고
    # 그래디언트 문제의 벌레를 잡는 데 꼭 필요하다. 이 스크립트는
    # 컨볼루션 연산을 밑바닥부터 만든다:
    #
    #   1부 – 1차원 컨볼루션: 순전파, 입력 그래디언트, 파라메터 그래디언트
    #   2부 – 배치를 쓰는 1차원 컨볼루션
    #   3부 – 2차원 컨볼루션: 순전파와 역전파
    #   4부 – 다채널 2차원 컨볼루션 (배치 × 채널 × H × W)
    #   5부 – 수치적 그래디언트 확인
    #   6부 – PyTorch nn.functional.conv2d와 견주어 검증
    #
    # 여기서 만드는 연산은 딥러닝이 "컨볼루션"이라 부르는 것, 곧 필터를
    # 뒤집지 않는 상관(cross-correlation)이다. F.conv2d도 같은 것을 계산하므로
    # 6부의 견주기가 성립한다.
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
    # 1부 – 1차원 컨볼루션 (표본 하나)
    # =====================================================================
    def _pad_1d(inp: ndarray, num: int) -> ndarray:
        """1차원 배열의 양쪽에 0을 덧댄다."""
        z = np.zeros(num)
        return np.concatenate([z, inp, z])


    def conv_1d(inp: ndarray, param: ndarray) -> ndarray:
        """출력 크기가 같아지도록 덧댄 1차원 컨볼루션.

        out[o] = Σ_p param[p] · inp[o + p − mid],  mid = len(param) // 2

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
        """1차원 컨볼루션의 입력에 대한 로스의 그래디언트.

        핵심: inp[i]는 out[i−mid] … out[i+mid]에 param[mid+i−o]를 곱해
        들어가므로

            input_grad[i] = Σ_p param[p] · output_grad[i + mid − p]

        이다. 순전파와 같은 상관 연산을 쓰되 필터를 180° 뒤집어 넣는 것과
        같다. (뒤집은 상관 = 뒤집지 않은 진짜 컨볼루션이다.)
        """
        param_len = param.shape[0]
        param_mid = param_len // 2

        if output_grad is None:
            output_grad = np.ones_like(inp, dtype=float)
        assert_same_shape(inp, output_grad)

        output_pad = _pad_1d(output_grad, param_mid)
        input_grad = np.zeros_like(inp, dtype=float)

        for o in range(inp.shape[0]):
            for f in range(param_len):
                # 거꾸로 놓는 것에 주의: param_len - f - 1  (뒤집은 필터)
                # param_len - f - 1 - param_mid = param_mid - f 이므로
                # 실제로 읽는 자리는 output_grad[o + param_mid - f]이다.
                input_grad[o] += output_pad[o + param_len - f - 1] * param[f]
        return input_grad


    def _param_grad_1d(inp: ndarray, param: ndarray,
                       output_grad: ndarray = None) -> ndarray:
        """1차원 컨볼루션의 필터에 대한 로스의 그래디언트.

        핵심: param[p]는 모든 출력 자리에 쓰이므로 그래디언트는 그 전부의 합이다.

            param_grad[p] = Σ_o inp_pad[o + p] · output_grad[o]

        곧 덧댄 입력과 출력 그래디언트의 상관이다.
        """
        param_len = param.shape[0]
        param_mid = param_len // 2
        input_pad = _pad_1d(inp, param_mid)

        if output_grad is None:
            output_grad = np.ones_like(inp, dtype=float)
        assert_same_shape(inp, output_grad)

        param_grad = np.zeros_like(param, dtype=float)
        for o in range(inp.shape[0]):
            for p in range(param_len):
                param_grad[p] += input_pad[o + p] * output_grad[o]
        return param_grad


    if __name__ == "__main__":
        print("=" * 60)
        print("Part 1: 1D Convolution from Scratch")
        print("=" * 60)

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
    # 2부 – 배치를 쓰는 1차원 컨볼루션
    # =====================================================================
    def _pad_1d_batch(inp: ndarray, num: int) -> ndarray:
        """배치 [batch, length]의 표본마다 덧댄다."""
        return np.stack([_pad_1d(obs, num) for obs in inp])


    def conv_1d_batch(inp: ndarray, param: ndarray) -> ndarray:
        """배치 [batch, length]에 대한 1차원 컨볼루션."""
        return np.stack([conv_1d(obs, param) for obs in inp])


    def input_grad_1d_batch(inp: ndarray, param: ndarray) -> ndarray:
        """배치 1차원 컨볼루션의 입력 그래디언트."""
        out = conv_1d_batch(inp, param)
        out_grad = np.ones_like(out)
        grads = [_input_grad_1d(inp[i], param, out_grad[i])
                 for i in range(inp.shape[0])]
        return np.stack(grads)


    def param_grad_1d_batch(inp: ndarray, param: ndarray) -> ndarray:
        """배치 1차원 컨볼루션의 파라메터 그래디언트 (배치에 대해 합)."""
        output_grad = np.ones_like(inp, dtype=float)
        # 덧대는 폭은 필터에서 끌어내야 한다. 여기에 1을 박아 두면
        # 길이가 3인 필터에서만 맞고, 길이 5부터는 IndexError로 터진다.
        inp_pad = _pad_1d_batch(inp, param.shape[0] // 2)
        param_grad = np.zeros_like(param, dtype=float)

        for i in range(inp.shape[0]):
            for o in range(inp.shape[1]):
                for p in range(param.shape[0]):
                    param_grad[p] += inp_pad[i][o + p] * output_grad[i][o]
        return param_grad


    if __name__ == "__main__":
        print("=" * 60)
        print("Part 2: Batched 1D Convolution")
        print("=" * 60)

        batch_input = np.array([[0, 1, 2, 3, 4, 5, 6],
                                [1, 2, 3, 4, 5, 6, 7]], dtype=float)
        print(f"  Batch input shape: {batch_input.shape}")
        print(f"  Batch output:\n{conv_1d_batch(batch_input, param_1d)}")
        print(f"  Input grad:\n{input_grad_1d_batch(batch_input, param_1d)}")
        print(f"  Param grad: {param_grad_1d_batch(batch_input, param_1d)}")
        # 덧대는 폭이 필터를 따라가는지: 길이 5인 필터로도 터지지 않아야 한다.
        param_1d_k5 = np.ones(5)
        print(f"  Param grad (k=5): {param_grad_1d_batch(batch_input, param_1d_k5)}")
        print()


    # =====================================================================
    # 3부 – 2차원 컨볼루션 (단일 채널, 배치)
    # =====================================================================
    def _pad_2d_obs(inp: ndarray, num: int) -> ndarray:
        """2차원 배열의 네 면에 0을 덧댄다."""
        inp_pad = _pad_1d_batch(inp, num)  # 행마다 좌우로 덧대기
        pad_row = np.zeros((num, inp.shape[1] + num * 2))
        return np.concatenate([pad_row, inp_pad, pad_row])


    def _pad_2d(inp: ndarray, num: int) -> ndarray:
        """2차원 배열의 배치 [batch, H, W]에 덧댄다."""
        return np.stack([_pad_2d_obs(obs, num) for obs in inp])


    def _compute_output_obs_2d(obs: ndarray, param: ndarray) -> ndarray:
        """관측값 하나에 대한 2차원 컨볼루션 순전파.

        인수:
            obs:   [H, W]
            param: [fH, fW] (정사각 필터, 홀수 크기)

        반환값:
            out:   [H, W] (공간 크기가 같다)
        """
        param_mid = param.shape[0] // 2
        obs_pad = _pad_2d_obs(obs, param_mid)
        out = np.zeros_like(obs, dtype=float)

        for o_h in range(out.shape[0]):
            for o_w in range(out.shape[1]):
                for p_h in range(param.shape[0]):
                    for p_w in range(param.shape[1]):
                        out[o_h][o_w] += param[p_h][p_w] * obs_pad[o_h + p_h][o_w + p_w]
        return out


    def _compute_output_2d(img_batch: ndarray, param: ndarray) -> ndarray:
        """배치 [batch, H, W]에 대한 2차원 컨볼루션 순전파."""
        return np.stack([_compute_output_obs_2d(obs, param) for obs in img_batch])


    def _compute_grads_obs_2d(input_obs: ndarray, output_grad_obs: ndarray,
                              param: ndarray) -> ndarray:
        """2차원 관측값 하나의 입력 그래디언트.

        1차원과 같다. 필터를 두 축 모두에서 뒤집어 덧댄 출력 그래디언트와
        상관을 취한다.
        """
        param_size = param.shape[0]
        output_obs_pad = _pad_2d_obs(output_grad_obs, param_size // 2)
        input_grad = np.zeros_like(input_obs, dtype=float)

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
        """2차원 관측값 배치의 입력 그래디언트."""
        return np.stack([
            _compute_grads_obs_2d(inp[i], output_grad[i], param)
            for i in range(output_grad.shape[0])
        ])


    def _param_grad_2d(inp: ndarray, output_grad: ndarray,
                       param: ndarray) -> ndarray:
        """배치 2차원 컨볼루션의 파라메터(필터) 그래디언트."""
        param_size = param.shape[0]
        inp_pad = _pad_2d(inp, param_size // 2)
        param_grad = np.zeros_like(param, dtype=float)
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


    if __name__ == "__main__":
        print("=" * 60)
        print("Part 3: 2D Convolution from Scratch")
        print("=" * 60)

        np.random.seed(42)
        imgs_2d = np.random.randn(3, 8, 8)   # 8×8 이미지 3장의 배치
        filter_2d = np.random.randn(3, 3)    # 3×3 필터

        out_2d = _compute_output_2d(imgs_2d, filter_2d)
        print(f"  Input shape:  {imgs_2d.shape} (batch, H, W)")
        print(f"  Filter shape: {filter_2d.shape}")
        print(f"  Output shape: {out_2d.shape}")
        print()


    # =====================================================================
    # 4부 – 다채널 2차원 컨볼루션
    # =====================================================================
    def _pad_2d_channel(inp: ndarray, num: int) -> ndarray:
        """[C, H, W]의 채널마다 덧댄다."""
        return np.stack([_pad_2d_obs(ch, num) for ch in inp])


    def _pad_conv_input(inp: ndarray, num: int) -> ndarray:
        """[batch, C, H, W]에 덧댄다."""
        return np.stack([_pad_2d_channel(obs, num) for obs in inp])


    def conv2d_forward(inp: ndarray, param: ndarray) -> ndarray:
        """온전한 다채널 2차원 컨볼루션 순전파.

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
        # H와 W를 따로 읽는다. 둘 다 inp.shape[2]로 두면 정사각이 아닌
        # 입력에서 출력이 조용히 잘린다 (6×9 입력이 6×6으로 나온다).
        img_h, img_w = inp.shape[2], inp.shape[3]

        inp_pad = _pad_conv_input(inp, param_mid)
        out = np.zeros((batch_size, out_channels, img_h, img_w))

        for b in range(batch_size):
            for c_in in range(in_channels):
                for c_out in range(out_channels):
                    for o_h in range(img_h):
                        for o_w in range(img_w):
                            for p_h in range(param_size):
                                for p_w in range(param_size):
                                    out[b][c_out][o_h][o_w] += (
                                        param[c_in][c_out][p_h][p_w]
                                        * inp_pad[b][c_in][o_h + p_h][o_w + p_w]
                                    )
        return out


    def conv2d_input_grad(inp: ndarray, output_grad: ndarray,
                          param: ndarray) -> ndarray:
        """다채널 2차원 컨볼루션의 입력 그래디언트.

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
        img_h, img_w = inp.shape[2], inp.shape[3]

        input_grad = np.zeros_like(inp, dtype=float)
        output_grad_pad = _pad_conv_input(output_grad, param_mid)

        for b in range(batch_size):
            for c_in in range(in_channels):
                for c_out in range(out_channels):
                    for i_h in range(img_h):
                        for i_w in range(img_w):
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
        """다채널 2차원 컨볼루션의 파라메터 그래디언트.

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
        img_h, img_w = inp.shape[2], inp.shape[3]

        inp_pad = _pad_conv_input(inp, param_mid)
        param_grad = np.zeros_like(param, dtype=float)

        for b in range(batch_size):
            for c_in in range(in_channels):
                for c_out in range(out_channels):
                    for o_h in range(img_h):
                        for o_w in range(img_w):
                            for p_h in range(param_size):
                                for p_w in range(param_size):
                                    param_grad[c_in][c_out][p_h][p_w] += (
                                        inp_pad[b][c_in][o_h + p_h][o_w + p_w]
                                        * output_grad[b][c_out][o_h][o_w]
                                    )
        return param_grad


    if __name__ == "__main__":
        print("=" * 60)
        print("Part 4: Multi-channel 2D Convolution")
        print("=" * 60)

        # CIFAR 비슷한 차원(작게)으로 시연
        np.random.seed(42)
        imgs_mc = np.random.randn(2, 3, 8, 8)    # 이미지 2장, 채널 3개, 8×8
        param_mc = np.random.randn(3, 4, 3, 3)   # 채널 3→4개, 3×3 필터

        out_mc = conv2d_forward(imgs_mc, param_mc)
        print(f"  Input shape:  {imgs_mc.shape}  (batch, in_ch, H, W)")
        print(f"  Filter shape: {param_mc.shape} (in_ch, out_ch, fH, fW)")
        print(f"  Output shape: {out_mc.shape}  (batch, out_ch, H, W)")

        # 정사각이 아닌 입력도 모양을 지켜야 한다
        rect = np.random.randn(1, 3, 6, 9)
        print(f"  Non-square in:  {rect.shape}  ->  "
              f"out {conv2d_forward(rect, param_mc).shape}")
        print()


    # =====================================================================
    # 5부 – 수치적 그래디언트 확인
    # =====================================================================
    def numerical_grad_check(forward_fn, x, idx, eps=1e-5):
        """특정 색인에서 유한 차분으로 그래디언트를 확인한다."""
        x_plus = x.copy()
        x_plus.flat[idx] += eps
        x_minus = x.copy()
        x_minus.flat[idx] -= eps
        return (forward_fn(x_plus) - forward_fn(x_minus)) / (2 * eps)


    if __name__ == "__main__":
        print("=" * 60)
        print("Part 5: Numerical Gradient Checking")
        print("=" * 60)

        # --- 1차원 그래디언트 확인 ---
        np.random.seed(42)
        inp_1d = np.random.randn(5)
        par_1d = np.random.randn(3)

        # 입력 그래디언트 확인
        for idx in range(5):
            numerical = numerical_grad_check(
                lambda x: conv_1d(x, par_1d).sum(), inp_1d, idx
            )
            analytical = _input_grad_1d(inp_1d, par_1d)[idx]
            assert abs(numerical - analytical) < 1e-7, f"1D input grad mismatch at {idx}"

        # 파라메터 그래디언트 확인
        for idx in range(3):
            numerical = numerical_grad_check(
                lambda p: conv_1d(inp_1d, p).sum(), par_1d, idx
            )
            analytical = _param_grad_1d(inp_1d, par_1d)[idx]
            assert abs(numerical - analytical) < 1e-7, f"1D param grad mismatch at {idx}"

        print("  ✓ 1D convolution gradients pass numerical check")

        # --- 2차원 그래디언트 확인 (단일 채널) ---
        np.random.seed(42)
        imgs = np.random.randn(2, 6, 6)
        filt = np.random.randn(3, 3)

        # 무작위 자리에서 입력 그래디언트 확인
        test_idx = 25  # imgs(원소 72개)에 대한 평평한 색인
        numerical = numerical_grad_check(
            lambda x: _compute_output_2d(x.reshape(2, 6, 6), filt).sum(),
            imgs.ravel(), test_idx,
        )
        analytical = _compute_grads_2d(imgs, np.ones_like(imgs), filt).ravel()[test_idx]
        assert abs(numerical - analytical) < 1e-6, "2D input grad mismatch"

        # 무작위 자리에서 파라메터 그래디언트 확인
        test_idx_p = 4  # filt(원소 9개)의 한가운데
        numerical = numerical_grad_check(
            lambda p: _compute_output_2d(imgs, p.reshape(3, 3)).sum(),
            filt.ravel(), test_idx_p,
        )
        analytical = _param_grad_2d(imgs, np.ones_like(imgs), filt).ravel()[test_idx_p]
        assert abs(numerical - analytical) < 1e-6, "2D param grad mismatch"

        print("  ✓ 2D convolution gradients pass numerical check")

        # --- 다채널 그래디언트 확인 ---
        np.random.seed(42)
        imgs_small = np.random.randn(2, 2, 6, 6)
        param_small = np.random.randn(2, 3, 3, 3)

        # 입력 그래디언트 확인
        test_idx = 50  # imgs_small(원소 144개)에 대한 평평한 색인
        numerical = numerical_grad_check(
            lambda x: conv2d_forward(x.reshape(2, 2, 6, 6), param_small).sum(),
            imgs_small.ravel(), test_idx,
        )
        analytical = conv2d_input_grad(
            imgs_small, np.ones((2, 3, 6, 6)), param_small
        ).ravel()[test_idx]
        assert abs(numerical - analytical) < 1e-5, "Multi-channel input grad mismatch"

        # 파라메터 그래디언트 확인
        test_idx_p = 20  # param_small(원소 54개)에 대한 평평한 색인
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
    if __name__ == "__main__":
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

            # 출력 그래디언트를 1로만 두면 덧댄 자리의 실수가 가려질 수 있으므로
            # 무작위 출력 그래디언트로도 함께 견준다.
            G_np = np.random.randn(2, 3, 8, 8).astype(np.float64)

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

            # 입력 그래디언트 견주기
            in_grad_ours = conv2d_input_grad(X_np, np.ones_like(out_ours), W_np)
            in_grad_diff = np.abs(X_pt.grad.numpy() - in_grad_ours).max()
            print(f"  Input grad max |diff|:   {in_grad_diff:.2e}")

            # 파라메터 그래디언트 견주기
            param_grad_ours = conv2d_param_grad(X_np, np.ones_like(out_ours), W_np)
            # 견주려고 다시 전치
            param_grad_ours_pt = param_grad_ours.transpose(1, 0, 2, 3)
            param_grad_diff = np.abs(W_pt.grad.numpy() - param_grad_ours_pt).max()
            print(f"  Param grad max |diff|:   {param_grad_diff:.2e}")

            # 무작위 출력 그래디언트로 다시 견주기
            X_pt2 = torch.tensor(X_np, requires_grad=True)
            W_pt2 = torch.tensor(W_pt_format, requires_grad=True)
            (F.conv2d(X_pt2, W_pt2, padding=1) * torch.tensor(G_np)).sum().backward()
            in_g2 = np.abs(X_pt2.grad.numpy()
                           - conv2d_input_grad(X_np, G_np, W_np)).max()
            pa_g2 = np.abs(W_pt2.grad.numpy()
                           - conv2d_param_grad(X_np, G_np, W_np).transpose(1, 0, 2, 3)).max()
            print(f"  Input grad (random dL/dout) max |diff|: {in_g2:.2e}")
            print(f"  Param grad (random dL/dout) max |diff|: {pa_g2:.2e}")

            # 1차원도 같은 방식으로 견준다
            x1 = np.random.randn(9)
            h1 = np.random.randn(3)
            x1_t = torch.tensor(x1, requires_grad=True)
            h1_t = torch.tensor(h1, requires_grad=True)
            o1 = F.conv1d(x1_t.view(1, 1, -1), h1_t.view(1, 1, -1), padding=1)
            o1.sum().backward()
            f1_diff = np.abs(o1.detach().numpy().ravel() - conv_1d(x1, h1)).max()
            i1_diff = np.abs(x1_t.grad.numpy() - _input_grad_1d(x1, h1)).max()
            p1_diff = np.abs(h1_t.grad.numpy() - _param_grad_1d(x1, h1)).max()
            print(f"  1D forward max |diff|:   {f1_diff:.2e}")
            print(f"  1D input grad max |diff|:{i1_diff:.2e}")
            print(f"  1D param grad max |diff|:{p1_diff:.2e}")

            all_match = max(out_diff, in_grad_diff, param_grad_diff,
                            in_g2, pa_g2, f1_diff, i1_diff, p1_diff) < 1e-10
            print(f"  All match: {all_match}")

        except ImportError:
            print("  PyTorch not available — skipping validation")

        print("\nDone.")
    ```


**출력:**

```
============================================================
Part 1: 1D Convolution from Scratch
============================================================
  Input:  [1. 2. 3. 4. 5.]
  Filter: [1. 1. 1.]
  Output: [ 3.  6.  9. 12.  9.]
  Input grad:  [2. 3. 3. 3. 2.]
  Param grad:  [10. 15. 14.]

============================================================
Part 2: Batched 1D Convolution
============================================================
  Batch input shape: (2, 7)
  Batch output:
[[ 1.  3.  6.  9. 12. 15. 11.]
 [ 3.  6.  9. 12. 15. 18. 13.]]
  Input grad:
[[2. 3. 3. 3. 3. 3. 2.]
 [2. 3. 3. 3. 3. 3. 2.]]
  Param grad: [36. 49. 48.]
  Param grad (k=5): [25. 36. 49. 48. 45.]

============================================================
Part 3: 2D Convolution from Scratch
============================================================
  Input shape:  (3, 8, 8) (batch, H, W)
  Filter shape: (3, 3)
  Output shape: (3, 8, 8)

============================================================
Part 4: Multi-channel 2D Convolution
============================================================
  Input shape:  (2, 3, 8, 8)  (batch, in_ch, H, W)
  Filter shape: (3, 4, 3, 3) (in_ch, out_ch, fH, fW)
  Output shape: (2, 4, 8, 8)  (batch, out_ch, H, W)
  Non-square in:  (1, 3, 6, 9)  ->  out (1, 4, 6, 9)

============================================================
Part 5: Numerical Gradient Checking
============================================================
  ✓ 1D convolution gradients pass numerical check
  ✓ 2D convolution gradients pass numerical check
  ✓ Multi-channel 2D convolution gradients pass numerical check

============================================================
Part 6: Validation against PyTorch conv2d
============================================================
  Forward pass max |diff|: 3.55e-15
  Input grad max |diff|:   1.78e-15
  Param grad max |diff|:   1.07e-14
  Input grad (random dL/dout) max |diff|: 5.33e-15
  Param grad (random dL/dout) max |diff|: 1.78e-14
  1D forward max |diff|:   2.22e-16
  1D input grad max |diff|:0.00e+00
  1D param grad max |diff|:1.11e-16
  All match: True

Done.
```

## 2. 논의

### 순전파: 세 개의 인덱스

필터 길이를 $K=2m+1$로 두고 양쪽에 m개씩 0 패딩하면 스트라이드 1에서 길이를 유지한다. 패딩 범위 밖 입력은 0으로 해석하고 1차원 순전파를 다음과 같이 쓴다.

$$y[o] = \sum_{p=0}^{K-1} w[p]\, x[o + p - m]$$

이다. 코드의 `out[o] += param[p] * inp_pad[o + p]`가 바로 이 식인데, `inp_pad`가 이미 $m$만큼 밀려 있으므로 `inp_pad[o + p]`가 $x[o+p-m]$을 가리킨다. 인덱스의 부호가 뒤집히지 않는다는 점이 중요하다. 만약 $x[o - p + m]$이었다면 그것은 진짜 컨볼루션이고, `F.conv1d`와 어긋난다.

다채널 2차원 구현은 같은 위치별 합을 배치·입력 채널·출력 채널로 확장한다. 여기의 필터 배열 순서는 $(C_{in},C_{out},K,K)$이며, PyTorch Conv2d의 $(C_{out},C_{in},K,K)$와 다르므로 비교할 때 첫 두 축을 바꾼다.

$$y[b, c_{\text{out}}, h, v] = \sum_{c_{\text{in}}} \sum_{p_h=0}^{K-1} \sum_{p_w=0}^{K-1} w[c_{\text{in}}, c_{\text{out}}, p_h, p_w]\; x[b, c_{\text{in}},\, h + p_h - m,\, v + p_w - m]$$

입력 채널은 **합쳐 없어지고** 출력 채널은 **새로 생긴다**. 이 비대칭이 `conv2d_forward`에서 `c_in` 반복문이 누적만 하고 `c_out` 반복문이 출력 자리를 고르는 까닭이다.

### 역전파: 같은 식을 두 번 다르게 읽기

$y[o]$가 $w[p]\,x[o+p-m]$의 합이라는 사실 하나에서 두 그래디언트가 모두 나온다. $x[i]$를 고정하고 그것이 들어간 출력 자리를 모으면 $o = i + m - p$이므로

$$\frac{\partial L}{\partial x[i]} = \sum_{p=0}^{K-1} w[p]\, \frac{\partial L}{\partial y[i + m - p]}$$

이 식에서 p의 부호가 순전파와 반대이다. 홀수 K와 대칭 패딩이라는 현재 조건에서는 필터를 뒤집어 같은 상호상관을 수행하는 식으로 바꿀 수 있다. 스트라이드·팽창률이나 패딩 규약을 바꾸면 이 인덱스 대응도 다시 유도해야 한다.

반대로 $w[p]$를 고정하고 그것이 쓰인 자리를 모으면 뒤집기가 없다.

$$\frac{\partial L}{\partial w[p]} = \sum_{o} x[o + p - m]\, \frac{\partial L}{\partial y[o]}$$

필터의 원소는 모든 출력 자리에 그대로 재사용되므로, 그 그래디언트는 패딩한 입력과 출력 그래디언트의 상관을 통째로 더한 값이다. 1부 시연에서 $x = [1,2,3,4,5]$, $w=[1,1,1]$, $L = \sum_o y[o]$일 때 나오는 `[10. 15. 14.]`가 그 결과다. 세 값이 서로 다른 것은 덧댄 0 때문이다 — $w[0]$은 오른쪽 끝의 5를 한 번도 만나지 못하고, $w[2]$는 왼쪽 끝의 1을 만나지 못한다.

같은 까닭으로 입력 그래디언트가 `[2. 3. 3. 3. 2.]`가 된다. 가운데 원소는 출력 세 자리에 들어가지만 양 끝은 두 자리에만 들어간다.

### 맞는지 어떻게 아는가

인덱스나 축 순서를 잘못 써도 숫자와 모양이 그럴듯할 수 있으므로 순전파의 기준 구현 비교와 그래디언트 점검을 구분한다. 아래 결과는 문서에 기록된 특정 입력·자료형에서의 검증 예이며 모든 모양과 입력의 올바름을 증명하지 않는다.

- **5부, 수치적 그래디언트 점검**: 구현된 순전파의 중심 차분과 해석적 그래디언트를 비교한다. 의도한 순전파 자체가 틀려도 그 구현의 도함수를 일관되게 계산했다면 통과할 수 있다. 차분 간격과 자료형에 따른 절삭·반올림 오차도 고려한다.
- **6부, PyTorch와 비교**: 입력·필터 축과 패딩 규약을 맞춰 순전파와 그래디언트를 비교한다. 기록된 작은 잔차는 비교한 예제에서의 수치적 일치를 뒷받침한다. 모든 자료형과 경계 조건에서 같은 허용 오차를 사용할 수 있는 것은 아니다.

!!! note "출력 그래디언트의 선택"
    출력 그래디언트를 모두 1로 두는 경우와 임의의 값을 주는 경우를 함께 비교하면 더 다양한 인덱스 오류를 드러낼 수 있다. 큰 잔차가 있으면 축 순서·패딩·필터 뒤집기와 자료형을 점검하며 오차의 크기만으로 원인을 하나로 확정하지 않는다.

### 조용히 틀리는 두 자리

직접 구현에서는 오류를 발생시키는 입력과 조용히 잘못된 결과를 내는 입력을 모두 검토한다. 다음 두 사례는 현재 구현이 가변 필터 길이와 비정사각 이미지 크기를 따로 처리하는 이유를 설명한다.

- `param_grad_1d_batch`가 패딩 폭을 `1`로 박아 두고 있었다. 길이 3인 필터에서만 맞고, 길이 5인 필터를 넣으면 `IndexError: index 9 is out of bounds for axis 0 with size 9`로 터진다. 이쪽은 그나마 터지므로 나은 편이다. 이제 `param.shape[0] // 2`로 필터에서 끌어내고, 2부 출력의 `Param grad (k=5)` 줄이 그것을 보인다.
- 다채널 함수 셋이 높이와 너비를 모두 `inp.shape[2]`에서 읽고 있었다. 문서 문자열은 `[batch, in_channels, H, W]`라고 적어 두었으면서 실제로는 정사각만 가정한 것이다. $6 \times 9$ 입력을 넣으면 오류 없이 $6 \times 6$ 출력이 나온다 — `F.conv2d`가 내는 $6 \times 9$와 다른데도 아무 경고가 없다. 이제 `img_h, img_w`를 따로 읽으며, 4부 출력의 `Non-square in` 줄이 모양이 지켜지는지 보인다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
$x = [1, 2, 3, 4, 5]$, $w = [1, 1, 1]$, 양쪽에 0을 하나씩 덧댄 경우에 대하여 순전파 $y$를 손으로 구하라. 이어서 $L = \sum_o y[o]$일 때 $\partial L/\partial x$와 $\partial L/\partial w$를 구하고, 1부 출력과 맞는지 확인하라.

</div>

??? success "풀이"
    패딩한 입력은 $x_{\text{pad}} = [0, 1, 2, 3, 4, 5, 0]$이고 $y[o] = x_{\text{pad}}[o] + x_{\text{pad}}[o+1] + x_{\text{pad}}[o+2]$이므로

    $$y = [0{+}1{+}2,\; 1{+}2{+}3,\; 2{+}3{+}4,\; 3{+}4{+}5,\; 4{+}5{+}0] = [3, 6, 9, 12, 9]$$

    이다. $L = \sum_o y[o]$이므로 $\partial L/\partial y[o] = 1$이고, 위 식에서 $\partial L/\partial x[i]$는 $x[i]$가 들어간 출력 자리의 개수와 같다. $x[0]$은 $y[0], y[1]$에만, $x[4]$는 $y[3], y[4]$에만 들어가고 가운데 셋은 세 자리씩 들어가므로

    $$\frac{\partial L}{\partial x} = [2, 3, 3, 3, 2]$$

    이다. 파라메터 쪽은 $\partial L / \partial w[p] = \sum_{o=0}^{4} x_{\text{pad}}[o+p]$이므로

    $$\frac{\partial L}{\partial w} = [0{+}1{+}2{+}3{+}4,\; 1{+}2{+}3{+}4{+}5,\; 2{+}3{+}4{+}5{+}0] = [10, 15, 14]$$

    이다. 세 값 모두 1부 출력의 `Output`, `Input grad`, `Param grad` 줄과 맞는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`param_grad_1d_batch`가 예전에는 `inp_pad = _pad_1d_batch(inp, 1)`로 패딩 폭을 상수 `1`에 박아 두고 있었다. 2부의 배치 입력(모양 `(2, 7)`)에 길이 5인 필터를 넣으면 무슨 일이 일어나는지 인덱스을 따라가며 예측하라. 이 결함이 조용히 틀린 값을 내는 쪽과 터지는 쪽 가운데 어디에 속하는지도 답하라.

</div>

??? success "풀이"
    길이 7인 행을 폭 1로 덧대면 길이가 9가 된다. 그런데 반복문은 `o`가 $0$부터 $6$까지, `p`가 $0$부터 $4$까지 돌므로 읽으려는 가장 큰 인덱스은 $o + p = 6 + 4 = 10$이다. 길이 9인 배열에 인덱스 9와 10은 없으므로

    ```text
    IndexError: index 9 is out of bounds for axis 0 with size 9
    ```

    해당 잘못된 구현은 인덱스 9를 처음 읽을 때 IndexError를 낸다. 홀수 필터 K=5의 올바른 패딩은 2이며 길이는 11이 되어 최대 인덱스 10을 포함한다. 아래 기록값은 해당 입력의 필터 그래디언트이며 다른 입력에서도 같은 숫자가 나오는 것은 아니다.

    예외가 발생하면 원인 추적이 쉬울 수 있지만 조용한 오류와 예외 모두 수정과 검증이 필요하다. 다음 문제는 모양만 잘못되는 입력을 다룬다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
순전파 $y[o] = \sum_p w[p]\, x[o+p-m]$에서 출발하여

$$\frac{\partial L}{\partial x[i]} = \sum_{p} w[p]\, \frac{\partial L}{\partial y[i + m - p]}$$

를 이끌어 내라. 그리고 이 식이 "필터를 180° 돌려 같은 상호상관 연산을 한 것"과 같음을 보여라.

</div>

??? success "풀이"
    연쇄 법칙으로

    $$\frac{\partial L}{\partial x[i]} = \sum_{o} \frac{\partial L}{\partial y[o]} \frac{\partial y[o]}{\partial x[i]}$$

    인데, $y[o] = \sum_p w[p]\, x[o+p-m]$이므로 $\partial y[o] / \partial x[i]$는 $o + p - m = i$를 만족하는 $p$가 있을 때만 0이 아니고 그때 값은 $w[p]$이다. 그 조건은 $p = i + m - o$, 곧 $o = i + m - p$와 같으므로 $o$에 대한 합을 $p$에 대한 합으로 바꾸면

    $$\frac{\partial L}{\partial x[i]} = \sum_{p=0}^{K-1} w[p]\, \frac{\partial L}{\partial y[i + m - p]}$$

    를 얻는다(범위를 벗어나는 $y$ 자리는 0으로 본다 — 코드가 출력 그래디언트를 $m$만큼 덧대는 것이 이것이다).

    뒤집은 필터 $\tilde{w}[p] = w[K - 1 - p]$를 두고 이 식에 $p \mapsto K - 1 - p$를 넣으면

    $$\frac{\partial L}{\partial x[i]} = \sum_{p} \tilde{w}[p]\, \frac{\partial L}{\partial y[i + m - (K-1-p)]} = \sum_{p} \tilde{w}[p]\, \frac{\partial L}{\partial y[i + p - m]}$$

    이 되는데($K - 1 - m = m$을 썼다), 오른쪽은 $x$ 자리에 $\partial L / \partial y$를, $w$ 자리에 $\tilde{w}$를 넣은 순전파 식 그 자체다. 곧 입력 그래디언트는 **출력 그래디언트에 뒤집은 필터로 같은 상호상관 연산을 적용한 것**이다. 코드의 `output_pad[o + param_len - f - 1]`가 이 뒤집기를 인덱스에서 바로 수행한다.

    바꾸어 말하면, 뒤집은 필터와의 상관은 뒤집지 않은 필터와의 **진짜 컨볼루션**이다. 순전파가 상관이고 역전파가 컨볼루션인 이 쌍대성이 컨볼루션 레이어의 구조를 이룬다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`conv2d_forward`가 예전에는 `img_size = inp.shape[2]` 하나로 높이와 너비를 모두 삼았다. 모양이 `(1, 3, 6, 9)`인 입력과 `(3, 4, 3, 3)`인 필터를 넣으면 무엇이 나오는지, 그리고 `F.conv2d`가 내는 것과 어떻게 다른지 답하라. 이 결함이 연습문제 2의 결함보다 나쁜 까닭을 설명하고, 그것을 잡아낼 시험을 한 줄로 적어라.

</div>

??? success "풀이"
    출력 버퍼가 `np.zeros((1, 4, 6, 6))`으로 잡히고 너비 반복문도 6까지만 돌므로, 오류 없이 모양 `(1, 4, 6, 6)`인 배열이 나온다. 마지막 세 열이 통째로 사라지는데 남은 6열의 값은 모두 맞으므로, 눈으로 보아서는 정상적인 피처 맵와 구별되지 않는다. 반면 `F.conv2d(X, W_pt, padding=1)`는 `(1, 4, 6, 9)`를 낸다.

    이 결함이 연습문제 2보다 나쁜 까닭은 **터지지 않기 때문**이다. `IndexError`는 곧바로 눈에 띄고 스택 추적이 범인을 가리키지만, 조용히 잘린 피처 맵는 그대로 다음 레이어로 흘러가 몇 레이어 뒤 `view`나 선형 레이어의 `in_features`에서야 어긋난다. 그때쯤이면 원인이 여기라는 단서가 남아 있지 않다. 더 나쁘게는, 모든 레이어가 정사각 입력만 다루는 뉴럴 넷트웍에서는 끝까지 터지지 않고 학습까지 되면서 비정사각 입력에서만 틀린다.

    정사각이 아닌 입력을 한 번 통과시켜 모양만 견주면 잡힌다.

    ```python
    X = np.random.randn(1, 3, 6, 9)
    W = np.random.randn(3, 4, 3, 3)
    assert conv2d_forward(X, W).shape == (1, 4, 6, 9)
    ```

    축 길이를 다르게 선택한 입력은 축 교환이나 잘못된 길이 참조를 드러내는 데 유용하다. 그러나 모든 오류가 모양으로 드러나는 것은 아니므로 출력값과 그래디언트도 비교한다. 최소 길이, 경계, 입력·출력 채널 수가 다른 경우와 여러 필터 크기도 점검 대상으로 둔다.

## 정리하며

상호상관의 순전파에서 입력·필터 인덱스를 각각 고정하면 두 그래디언트를 유도할 수 있다. 비교 시 필터 축과 패딩 규약을 맞추고 중심 차분 점검과 기준 구현의 순전파 비교를 함께 사용한다. 제시된 예제의 작은 잔차와 올바른 모양은 검증 자료이며 구현 범위 밖의 동작을 보장하지 않는다.
