┌───────────────┬──────────┬──────────┐
│               │    v1    │    v2    │
├───────────────┼──────────┼──────────┤
│ Math score    │ 5.5 / 10 │ 9.5 / 10 │
├───────────────┼──────────┼──────────┤
│ Writing score │ 3.5 / 10 │ 9.0 / 10 │
└───────────────┴──────────┴──────────┘

v1 → v2  2026-09-29
Math fixes: 8 (🔴 2, 🟡 4, 🟢 2)
Writing fixes: 7 (🔴 3, 🟡 3, 🟢 1)
Skipped: nothing.

Notes
- 🔴 `conv2d_forward`, `conv2d_input_grad`, `conv2d_param_grad` all read the height
  and the width from `inp.shape[2]`, while their docstrings advertised
  `[batch, in_channels, H, W]`. A (1,3,6,9) input came back as (1,4,6,6) — the last
  three columns dropped, no error, no warning, and the surviving 6 columns correct
  so the feature map looks normal. `F.conv2d` gives (1,4,6,9) on the same input.
  Split into `img_h, img_w`; verified against torch on (2,3,6,9) with a random
  dL/dout: forward 7.1e-15, input grad 5.3e-15, param grad 1.2e-14. The page now
  prints a `Non-square in: (1, 3, 6, 9) -> out (1, 4, 6, 9)` line so the regression
  cannot come back silently. Part 3's single-channel code was already correct on
  non-square input, so Part 4 was a regression from the page's own earlier section.
- 🔴 The four exercises were template boilerplate, not exercises about this page.
  Solution 2 and 4 called `Conv From Scratch(...)` and defined
  `def test_conv from scratch():` — identifiers with spaces, i.e. not Python at all,
  in ```python fences. Exercise 3 was about depthwise separable convolution, which
  is `depthwise_separable.md`'s subject and appears nowhere in this page's 578 lines.
  Replaced with four exercises on what the page actually implements, ordered
  쉬움 → 중간 → 중간 → 어려움: hand-compute [3,6,9,12,9] / [2,3,3,3,2] / [10,15,14];
  predict the k=5 IndexError; derive ∂L/∂x[i] = Σ_p w[p]·∂L/∂y[i+m−p] and show it
  equals correlation with the 180°-flipped filter; and diagnose the non-square bug.
- 🟡 `param_grad_1d_batch` hard-coded its pad width as `_pad_1d_batch(inp, 1)`.
  Correct only for a length-3 filter; a length-5 filter on the page's own (2,7) batch
  raises `IndexError: index 9 is out of bounds for axis 0 with size 9` (loop reaches
  o+p = 10, padded length is 9, needs 11). Now `param.shape[0] // 2`. Checked against
  `F.conv1d` for k = 3, 5, 7: exact match, diff 0.0 in all three. The page prints the
  k=5 result `[25. 36. 49. 48. 45.]`.
- 🟡 `if __name__ == "__main__": pass` sat at the bottom of the file — a guard that
  guards nothing, with all six demo sections running at module level. Wrapped each
  part's demo in its own `if __name__ == "__main__":` block rather than collecting
  them into one `main()` at the end, which would have broken the part-by-part reading
  order the file is built around.
- 🟡 Both gradient docstrings said the input gradient is "출력 기울기와 뒤집은 필터의
  합성곱" — flipping and then convolving undoes the flip, so as written it describes
  the forward correlation. Replaced with the index identity
  `input_grad[i] = Σ_p param[p]·output_grad[i + mid − p]` plus the note that this is
  the same correlation routine with the filter turned 180°. Also added the arithmetic
  the code leaves implicit: `param_len − f − 1 − param_mid = param_mid − f`, which is
  what makes the loop index agree with the formula.
- 🟡 Part 6 compared against torch only at ∂L/∂y = 1. A constant output gradient can
  mask padding/index errors by cancellation, so added a random-dL/dout comparison for
  both gradients and a full 1D comparison against `F.conv1d`. Eight residuals now
  print, all ≤ 1.8e-14, and `All match: True` is computed over all eight rather than
  over three.
- 🟢 The accumulators used `np.zeros_like(inp)` / `np.zeros_like(param)`, so an
  integer input array silently truncates every gradient to an integer. The page's own
  demo writes `dtype=float` explicitly, which suggests the author had met this.
  Added `dtype=float`.
- 🟢 `conv2d_input_grad` reshaped `output_grad` to (B, C_out, img_size, img_size)
  before padding — a no-op on square input and a second place enforcing squareness.
  Removed.
- 🔴 The `## 2. 논의` section and `## 정리하며` were about loss functions
  ("손실 계산은 모델의 출력을 최적화 목표와 이어 준다") on a page that implements
  convolution and contains no loss function. Both rewritten to the page's subject.
- 🟡 578 lines of code sat behind one sentence, with not a single formula stated in
  prose. The 논의 section now derives the forward correlation, both gradients, and the
  multi-channel form, and explains why the numerical gradient check alone is not
  enough (it passes even when the forward pass is wrong — it only checks that the
  gradient matches whatever forward pass you wrote).
- 🟡 Opening line was the title repeated verbatim. Difficulty dots ran
  중간·어려움·중간·어려움; the new set runs 쉬움·중간·중간·어려움.

Verification
- Every formula on the page was checked numerically against torch, not read.
  `conv_1d` matches `F.conv1d` to 2.2e-16 and differs from the flipped (true
  convolution) version by 3.36 on random input, confirming which of the two it is.
  1D input grad 0.0, 1D param grad 1.1e-16, both with random dL/dout as well.
  2D single-channel forward 8.9e-16, input grad 0.0, param grad 1.8e-15 with random
  dL/dout. Multi-channel forward 3.6e-15, input grad 5.3e-15, param grad 8.9e-15,
  also with random dL/dout, and 7.1e-15 for a 5×5 filter.
- Every comment asserting a shape, a count, or an arithmetic result was hand-checked.
  Unlike `receptive_field.md` and `convolution.md`, v1's comments were all correct —
  `# 8×8 이미지 3장`, `# 채널 3→4개, 3×3 필터`, the `[out_ch, in_ch]` transpose note,
  the `param_len - f - 1` flip note. The flat indices 25 / 4 / 50 / 20 are all inside
  their arrays (72, 9, 144, 54 elements); the comments now say which array and how
  many elements it has, which is what made them checkable.
- verify_outputs: 29/29 before, 36/36 after. The seven new lines are the k=5 filter
  gradient, the non-square shape, and the five new torch residuals.
