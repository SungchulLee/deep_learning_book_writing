┌───────────────┬──────────┬──────────┐
│               │    v1    │    v2    │
├───────────────┼──────────┼──────────┤
│ Math score    │ 5.5 / 10 │ 9.5 / 10 │
├───────────────┼──────────┼──────────┤
│ Writing score │ 7.0 / 10 │ 9.0 / 10 │
└───────────────┴──────────┴──────────┘

v1 → v2  2026-09-29
Math fixes: 7 (🔴 3, 🟡 3, 🟢 1)
Writing fixes: 5 (🔴 0, 🟡 3, 🟢 2)
Skipped: 🟢 `### $\mathbf{T}^\top$으로 본 전치 합성곱` puts LaTeX in a heading.
The rule in CLAUDE.md names `#` headings, and 16 other pages under `docs/` do the
same in `###` (`ch07/linear_regression/closed_form.md:231`,
`ch06/tensors/23_gpu_operations.md:223`, …) with CI green, so this is the book's
own usage. Rewriting it would also move the section's anchor. 🟢 Difficulty dots
now run 중간·쉬움·어려움·중간·중간·어려움, not ascending — CLAUDE.md explicitly
forbids renumbering an existing page to fix ordering, so the two new exercises
were appended rather than inserted. 🟢 No module docstring / `__main__` guard on
the snippets; there is no `conv1d.py` beside this page and no sibling page in
`ch10/cnn` does this either.

Notes
- 🔴 The page's own code never ran. The 인터페이스 block called
  `nn.Conv1d(in_channels, out_channels, kernel_size, …)` with bare undefined
  names as a pseudo-signature, so the concatenated page died on `NameError:
  name 'in_channels' is not defined` at its sixth line. `verify_outputs.py` then
  fell back to the single longest block and scored **2/14 lines** — twelve
  published numbers, including both receptive fields and the whole NumPy
  gradient check, had never been reproduced by anything. Turned it into a real
  call with concrete values (the same 8/32/5 the 다채널 예제 uses) plus
  `print(conv1d)`, whose output `Conv1d(8, 32, kernel_size=(5,), stride=(1,),
  padding=(2,))` also teaches that PyTorch hides default-valued arguments in
  `repr`. Page now verifies **19/19**.
- 🔴 The symbolic backward formula was wrong while the index formula right above
  it was right. Line 194 said $\partial L/\partial x_i = \sum_j g_j w_{i-j}$
  (correct — checked against autograd), but line 198 then summarised it as
  $g *_{\text{full}} \text{flip}(\mathbf{w})$, which flips twice. On
  `np.random.seed(0)`, n=8, k=3 the true grad_x is
  `[-0.150108, 0.518569, 0.509401, 0.113767, 0.165335, 0.046725, 0.661530,
  0.215212]`; `np.convolve(g, w)` reproduces it exactly, `np.convolve(g,
  w[::-1])` gives `[0.209479, 0.706745, 0.179899, …]` — a different vector.
  The page's own NumPy code was correct (`w_flip` is needed there precisely
  because the loop computes a *cross-correlation*), so the prose contradicted
  the code it introduced. Rewritten as
  $g *_{\text{full}} \mathbf{w} = \text{pad}_{k-1}(g) \star \text{flip}(\mathbf{w})$
  with $\star$ glossed, and 핵심 정리 6 — which repeated the same error — fixed
  to say the flip belongs to the implementation, not the formula. Added two
  printed checks to the NumPy block so the page now proves this rather than
  asserting it (`True` for `w`, `False` for `flip(w)`).
- 🔴 `# 수용 영역: i=0..7에 대해 1 + sum(2^i * (3-1)) = 1 + 2*(1+2+...+128) = 511`
  and `print(f"Receptive field: 511 time steps")` undercount by half.
  `TCNBlock` holds **two** dilated convs at the same dilation (`conv1`,
  `conv2`), so each block adds $2d(k-1)$, giving
  $1 + 2\cdot2\cdot255 = 1021$, not $1 + 2\cdot255 = 511$. Measured it by
  flowing a gradient from the last output of `model.network` back to a
  zero probe of length 2048 and counting non-zero positions: **1021**, matching
  the corrected formula. The print now computes the number from
  `convs_per_block`, `kernel_size` and `num_layers` instead of hardcoding it,
  and prints the measured value beside it. Also noted that 1021 exceeds the
  page's own input length of 256, so the last output already sees the whole
  sequence.
- 🟡 The page is titled 1차원 합성곱 and every formula on it is cross-correlation,
  but it only admitted this in one NumPy docstring. The difference is visible in
  the page's own worked example: $\mathbf{w} = [1, 0, -1]$ gives $[-2,-2,-2]$
  under `F.conv1d` and $[2,2,2]$ under `np.convolve(x, w, 'valid')` — the sign
  flips. Added a warning admonition in 1절 with those two vectors and a link to
  `convolution.md`, which was fixed for exactly this last week.
- 🟡 Three blocks drew `torch.randn` unseeded (전치 합성곱 확인, 다채널 예제, TCN,
  금융 특징 추출기). Only shapes and a boolean were printed so nothing was
  false, but per CLAUDE.md an unseeded number on a page is not reproducible for
  a reader who changes one line. All seeded with `torch.manual_seed(0)`; all
  published values unchanged.
- 🟡 Two `print(f"…")` calls carried no placeholders and hardcoded the claim they
  were pretending to compute (`Receptive field: 1024 samples`,
  `Receptive field: 511 time steps`), and `stack` was built and then never used.
  Both now compute from the loop variables; the WaveNet figure comes out 1024 as
  published (dilation sum $1+2+\cdots+512 = 1023$), so only the TCN number moved.
- 🟡 5절 asserted causality in a trailing comment ("output[t]는 input[t-2],
  input[t-1], input[t]에 기댄다") and printed only `Input length: 10, Output
  length: 10` — which proves nothing, since symmetric padding preserves length
  too. Added a perturbation test: raising the input by 100 from t=5 onward moves
  the output by `0.00e+00` for t < 5 and first changes it at index 5 exactly.
- 🟡 연습문제 3's solution was six lines referencing an undefined `num_classes`,
  with no output — the one exercise marked 어려움 was the least finished. Made it
  runnable and self-contained, added the 길이 불변 point that
  `AdaptiveAvgPool1d(1)` is what the exercise is actually about, and ran it:
  `Logits: torch.Size([16, 5])`, `Parameters: 10821`, which matches
  $192 + 10304 + 325$ by hand.
- 🟡 연습문제 2's solution introduced $\lfloor(L + 2P - K)/S\rfloor + 1$ out of
  nowhere, a different formula from the one 2절 states. Connected them (it is
  the $d = 1$ case) and kept the answer, which checks out: $(100 + 2 - 1\cdot4 -
  1)/2 = 48.5 \to 48$, $+1 = 49$.
- 🟢 Added 연습문제 5 (중간, why causal padding preserves length — derives
  $L_{out} = L + d(k-1)$ and ties it to the 12 → 10 in 5절's code) and 6 (어려움,
  closed form $1 + 2(k-1)(2^L-1)$ for the TCN receptive field, with the 511 trap
  as the last part). The page had 4 against the writer spec's 4–6, and both new
  ones are page-specific rather than generic.
- 🟢 `## 정리하며` had no `---` before it, unlike every other section break on the
  page. Added, plus cross-references to `dilated_convolutions.md` and
  `receptive_field.md`, which the page leaned on without ever linking.
- 🟢 8절's 매개변수 수 row read $C_{out} \times C_{in} \times K$ while 2절 prints
  1,312 = $8\cdot32\cdot5 + 32$. Labelled the row 편향 제외 rather than changing
  the formula.
- Checked every comment asserting a shape or an arithmetic result by hand:
  `(1,1,5)`, `(1,1,3)`, `[-2,-2,-2]`, `[32,8,100]`, `[32,32,100]`,
  `8 × 32 × 5 + 32 = 1,312`, dilations `1…512`, `[32,1]`, `[16,96,252]`
  ($3\times32 = 96$; paddings 1/2/10 all preserve 252), `1 + 4 * 255 = 1021`.
  All correct after the TCN fix. `verify_outputs.py`: 2/14 before, 19/19 after.
