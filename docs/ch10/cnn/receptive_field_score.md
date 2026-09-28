┌───────────────┬──────────┬──────────┐
│               │    v1    │    v2    │
├───────────────┼──────────┼──────────┤
│ Math score    │ 7.0 / 10 │ 9.5 / 10 │
├───────────────┼──────────┼──────────┤
│ Writing score │ 8.5 / 10 │ 8.5 / 10 │
└───────────────┴──────────┴──────────┘

v1 → v2  2026-09-28
Math fixes: 2 (🔴 0, 🟡 2, 🟢 0)
Writing fixes: 0
Skipped: nothing.

Notes
- 🟡 Line 157's comment claimed `# 22×22` for the VGG example. The answer is 16×16,
  which the page's own output block prints two lines below and which its own
  layer-by-layer trace (lines 654-659) derives correctly. So the page contradicted
  itself twice over in favour of the one wrong number. Hand-checked:
  3→5→6(pool,jump 2)→10→14→16(pool,jump 4).
- 🟡 Line 339's comment read `RF = 3 × (2^10 - 1) + 1 = 3069`. The expression is
  right and evaluates to 3070, which is what the output block prints; only the
  stated result was wrong. Rewrote it as `3 × 1023 + 1 = 3070` so the intermediate
  step is visible and the slip cannot recur.
- Both defects were in comments, which is why nothing caught them: the page's
  outputs verify 40/40 because comments are not output. This is a blind spot in
  the checking — a reader who trusts the comment over the printed value gets the
  wrong answer, and on the VGG example the wrong answer is 38% too large.
- Everything else checked and correct: the recurrence r_l = r_{l-1} + (K_l-1)·d_l·∏s_i,
  the uniform-stride and stride-1 closed forms, the two-3×3-equals-one-5×5 argument
  (18 vs 25 weights), the dilated comparison (11 vs 63, ratio 5.7), the dilated
  trace k_eff = 5/9/17 giving RF 32/64/128, and all four exercise solutions
  (11; L=50 and 1+2·63=127; 21 vs 63 with the 3× claim; the ERF gradient recipe).
- measure_effective_receptive_field feeds a zero input. Unusual — Luo et al. use
  random input — but not wrong here: with a spatially uniform bias the ReLU pattern
  is uniform, so the gradient still traces the path-count falloff the section
  describes. Left alone; the function is never called and has no output block.
