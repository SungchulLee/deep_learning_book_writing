┌───────────────┬──────────┬──────────┐
│               │    v1    │    v2    │
├───────────────┼──────────┼──────────┤
│ Math score    │ 5.5 / 10 │ 9.5 / 10 │
├───────────────┼──────────┼──────────┤
│ Writing score │ 6.5 / 10 │ 9.0 / 10 │
└───────────────┴──────────┴──────────┘

v1 → v2  2026-09-29
Math fixes: 7 (🔴 4, 🟡 2, 🟢 1)
Writing fixes: 6 (🔴 1, 🟡 4, 🟢 1)
Skipped: 🟢 연습문제 difficulty runs 어려움 → 어려움 → 중간 → 쉬움, exactly the
reverse of the house order. CLAUDE.md says not to renumber an existing page for
order alone, so the four are left where they are. 🟢 `ch21/index.md` line 18
still calls ResNeXt's grouped convolution "배치 합성곱"; this run was told to
touch only this page, so that one line now disagrees with the book's other
1,144 uses of 배치 and needs a separate pass. 🟢 The nav title in `mkdocs.yml`
is "깊이별 분리 합성곱" while the H1 names both techniques — the same mismatch
existed in v1 and fixing it means editing a shared file.

Notes
- 🔴 The headline defect is the sibling's bias-omission error, twice. §2's code
  comments read `# 36,992 (2배 줄어듦)` and `# 18,560 (4배 줄어듦)`. Counted on
  the real modules: weights go 73,728 → 36,864 → 18,432, which is exactly 2× and
  4×, but the 128 biases do not scale, so the **layer totals** are
  73,856 / 36,992 = **1.997×** and 73,856 / 18,560 = **3.979×**. The code now
  prints weight, bias and total separately for all three convolutions and then
  prints both ratios, so the reader sees 2.000 next to 1.997 instead of being
  told "2배" and shown 36,992.
- 🔴 §3's table claimed a reduction of $C\times$ = 64× while the page's own
  printed output three lines below said `Reduction: 57.7×`. Same cause: weights
  36,864 → 576 is exactly 64×, biases 64 → 64 is 1×, layer total
  36,928 / 640 = 57.7×. Table split into 가중치 / 편향 / MAC rows, and the
  snippet now prints both ratios with the reason attached.
- 🔴 §1 defined the operation count as $C_{out} C_{in} K^2 H W$ while §8's code
  multiplied by 2. A reader who computes from §1 gets 231,211,008 and the page
  prints 462,422,016 — exactly double, with nothing on the page saying why.
  Added a warning admonition naming the MAC/FLOP convention, giving both
  numbers, warning that papers' "569M MAdds" is the other convention, and noting
  that ratios are unaffected. §8's docstring says which one it counts.
- 🔴 "배치 합성곱" was the page's term for grouped convolution, but 배치 means
  batch everywhere else in this book (948 occurrences of 배치 크기 / 배치 정규화
  / 미니배치) — including on this page, where §5 says "배치 정규화" and §7's code
  comments `# 배치 합성곱 1×1` sit directly above `nn.BatchNorm2d`. §7's opening
  sentence used the word twice with two different meanings: "배치 합성곱은 배치
  사이에 정보가 흐르지 못하게 한다". Renamed to **묶음 합성곱**, which is the
  word the page's own prose already used ($G$개의 묶음으로 나누고), plus a
  `!!! note` at the top distinguishing 묶음 from 배치. 21 occurrences replaced.
- 🟡 §4 published 8,768 매개변수 for the depthwise-separable block and §8
  published 8,960 for the same 64 → 128 comparison. Neither mentioned the other.
  The gap is exactly the 192 biases (64 depthwise + 128 pointwise); §4 builds
  with `bias=False`, §8 with the PyTorch default. Both snippets now say which,
  and §4 carries a paragraph reconciling 8.41× with 8.24×.
- 🟡 연습문제 1's solution asked for a derivation and gave three lines ending
  "비는 $1/C_{out} + 1/K^2$이다" — which is the reciprocal of the reduction, and
  it skipped the FLOP half the exercise asked for. Rewritten to derive weights
  and MACs separately, show $H'W'$ and $C_{in}$ both cancel (so the ratio is
  independent of the input channel count), and then work the bias case: the
  depthwise-separable block has $C_{in} + C_{out}$ biases against the standard
  layer's $C_{out}$, so biases **increase**, which is why 8.41 (MAC) and 8.24
  (parameters) differ.
- 🟢 `count_ops_and_params`'s hook shadowed the builtin `input`; renamed to
  `inputs`. The FLOP arithmetic itself is right — checked that dividing by
  `groups` gives the correct $K^2 C_{in} C_{out} HW / G$ for the grouped case.
- 🔴 The opening paragraph promised "성능은 지키거나 오히려 높이면서", which
  연습문제 4 contradicts on the same page ("모델의 용량이 줄어들 수 있다").
  Replaced with what the page actually demonstrates — same input/output shape,
  fewer parameters and MACs — plus a paragraph saying accuracy is a separate
  question and pointing forward to 연습문제 4.
- 🟡 The page measured FLOPs and never warned that they are not wall-clock time,
  while §8 is titled 효율 견주기 and §1 opens with 계산 비용이 크다. Depthwise
  convolution has low arithmetic intensity and is usually memory-bound, so 8.4×
  fewer FLOPs is not 8.4× faster. Added a `!!! warning` saying so and stating
  plainly that **no timing is published on this page**. None was measured: other
  agents are training in this repo right now, so any number taken today would be
  meaningless.
- 🟡 §5's MobileNetV1Block and §7's channel_shuffle were defined and never run —
  §5 imported `torch` without using it, and §7 printed only shapes. Both now
  measure. §5: depthwise 704 + pointwise 8,448 = 9,152 against 73,984 for
  Conv+BN+ReLU, a 8.08× reduction, and a paragraph explaining why 8.08 < 8.41
  (BatchNorm appears twice on the separable path, 64·2 + 128·2 = 384 parameters,
  against once on the standard path, 128·2 = 256). §7: prints the permutation
  `[0, 3, 6, 1, 4, 7, 2, 5, 8]`, which is the ASCII diagram made concrete, and
  the block's 222 parameters against 24 × 24 × 9 = 5,184 weights for one plain
  3×3.
- 🟡 Five ASCII diagrams sat in bare ``` fences; changed to ```text.
- 🟡 §4 jumped into the two-stage split with no reason for the pointwise stage,
  and §6 presented the inverted residual without saying why a 6× expansion is
  affordable. One-sentence motivations added to both.
- 🟢 Added the missing `---` before `## 정리하며`, and made the closing table say
  가중치 rather than 매개변수 with a line giving the three measured pairs.
- Verification: **21/21 lines before, 35/35 after.** Every parameter count on the
  page was recomputed from a live `nn.Module` with
  `sum(p.numel() for p in m.parameters())`, not from a formula: 73,856 / 36,992 /
  18,560 (§2), 640 / 36,928 (§3), 73,728 / 576 / 8,192 / 8,768 (§4),
  704 / 8,448 / 9,152 / 73,984 (§5), 14,848 (§6), 222 (§7),
  73,856 / 8,960 / 18,560 (§8), 8,960 (연습문제 2). Hand-checked every comment
  asserting a number — 64 × 3 × 3 = 576, 64 × 64 × 3 × 3 = 36,864,
  64 × 128 × 9 = 73,728, 576 + 8,192 = 8,768, 32 × 6 = 192,
  6,144 + 384 + 1,728 + 384 + 6,144 + 64 = 14,848, 24 // 4 = 6,
  24 × 24 × 9 = 5,184, 128 · 64 · 9 · 56 · 56 = 231,211,008 and its double,
  1/(1/128 + 1/9) = 8.41, 1/(1/256 + 1/9) = 8.69, 1/(1/16 + 1/9) = 5.76 — all
  correct as written. `mkdocs build` was not run (nine agents share this repo).
