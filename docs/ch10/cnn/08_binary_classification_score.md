┌───────────────┬──────────┬──────────┐
│               │    v1    │    v2    │
├───────────────┼──────────┼──────────┤
│ Math score    │ 4.0 / 10 │ 9.5 / 10 │
├───────────────┼──────────┼──────────┤
│ Writing score │ 5.5 / 10 │ 9.0 / 10 │
└───────────────┴──────────┴──────────┘

v1 → v2  2026-09-29

Math fixes: 10 (🔴 6, 🟡 3, 🟢 1)
Writing fixes: 5 (🔴 0, 🟡 4, 🟢 1)

Skipped: 🟢 `if __name__ == "__main__": pass` with the whole program at module
level — every page in this section does it, and the siblings updated on
2026-09-29 all left it. 🟢 The docstring's `예상 시간: 2~3시간` is a wall-clock
claim; the machine was carrying a load average above 100 while this page was
measured, so any timing taken now would be unrepresentative. Left untouched
rather than replaced with a bad number. 🟢 The docstring names
`08_binary_classification.py`, which does not exist beside the page —
`cnn_utils.py` is the section's only `.py`, so this is section-wide. 🟢 4절
draws a 10×10 grid of learning curves with `plt.show()` and the page never
shows the image; publishing it would mean creating `figures/`, which this run
was told not to do. The log-scale fix below is therefore invisible on the page
and only helps a reader who runs the code.

Notes
- 🔴 **The page published no output at all.** The script prints 48 lines and
  the page carried none of them, so `verify_outputs.py` answered
  `건너뜀 … (코드나 출력 블록이 없다)` — the page was outside the checking
  system entirely. That is what let every empirical claim below survive. The
  code now prints each pair's train size, batch count and ten-epoch loss trace,
  plus a 45-line ranking by final loss, and the page carries all 142 lines.
- 🔴 **Nothing was seeded.** `BinaryCNN()` init and `shuffle=True` both ran off
  an unseeded global RNG, so no number the page described was reproducible by
  anyone, including its author. Added `torch.manual_seed(42)` **inside** the
  pair loop rather than once at the top: all 45 models then start from the same
  initial weights (so the only difference between pairs is the data), and any
  single pair can be rerun on its own. A single seed at the top would make each
  pair's result depend on the RNG consumed by every pair before it — and
  `next(iter(DataLoader(...)))` draws from the global RNG even with
  `shuffle=False`, so the dependence is not even obvious from the loop.
- 🔴 **논의 called 1 대 7 an easy pair. It is the fifth hardest of 45.** The
  sentence was "0 대 1이나 1 대 7 같은 쉬운 쌍은 공간 구조가 근본적으로 다른
  숫자". Measured final-epoch loss puts 1 vs 7 at 0.002791, rank 5; 0 vs 1 is
  0.000173, rank 29 of 45 (middling, not extreme). Rewrote 2.2 around the
  measured ranking. The genuinely surprising entries are 1 vs 7 (5th) and
  0 vs 6 (6th, 0.002538) — and all ten of 0 vs 6's test errors run one way,
  6 read as 0.
- 🔴 **연습문제 1 풀이 named the three easiest and three hardest pairs; four of
  the six were wrong.** Claimed easiest: 0 대 1, 1 대 8, 0 대 7. Measured
  easiest: 6 대 7 (0.000009), 3 대 6 (0.000027), 0 대 7 (0.000027) — only
  0 vs 7 survives, and 1 vs 8 is actually rank 13 of 45 hardest. Claimed
  hardest: 3 대 5, 4 대 9, 7 대 9. Measured: 4 대 9, 3 대 5, 5 대 8 — 7 vs 9 is
  rank 11, behind 5 vs 8, 8 vs 9, 1 vs 7, 0 vs 6 and 5 vs 6. The solution also
  said these converge "1~2세대 안에 손실이 거의 0으로": epoch-2 losses run from
  0.002392 (6 vs 7) to 0.029795 (4 vs 9), so the statement is about the whole
  set, not about the easy three. Rewrote the exercise to read the ranking out of
  the page's own output block.
- 🔴 **연습문제 3 풀이 (now 5) named 3-8 as a low-accuracy cell. It is
  mid-pack.** Claim: "어려운 쌍(3-5, 4-9, 7-9, 3-8) 둘레에 정확도가 낮은 칸이
  조금 모여 있다". Measured 3 vs 8: 99.8992%, 2 errors out of 1,984, rank 21 of
  45 from the top, squarely mid-pack. The four genuinely worst are 2 vs 7
  (99.4175%, 12/2,060), 3 vs 9 (99.4552%, 11/2,019), 3 vs 5 (99.4742%,
  10/1,902) and 0 vs 6 (99.4840%, 10/1,938); two of those four are never
  mentioned in v1. What probably produced the wrong guess is worth keeping, so
  it went into 2.2: 3 vs 8 has the *fourth highest first-epoch* loss (0.124614)
  and finishes 20th. Also corrected "대부분의 성분이 99%를 넘고" — all 45 do,
  the lowest being 99.4175%.
- 🔴 **연습문제 3 풀이 (now 5) built the heat map with `np.zeros`.** The
  diagonal then holds 0.0 while the 45 real values sit in a 0.58%p band between
  99.4175 and 100.0000, so the colour scale spans 0–100 and every off-diagonal
  cell renders the same shade — the map destroys exactly the differences it was
  drawn to show. The v1 text said "대각선은 정의되지 않는다", which the code did
  not implement. Switched to `np.full((10, 10), np.nan)`, added the seeding line,
  the `imshow`/`colorbar` call the exercise asks for, and a paragraph naming the
  trap.
- 🟡 **`set_ylim([0, 1])` on a linear axis flattens all 90 panels.** Across all
  45 runs the loss never exceeds 0.183971 and reaches 0.000009, so every curve
  lived in the bottom 18% of its panel and the "빠른 수렴 대 느린 수렴" contrast
  the 논의 rests on could not be seen. Switched to `set_yscale('log')` with
  `set_ylim([1e-6, 1])`, which covers the measured range (the lower bound has to
  be 1e-6, not 1e-5: 6 vs 7 ends at 9.09e-6).
- 🟡 **The lower triangle is the same curve drawn twice, and the title says
  otherwise.** `axes[j, i]` replots `loss_trace` in blue under the title
  `f'{j} vs {i}'`, which reads as a second, independent experiment. It is the
  same 45 runs in 90 panels. Added a comment saying so.
- 🟡 **연습문제 2 풀이 (now 3) priced the 45-model route at "10~50배".** The
  factor is exactly 9 and falls out of the structure: each digit appears in 9 of
  the 45 pairs, so one epoch moves 9 × 60,000 = 540,000 images against 60,000
  for a single ten-class model. Added the weight count too — 45 × 105,346 =
  4,740,570 against 105,866 for the ten-class variant, 44.78×.
- 🟡 **The page never said why a "binary" classifier has two output units.**
  Given the page's title this is the one design decision a reader will copy, and
  it is the decision that produces the classic silent bug. Added 2.4 with the
  derivation (divide through by $e^{z_0}$; the 2-logit softmax loss is
  `BCEWithLogitsLoss` on $z_1 - z_0$, checked numerically: on 1,000 random logit
  pairs both return 0.9212254285812378, difference exactly 0.0) and 연습문제 4
  with the failure mode measured rather than asserted.
- 🟡 **정리하며 was three lines whose middle line was a verbatim copy of 논의's
  opening sentence** ("10개 부류 문제를 45개의 이진 부분 문제로 쪼개면 분류
  지형의 짜임이 드러난다"), the same defect fixed on the sibling pages. Replaced
  with the three things the page now establishes, and added the `---` before the
  heading that the sibling pages carry.
- 🟡 **The page linked to nothing.** Added `01_mnist_dataset.md`,
  `ch03/mnist/04_cnn.md` (whose 99.21% is the natural yardstick) and
  `ch07/logistic_regression/binary_classification.md` (which uses the
  `torch.sigmoid` + `nn.BCELoss` pair, the counterpart to this page's choice).
- 🟡 **Three exercises, none 쉬움, no class-balance check anywhere.** Now five,
  ordered 쉬움 → 중간 → 중간 → 어려움 → 어려움. The new 연습문제 1 is the
  majority-class baseline, which the page needed: the most imbalanced pair is
  1 vs 5, where a constant predictor already scores 55.99% on the test set.
- 🟢 Particle agreement after Latin-script and math tokens: `성분 $(i, j)$은` →
  `는`, `성분 $(i, j)$이` → `가`, `숫자 $i$과 $j$을` → `$i$와 $j$를`. Removed
  the unused `import numpy as np` from the main script and moved it into the
  exercise that actually uses it. Split 논의 into 2.1–2.4 and added the `# ===`
  section dividers the sibling pages use.

Measurements behind the rewrite
- Trained all 45 pairs with the published code (per-pair seed 42, CPU), five
  shards in parallel; per-pair seeding makes sharding exact, since each pair
  resets the RNG before building its model.
- Hand-checked every number that appears in the prose against the output block
  and against the raw run: 105,346 = 160 + 4,640 + 100,416 + 130 and
  100,416/105,346 = 95.32%; 9 × 60,000 = 540,000; 45 × 105,346 = 4,740,570 and
  /105,866 = 44.78; ceil(12,665/64) = 198 and ceil(11,552/64) = 181; train sizes
  run 11,263 (4 vs 5) to 13,007 (1 vs 7); 6,742/12,163 = 0.5543 and
  1,135/2,027 = 0.5599, 99.9507 − 55.99 = 43.96%p; test-set majority fraction
  spans 50.02% (3 vs 9) to 55.99% (1 vs 5), second-largest 54.23% (1 vs 6);
  100.0000 − 99.4175 = 0.5825%p; 154 errors / 90,000 = 0.171%; mean accuracy
  99.8280%; 0.0081958/0.0000090914 = 901.49; Spearman ρ(final loss, test error)
  = 0.647, ρ(epoch-1 loss, final loss) = 0.577, ρ(epoch-1 loss, test error)
  = 0.536; 29 of 45 traces rise at least once and 4 end above epoch 9;
  σ(1) = 0.7311, −log σ(1) = 0.3133, −log 0.5 = 0.6931, balanced floor 0.5032,
  3-vs-5 floor 0.46927 × 0.3133 + 0.53073 × 0.6931 = 0.5149.
- The double-sigmoid claim in 연습문제 4 was **corrected after measuring it**.
  The first draft said the model "predicts class 1 for every input"; running it
  showed that is only true along one of the two evaluation paths. The model's
  own output is still σ(z) ∈ (0, 1) and thresholding *that* at 0.5 reaches
  98.56% train accuracy by epoch 2 — the bug is survivable if you read the
  output as a probability. It is fatal if you follow the `BCEWithLogitsLoss`
  convention and apply a sigmoid at eval, which pins every prediction above 0.5.
  The loss floor is the part that holds unconditionally: measured 0.576426 then
  0.522610 on 3 vs 5, against the 0.5149 floor and the correct run's 0.005721.
  Both paths are in the solution now.
- The published block was assembled from five 1-thread shards, so the one thing
  that could still have made it irreproducible is thread-count sensitivity in
  the CPU reductions. Checked it directly by rerunning four pairs standalone at
  the default 4 threads, chosen from across the difficulty ranking — 4 vs 9
  (rank 1), 2 vs 7 (rank 9), 0 vs 1 (rank 29) and 6 vs 7 (rank 45). All four
  ten-number lines came back identical to the published ones character for
  character, including 6 vs 7's `0.000009` and 4 vs 9's `0.183971 … 0.008196`.
  Thread count does not move these numbers, and per-pair seeding means a single
  pair rerun on its own is the same experiment the full loop runs.
- Verification: **skipped before, 136 checkable lines after.** Before the
  rewrite `verify_outputs.py` printed `건너뜀 … (코드나 출력 블록이 없다)` — the
  page had no output block, so there was nothing to check and the page had never
  been checked once. It now offers 136 numeric lines.

  A full `verify_outputs.py` pass has to train all 45 classifiers. When this
  entry was first written that run was still going — the machine was carrying a
  load average between 38 and 123 from other agents, and it was getting about a
  fifth of one core — so the four-pair bit-exact spot check above stood in for
  it, covering the two numbers the prose leans on hardest (4 vs 9's 0.008196 and
  6 vs 7's 0.000009).

  **The full pass has since completed on an idle machine: `일치 … 136/136 줄`.**
  Every published number reproduces, not just the four spot-checked pairs, so
  the caveat above is now historical. Anyone rerunning it should still pass a
  timeout well above the 7200 the other pages need; on 8 idle cores it finishes
  comfortably, under load it does not.

  No wall-clock number is published on this page and none was measured for it.
