┌───────────────┬──────────┬──────────┐
│               │    v1    │    v2    │
├───────────────┼──────────┼──────────┤
│ Math score    │ 3.5 / 10 │ 9.5 / 10 │
├───────────────┼──────────┼──────────┤
│ Writing score │ 5.5 / 10 │ 9.0 / 10 │
└───────────────┴──────────┴──────────┘

v1 → v2  2026-09-29
Math fixes: 8 (🔴 4, 🟡 4, 🟢 0)
Writing fixes: 7 (🔴 1, 🟡 5, 🟢 1)
Skipped: 🟢 `if __name__ == "__main__": pass` with the whole program at module
level — every page in this section does it, and the four siblings updated on
2026-09-29 all left it, so fixing it here alone would make this page the odd
one out. 🟢 The module docstring names `06_cifar10_basic.py`, which does not
exist beside the page; `cnn_utils.py` is the section's only `.py`, so this is
section-wide and not this page's to fix. 🟢 `예상 시간: 1~2시간` is a wall-clock
claim; several other agents were training on this machine tonight (load average
peaked at 117 on 8 cores), so any timing measured now would be
unrepresentative — left untouched rather than replaced with a bad number. 🟢
Particle after `$…$` math (`$5 \times 5$ 핵과`) — settled house habit, recorded
in `01_mnist_dataset_score.md`. 🟢 The seed sweep in 2.2 stops at two seeds
rather than three; a third would add ~35 min to every verification run of this
page, and the within-config spread (1.12%p, 0.48%p) is already 27× smaller than
the effect being claimed. The admonition says so and asks the reader for more.

Notes
- 🔴 **The page had no output block at all.** `verify_outputs.py` did not check
  it, it skipped it: `건너뜀 … (코드나 출력 블록이 없다)`. Every number on the
  page was therefore an assertion, and the headline one was wrong by a factor of
  six. Ran the page's own code (seeded, CPU) and published all 31 lines.
- 🔴 **The published code does not learn.** The lead and 논의 both claimed
  "60~70%". The code as written gets **10.70%** — chance level for ten classes.
  The cause is in one place: `optim.SGD(model.parameters(), lr=0.001)` has no
  `momentum`, and `optim.SGD`'s default is 0. Five epochs of plain SGD at lr
  0.001 move the epoch-average loss only from 2.3045 to 2.3010, i.e. it never
  leaves ln 10 = 2.302585. Total travel over five epochs is 0.0035, 0.07% of the
  distance from the untrained loss. Per-class, eight of ten classes are at
  **exactly 0.00%** and the 1,070 correct answers are 980 frogs and 90 cats —
  the model collapsed rather than "learned a little". Rewrote the lead and 논의
  around the measured number and added the per-class loop (`classes` had been
  defined and never used) so the collapse is visible on the page.
- 🔴 **Proved the cause rather than asserting it.** Added a second code block +
  output in 2.2 that re-runs the identical architecture with the identical seed
  and `momentum=0.9` as the only change: **41.07%**, a 30.37%p gap, 3.84×. So
  the 10.70% was never a capacity ceiling, which is the exact opposite of what
  v1's third 논의 paragraph taught ("모델의 용량은 문제의 복잡함에 걸맞아야
  한다 … 62,006개가 담을 수 있는 것보다"). Replicated at seed 1 (11.82% /
  41.55%): within-config spread 1.12%p and 0.48%p against a ~30%p effect, so the
  gap is not a seed artefact. Published the 2×2 table in a `!!! warning`.
- 🔴 **연습문제 2 (now 3) 풀이 quoted numbers from a run that does not exist.**
  It said Adam at 10 epochs lifts accuracy "약 63%에서 68~72%로". The baseline is
  not 63%, it is 10.70%. Measured Adam(lr=1e-3), 10 epochs, seed 0: **62.84%** —
  which is suspiciously close to the "63%" the page gave as the *starting*
  point, suggesting the original numbers came from an Adam run and were then
  written down as the SGD baseline. Published 62.84% with the 52.14%p and
  21.77%p gaps, and noted that 21.77%p cannot all be credited to Adam because
  the epoch budget is doubled.
- 🔴 No seed anywhere. `nn.Conv2d`/`nn.Linear` init and the shuffling DataLoader
  are all random, so nothing on the page was reproducible by anyone. Added
  `torch.manual_seed(0)` above the config block. Confirmed reproducible: two
  independent runs printed identical values on all 21 shared lines.
- 🟡 연습문제 3 (now 4) gave BatchNorm code, asserted "학습이 더 빨리 수렴한다",
  and measured nothing. Counted the parameters (`BatchNorm2d(C)` adds 2C:
  12 + 32 = 44, total 62,050, +0.07%) and ran both variants at seed 0:
  BN without momentum **30.78%**, BN with momentum **57.17%**. So BN alone does
  revive the collapsed run (+20.08%p) but falls 10.29%p short of momentum alone,
  and the two together (57.17%) come in 3.98%p below the sum of the separate
  gains (10.70 + 20.08 + 30.37 = 61.15) — they overlap because both are fixing
  the same thing from different sides. The exercise now asks for exactly this
  comparison.
- 🟡 `numpy` and `matplotlib.pyplot` were imported and never used, `classes` was
  defined and never used, and the code had no `# ===` dividers. Dropped the two
  dead imports, gave `classes` a job (the per-class table), and split the program
  into five numbered sections.
- 🟡 The loss print was wrong about its own coverage. With 782 batches per epoch,
  `if (i + 1) % 200 == 0` fires at 200/400/600 only; the remaining 182 batches
  accumulate into `running_loss` and are discarded when the next epoch resets it,
  so **23.3% of every epoch never appeared in any printed line**. Added
  `epoch_loss` and one epoch-average line per epoch, which is what 2.1's table
  now reads, plus a comment stating the gap.
- 🟡 논의's dimension claim was right but its parameter claim was unsupported.
  Verified 62,006 against torch and published the layer split: conv1 456,
  conv2 2,416, fc1 48,120, fc2 10,164, fc3 850. The two conv layers are 2,872 =
  4.63% of the model while `fc1` alone is 77.61% — so "합성곱 신경망"의 무게는
  합성곱에 있지 않다, the same shape of fact 07 records as 96.74%.
- 🟡 Added the receptive field, which the page never mentioned: $r \leftarrow r +
  (k-1)j$, $j \leftarrow js$ gives 1 → 5 → 6 → 14 → 16, so each cell of the final
  5×5 map sees 16×16, exactly a quarter of the image. Checked it with autograd
  (gradient of one centre cell is non-zero on rows and columns 8–23 inclusive =
  16). This is the *same* 16×16 that `07_cifar10_advanced.md` reports for its
  four-conv net — a sharper way to say that depth here does not buy field of view.
- 🔴 (writing) 논의 was three undivided paragraphs and referred to no output,
  because there was none. Split into 2.1 손실이 ln 10 에서 내려오지 않는다 /
  2.2 걸음이 모자란 것인가, 용량이 모자란 것인가 / 2.3 32에서 5까지, 그리고
  매개변수가 놓인 자리 / 2.4 이 수를 형제 쪽들과 어떻게 놓을 것인가.
- 🟡 정리하며 was three lines, the middle one a **verbatim copy** of 논의's
  opening sentence ("이 단순한 CNN은 LeNet에서 비롯한 고전적인 구조를 따라 …") —
  the same defect already fixed on `03_cifar10_dataset.md` and
  `05_fashion_mnist_classifier.md`. Replaced with the five things the page now
  establishes.
- 🟡 The page linked to nothing, though `03_cifar10_dataset.md` and
  `07_cifar10_advanced.md` both link here. Added 03, 05, 07 and
  `ch03/mnist/04_cnn.md`.
- 🟡 Only three exercises (spec asks 4–6) and none marked 쉬움. Added 연습문제 1
  (쉬움), which is page-specific: read the last eleven output lines and say why
  10.70% must not be read as "slightly better than chance". Order is now
  쉬움 → 중간 → 중간 → 어려움.
- 🟡 Added 2.4, a ladder table of every accuracy this book has actually measured
  (99.21% / 85.84% / 54.82% / 10.70% / 41.07% / 57.17%) with the optimiser,
  epoch count and parameter count of each, plus the warning that the rows are not
  subtractable. The striking row is the last: 62,050 parameters at 57.17% beats
  the 2,168,362-parameter 07 at 54.82%.
- 🟡 **Said out loud that a sibling is now stale.** `07_cifar10_advanced.md`
  builds its lead and its whole §2.4 on "기본 쪽이 적어 놓은 60~70%" and on the
  fact that this page had no output block (lines 3, 1294, 1298, 1340, 1349,
  1433). Both premises are gone. 07 was **not** touched — this run was scoped to
  one file — so 2.4 here states the conflict and gives the replacement numbers
  (10.70% and 41.07%). 07 needs its own update; note that its own headline claim
  ("대개 75~80%에 이른다") already contradicts its own printed 54.82%, which is a
  separate defect on that page.
- 🟢 Added the missing `---` before `## 정리하며`, and comments to a code block
  that had none.
- Hand-checked every comment and inline expression that asserts a number:
  62,006 = 456 + 2,416 + 48,120 + 10,164 + 850 (and against
  `sum(p.numel() …)`); 48,120/62,006 = 77.61%; 2,872/62,006 = 4.63%;
  400 × 120 + 120 = 48,120; 182/782 = 23.3%; ln 10 = 2.302585; 2.3045 − 2.3010 =
  0.0035 and /4 = 0.00088; (2.3010 − 1.5)/0.00088 ≈ 910; 980 + 90 = 1,070 and
  1,070/10,000 = 10.70%; 41.07 − 10.70 = 30.37 and 41.07/10.70 = 3.84;
  1/(1 − 0.9) = 10; 41.55 − 11.82 = 29.73; 11.82 − 10.70 = 1.12;
  41.55 − 41.07 = 0.48; 30.37/1.12 = 27.1; 2 × 6 + 2 × 16 = 44 and 62,006 + 44 =
  62,050 (and 44/62,006 = 0.07%); 30.78 − 10.70 = 20.08; 41.07 − 30.78 = 10.29;
  10.70 + 20.08 + 30.37 = 61.15 and 61.15 − 57.17 = 3.98; 62.84 − 10.70 = 52.14;
  62.84 − 41.07 = 21.77; 2,168,362/62,050 = 34.9 and 14/5 = 2.8;
  16² / 32² = 25%. All correct as published.
- Verification: **skipped entirely before (`건너뜀 … 코드나 출력 블록이 없다`),
  45/45 lines after.**
  The page's own compute was deliberately held to two trainings (section 4 and
  section 6) so that a full verification fits inside the timeout; the seed-1,
  Adam and BatchNorm numbers were measured off-page and are cited in 2.2 and the
  solutions, the same way `03_cifar10_dataset.md` cites its channel statistics.
  No wall-clock number is published on this page, and none was measured for it.
