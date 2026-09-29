┌───────────────┬──────────┬──────────┐
│               │    v1    │    v2    │
├───────────────┼──────────┼──────────┤
│ Math score    │ 4.5 / 10 │ 9.5 / 10 │
├───────────────┼──────────┼──────────┤
│ Writing score │ 6.0 / 10 │ 9.0 / 10 │
└───────────────┴──────────┴──────────┘

v1 → v2  2026-09-29
Math fixes: 8 (🔴 5, 🟡 3, 🟢 0)
Writing fixes: 7 (🔴 0, 🟡 5, 🟢 2)
Skipped: 🟢 `if __name__ == "__main__": pass` with the whole program at module
level — every page in this section does it, and the four siblings updated on
2026-09-29 all left it, so fixing it here alone would make this page the odd
one out. 🟢 The docstring's `예상 시간: 2시간` is a wall-clock claim. Several
agents were training concurrently on this machine today, so any timing measured
now would be unrepresentative; left untouched rather than replaced with a bad
number. 🟢 The two `show_batch_or_ten_images_with_label_and_predict` calls draw
ten-image grids the page never shows; adding them would mean creating
`figures/`, which this run was told not to do — comments saying what each call
draws, and why it leaves no trace in the output, went in instead. 🟢 The module
docstring names `07_cifar10_advanced.py`, which does not exist beside the page;
`cnn_utils.py` is the section's only `.py`, so this is section-wide.

Notes
- 🔴 **The page's headline number was wrong in four places, and its own output
  block said so.** The lead claimed the advanced model reaches **75~80%**
  against a basic-model **60~70%**; 논의 claimed a gain **"약 65%에서 78%로"**;
  연습문제 1 풀이 repeated 65% → 78%; 연습문제 3 풀이 predicted **80~83%** with
  augmentation. The output block 1,100 lines above prints
  `Test Accuracy: 5482/10000 (54.82%)` — and it verifies, **1123/1123 lines**,
  so four invented numbers sat next to a reproducible refutation of all of them.
  54.82% is confirmed: it is what this page prints, it is what
  `03_cifar10_dataset.md` line 167 already cites as the book's measured CIFAR-10
  number, and it reproduced exactly on a fresh 14-epoch run.
- 🔴 논의 misdescribed the architecture: "첫 블록은 padding=1인 Conv2d(3, 32, 3)
  층 두 개", "둘째 블록도 마찬가지로 Conv2d(32, 64, 3) 층 두 개". As written the
  first block would feed a 3-channel input to two layers in a row. `cnn_utils.py`
  lines 129–132 give conv1 `Conv2d(3, 32, 3)` → conv2 `Conv2d(32, 32, 3)` and
  conv3 `Conv2d(32, 64, 3)` → conv4 `Conv2d(64, 64, 3)`. Rewrote both, and said
  explicitly that the second layer of each block re-filters the first's output.
  The same error reappeared as "6개/16개 필터에서 32개/64개로", which hides two of
  the four conv layers — now 32/32/64/64.
- 🟡 연습문제 1 풀이 gave the model as "약 220만 개" and the ratio as "대략 35배"
  when both numbers are exactly knowable and one of them is printed on the page.
  Counted them in torch and published the per-layer table: conv1 896, conv2
  9,248, conv3 18,496, conv4 36,928, fc1 2,097,664, fc2 5,130, total
  **2,168,362** — matching the page's `Total parameters:` line. `SimpleCNN` is
  456 + 2,416 + 48,120 + 10,164 + 850 = **62,006**, so the ratio is
  2,168,362 / 62,006 = **34.97배**.
- 🟡 연습문제 2 풀이 justified the 0.5 dropout with "뉴런 512개짜리 층 하나만
  해도 매개변수가 200만 개가 넘는다" — true (2,097,664) but it stops one step
  short of the point. `fc1` is **96.74%** of the whole model and the four conv
  layers together are 65,568, or **3.02%**. A page that calls itself "합성곱 층
  네 개짜리 깊은 신경망" is really one fully-connected layer with a small
  convolutional front end, and the asymmetric dropout is exactly the right
  response to that. Added a 2.2 section for it.
- 🟡 **The page shipped a 1,139-line output block and the prose used none of it**
  beyond a number that was not even in it. Added 2.3, which reads the fourteen
  epoch-summary lines: +15.98%p from epoch 1 to 2, then a 0.82%p band
  (51.21–52.03%) from epoch 9 to 14, with epochs 10 and 13 *below* their
  predecessors and 14 equal to 13. That plateau is the schedule, not
  convergence — `StepLR(step_size=1, gamma=0.7)` puts epoch 9 at
  0.01 × 0.7⁸ = 5.765e-4 (1/17.3 of the start) and epoch 14 at
  0.01 × 0.7¹³ = 9.689e-5 (1/103.2). The last five epochs' learning rates sum to
  0.00112, barely a ninth of epoch 1 alone, and all fourteen sum to 0.0331
  against 14 × 0.01 = 0.14 for a constant schedule — 24%.
- 🔴/🟡 The deeper finding, now 2.4: **the page cannot claim a gain and must not
  pretend to.** It is the "advanced" half of a pair, but its measured 54.82% is
  *below* the 60~70% the basic page asserts. Neither direction is arguable:
  `06_cifar10_basic.md` has no output block at all, so its 60~70% is written, not
  measured; the two pages use different recipes (basic: plain SGD, lr 0.001,
  5 epochs; this page: SGD with momentum 0.5, lr 0.01, StepLR, 14 epochs); and
  one run is not a measurement — neither page carries a seed-to-seed spread. Per
  this run's instructions no spread was measured (other agents were training
  concurrently), so the page now says what it can defend and names what would
  have to be run instead. The strongest evidence that the 54.82% is a statement
  about the schedule rather than the architecture is on the page already: final
  **train** accuracy is 51.99%, i.e. a 2.17M-parameter model that gets barely
  half of its own training set right is underfitting, not capacity-limited.
- 🟡 정리하며 was three lines, the middle one a verbatim copy of 논의's opening
  sentence ("심화 구조는 풀링 연산마다 그 앞에 합성곱 층을 두 개씩 두어 …"), so
  the closing section carried no information — the same defect fixed on all four
  siblings this week. Replaced with the four things the page now establishes.
- 🟡 The page linked to nothing, though `03_cifar10_dataset.md` and `cnn_utils.md`
  both already link *here*. Added `06_cifar10_basic.md`,
  `05_fashion_mnist_classifier.md`, `03_cifar10_dataset.md`, `cnn_utils.md` and
  `ch03/mnist/04_cnn.md`, closing the loop.
- 🟡 Only three exercises (spec asks 4–6), all three marked 중간, none 쉬움. Now
  five, running 쉬움 → 중간 → 중간 → 어려움 → 어려움. New 연습문제 1 (쉬움) is
  page-specific: reconstruct the printed `Total parameters: 2,168,362` layer by
  layer and compare the conv share with the fc1 share. New 연습문제 4 (어려움) is
  the learning-rate schedule — write ηₖ = 0.01 × 0.7^(k−1), locate the plateau,
  and give the one number from the table that proves it is not convergence.
  연습문제 2 was rewritten from "정확도 향상이 매개변수 증가에 비례하는가" (which
  presupposed a gain that does not exist) into "can this page answer that at
  all?", and 연습문제 5's augmentation claim lost its invented 80~83% and gained
  the measured flip-asymmetry table that `03_cifar10_dataset.md` already
  publishes (0.1191 airplane to 0.1898 truck, a 1.59× spread, against
  Fashion-MNIST's 4.6×). It now also warns that adding augmentation — a
  regulariser — to a run that is already underfitting has no reason to help.
- 🟢 논의 was one undivided run of three paragraphs; split into 2.1 무엇이
  깊어졌는가 / 2.2 매개변수는 어디에 있는가 / 2.3 학습은 어디서 멈추는가 /
  2.4 "심화"가 "기본"보다 나은가. Added the missing `---` before `## 연습문제`
  and `## 정리하며`.
- 🟢 The code block carried no comments at all. Added the dataset counts and why
  each epoch's last printed line reads 49920 rather than 50000
  (781 × 64 + 16 = 50,000, and batch 781 is not a multiple of `log_interval`),
  the per-layer parameter breakdown, the StepLR decay, what each `plt.show()`
  grid draws and why it leaves no trace in the output, and why `Test Accuracy`
  appears on two consecutive lines (`compute_accuracy` prints it and also
  returns it).
- Hand-checked every number the page now asserts, against torch or arithmetic:
  32·3·3·3+32 = 896; 32·32·3·3+32 = 9,248; 64·32·3·3+64 = 18,496;
  64·64·3·3+64 = 36,928; 512·4096+512 = 2,097,664; 10·512+10 = 5,130; sum
  2,168,362 (equals `sum(p.numel() …)` on `utils.CNN_CIFAR10()`);
  65,568/2,168,362 = 3.02%; 2,097,664/2,168,362 = 96.74%; SimpleCNN 456 + 2,416 +
  48,120 + 10,164 + 850 = 62,006; 2,168,362/62,006 = 34.97; spatial trace
  32 → 32 → 32 → 16 → 16 → 16 → 8 and flatten 4096 (confirmed by pushing a tensor
  through: (2,64,8,8)); receptive field 1 → 3 → 5 → 6 → 10 → 14 → **16**, a
  quarter of the 32×32 frame by area; 0.7⁸ = 0.057648, 0.7¹³ = 0.0096889,
  1/0.7⁸ = 17.35, 1/0.7¹³ = 103.21; Σ lr(10..14) = 0.0011190, 0.01/0.0011190 =
  8.94; Σ lr(1..14) = 0.033107 vs 0.14, = 23.6%; 0.9¹³ = 0.25419 so 1/3.93;
  epoch deltas +15.98, +6.44, +3.47, +1.88, +1.37, +0.83, +0.65, +0.47, −0.10,
  +0.54, +0.28, −0.04, 0.00; band 52.03 − 51.21 = 0.82; 54.82 − 51.99 = 2.83;
  10,000 − 5,482 = 4,518 errors; 1139 output lines (93–1231) so the fold title
  `전체 출력 (1139줄)` is still right after the edits. All correct as published.
- Every number in the prose was checked against the output block line by line:
  2,168,362, 54.82%, 5482/10000, and all fourteen (Avg Loss, Acc) pairs are
  quoted exactly as printed. 99.21% comes from `ch03/mnist/04_cnn.md` and 85.84%
  from `05_fashion_mnist_classifier.md`, both of which publish those numbers.
- Verification: **1123/1123 lines before, 1123/1123 after.** The output block was
  not touched — it already reproduced exactly, which is precisely why the
  invented 75~80% survived so long. The code block gained comments only; its
  behaviour is unchanged. No wall-clock number is published on this page, and
  none was measured for it.

Cross-page problems found but NOT fixed here (this run was scoped to one file)
- `docs/ch10/index.md` line 34 advertises this page as "증강과 정칙화로 그 성능을
  끌어올리기". The page applies neither augmentation nor any regularisation beyond
  the two `Dropout` layers the basic page also lacks, and it does not raise the
  accuracy at all. The index line needs rewriting.
- `docs/ch10/cnn/06_cifar10_basic.md` publishes **no output block**, so its
  60~70% (stated three times: lead, 논의, and implicitly in 연습문제 2's
  "약 63%에서 68~72%로") is unmeasured. It is also now contradicted by the only
  measured CIFAR-10 number in the section, this page's 54.82%. That page needs a
  real run, or the range needs to be marked as unmeasured.
