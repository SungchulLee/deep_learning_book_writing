┌───────────────┬──────────┬──────────┐
│               │    v1    │    v2    │
├───────────────┼──────────┼──────────┤
│ Math score    │ 5.0 / 10 │ 9.5 / 10 │
├───────────────┼──────────┼──────────┤
│ Writing score │ 6.5 / 10 │ 9.0 / 10 │
└───────────────┴──────────┴──────────┘

v1 → v2  2026-09-29
Math fixes: 6 (🔴 4, 🟡 2, 🟢 0)
Writing fixes: 6 (🔴 0, 🟡 4, 🟢 2)
Skipped: 🟢 `if __name__ == "__main__": pass` with the whole program at module
level — every page in this section does it, and the two siblings updated on
2026-09-29 (`01_mnist_dataset.md`, `03_cifar10_dataset.md`) both left it, so
fixing it here alone would make this page the odd one out. 🟢 The docstring's
`예상 시간: 1~2시간` is a wall-clock claim. Nine other agents were training
concurrently on this machine today, so any timing measured now would be
unrepresentative; left untouched rather than replaced with a bad number.
🟢 `show_batch_or_ten_images_with_label_and_predict` draws a ten-image grid the
page never shows; adding it would mean creating `figures/`, which this run was
told not to do — a comment saying what it draws went in instead. 🟢 The module
docstring names `05_fashion_mnist_classifier.py`, which does not exist beside
the page; `cnn_utils.py` is the section's only `.py`, so this is section-wide.

Notes
- 🔴 The page's headline number was wrong, and its own output said so. The lead
  paragraph and 논의 both claimed Fashion-MNIST lands at **90~92%**, while the
  output block four screens below printed **85.84%** — and it verified, 1332/1332
  lines, so the false claim sat next to a reproducible refutation of itself. Both
  siblings had already been corrected to cite *this page's* 85.84%
  (`02_fashion_mnist_dataset.md` line 194, `03_cifar10_dataset.md` lines 167 and
  362), which left this page the only one in the chapter still saying 90~92%.
  Replaced with 85.84% in both places and with the 13.37%p gap against
  `ch03/mnist/04_cnn.md`'s measured 99.21%.
- 🔴 논의 named "가장 흔히 헷갈리는 쌍" as 티셔츠↔셔츠, 풀오버↔코트,
  운동화↔앵클부츠. Trained the page's own script (seed 1, MPS, reproduced
  epoch-for-epoch: Epoch 14 Avg Loss 0.4382 / Acc 84.27%, Test 8584/10000) and
  counted the confusion matrix. Summing both directions: 티셔츠↔셔츠 **260**,
  풀오버↔코트 **214**, 코트↔셔츠 **193**, 풀오버↔셔츠 **166**, and only then
  운동화↔앵클부츠 **87**. So the page promoted a fifth-place pair into the top
  three while dropping the third and fourth, both of which involve 셔츠 and both
  of which are more than twice as large. Rewrote with the measured counts; the
  real pattern is that 셔츠 appears in three of the top four.
- 🔴 연습문제 1 (now 2) 풀이 stated the three hardest classes as "셔츠(약
  70~75%), 티셔츠/윗옷(약 80~85%), 코트(약 82~86%)". Measured per-class accuracy
  on the same run: **셔츠 54.90%, 풀오버 74.00%, 티셔츠/윗옷 82.70%**, with 코트
  fourth at 82.80%. Two errors at once — 풀오버 (the actual runner-up, never
  mentioned) was displaced by 코트, and the 셔츠 figure was off by about 15%p in
  a range that does not contain the true value. Published the full ten-class
  table. The finding it was groping at is sharper than the guess: the four worst
  classes are exactly the four sleeved upper-body garments, the other six all sit
  above 87.80%, and 878 of the 1,416 errors (62.0%) fall among those four —
  5.3× the 167 among the three shoe classes. 셔츠's 451 errors go 185 → 티셔츠,
  123 → 코트, 92 → 풀오버, i.e. 400 of 451 (88.7%) into the other three.
- 🔴 연습문제 3 (now 4) 풀이 taught early stopping two ways wrong. It called
  `utils.compute_accuracy(model, testloader, …)` and named the result `val_acc`
  — selecting the stopping epoch on the test set, after which the page's own
  85.84% is no longer a held-out number. And the comment `# 가장 좋은 모델
  가중치 저장` sat on a branch that saved nothing, while the closing prose claimed
  "검증 성능이 가장 좋은 지점에서 멈추면 …": with patience 3 the loop exits three
  epochs *past* the best and, with no checkpoint, returns the worse weights. Split
  the training set 54,000/6,000 with `random_split`, added `copy.deepcopy` of
  `best_state` plus a `load_state_dict` restore after the loop, and named both
  traps in the prose. Also noted that on this page's own 14-epoch schedule early
  stopping barely fires, since the learning rate dies before overfitting starts.
- 🟡 논의 claimed `fashion_mnist=True` "데이터의 복잡함이 미치는 영향을 다른 모든
  변수에서 떼어 낸다" and called it a controlled comparison. It is not one. There
  is no `04_mnist_classifier.md` in this section (mkdocs.yml jumps 03 → 05), so
  the 99.21% being compared against comes from `ch03/mnist/04_cnn.md`, which uses
  Adam at lr 1e-3 for 10 epochs with a single Dropout(0.25); this page uses SGD at
  lr 0.01 with momentum 0.5, StepLR, 14 epochs and three dropout sites. Rewrote to
  say the flag makes the comparison cheap to *set up*, that 13.37%p is an
  order-of-magnitude reading rather than a controlled measurement, and what one
  would actually have to run to control it. This matches the caveat both siblings
  already carry.
- 🟡 연습문제 2 (now 3) asked for two architecture changes and accepted two
  paragraphs of plausible-sounding reasons with no cost attached, which is how a
  reader ends up believing the wide-FC option is the cheap one. Counted the
  parameters in torch and published the split: conv1 320, conv2 18,496, fc1
  401,536, fc2 1,290, total **421,642** — fc1 alone is 95.2%. Verified against the
  live model (`sum(p.numel() …)` on `utils.CNN()` returns 421,642). The third conv
  block costs conv3 73,856 but shrinks fc1 to 147,584 because another pool takes
  7×7 → 3×3, giving 241,546 total, **0.57×** the original: going deeper makes the
  model *smaller*. Widening fc1 to 256 gives 803,072 + 2,570, total 824,458,
  **1.96×**. The exercise statement now asks for the parameter count so the two
  proposals can be told apart.
- 🟡 정리하며 was three lines, the middle one a verbatim copy of 논의's opening
  sentence ("이 실험에서 가장 많은 것을 말해 주는 대목이 정확도의 차이이다"), so
  the closing section carried no information — the same defect fixed on both
  siblings. Replaced with the four things the page now establishes, and added the
  missing `---` before the heading that the sibling pages have.
- 🟡 The page shipped a 1,346-line output block and the prose used **none** of it
  beyond the final accuracy. Added 2.2, which reads the fourteen epoch-summary
  lines: +11.70%p from epoch 1 to 2, then 84.00–84.44% from epoch 9 on (a 0.44%p
  band, with epochs 11 and 14 *below* their predecessors). That plateau is the
  schedule, not convergence — `StepLR(step_size=1, gamma=0.7)` puts epoch 14 at
  0.01 × 0.7¹³ = 9.69e-5, 1/103 of the start (checked: 0.7¹³ = 0.0096889).
- 🟡 Only three exercises (spec asks 4–6) and none marked 쉬움. Added 연습문제 1
  (쉬움), which is page-specific rather than generic: the last epoch printed train
  accuracy 84.27% but the test accuracy is 85.84%, higher. The solution gives the
  two reasons — three dropout sites live during training and off under
  `model.eval()`, and `utils.train` accumulates `correct` across the whole epoch
  so early-epoch weights are folded in — and notes that train < test is normal
  absent overfitting. Order is now 쉬움 → 중간 → 중간 → 어려움.
- 🟡 The page linked to nothing. Added `02_fashion_mnist_dataset.md` and
  `ch03/mnist/04_cnn.md` in the lead and 논의, closing the loop with the two
  siblings that already link here.
- 🟢 논의 was one undivided run of three paragraphs; split into 2.1 어느 부류에서
  잃는가 / 2.2 학습은 어디서 멈추는가 / 2.3 통제된 비교라고 말할 수 있는가.
- 🟢 The code block carried no comments at all — unusual for this section, whose
  dataset pages comment the load call heavily. Added the dataset counts
  (60,000 = 6,000 × 10 train, 10,000 = 1,000 × 10 test — counted the targets
  directly), the parameter breakdown, what the plotting call draws and why it
  leaves no trace in the output, and why `Test Accuracy` appears on two
  consecutive lines (`compute_accuracy` prints it and also returns it).
- Hand-checked every comment and inline expression that asserts a number:
  3136 × 128 + 128 = 401,536; 320 + 18,496 + 401,536 + 1,290 = 421,642;
  401,536/421,642 = 95.23%; 64 × 128 × 9 + 128 = 73,856; 1152 × 128 + 128 =
  147,584; 241,546/421,642 = 0.5729; 3136 × 256 + 256 = 803,072; 256 × 10 + 10 =
  2,570; 824,458/421,642 = 1.9554; 7×7 → 3×3 under a third MaxPool2d(2,2)
  (confirmed by running a tensor through: (1,128,3,3)); 99.21 − 85.84 = 13.37;
  78.15 − 66.45 = 11.70; 10,000 − 8,584 = 1,416 errors; 878/1,416 = 62.0%;
  878/167 = 5.26; 1,000 − 549 = 451 셔츠 errors and 185 + 123 + 92 = 400 =
  88.7% of them. All correct as published.
- Verification: **1332/1332 lines before, 1332/1332 after.** The output block was
  not touched — it already reproduced exactly, which is precisely why the wrong
  90~92% survived so long. No wall-clock number is published on this page, and
  none was measured for it.
