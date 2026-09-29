┌───────────────┬──────────┬──────────┐
│               │    v1    │    v2    │
├───────────────┼──────────┼──────────┤
│ Math score    │ 5.5 / 10 │ 9.5 / 10 │
├───────────────┼──────────┼──────────┤
│ Writing score │ 7.0 / 10 │ 9.0 / 10 │
└───────────────┴──────────┴──────────┘

v1 → v2  2026-09-29
Math fixes: 5 (🔴 3, 🟡 2, 🟢 0)
Writing fixes: 4 (🔴 0, 🟡 3, 🟢 1)
Skipped: 🟢 `if __name__ == "__main__": pass` with the whole program at module
level. Same call as the sibling made on 2026-09-29 — every dataset page in this
section does it (`01_mnist_dataset.md` line 188, `03_cifar10_dataset.md` line
102), so fixing it here alone would make this page the odd one out. 🟢 3절 draws
an 8×8 grid with `plt.show()` and the page never shows the resulting image;
adding one would mean creating `figures/`, which this run was told not to do.
🟢 Left 연습문제 3 conceptual — it asks about a confusion matrix the page never
computes, and computing one would mean training a model, i.e. new content the
writer spec forbids. Added a cross-link to the page that does train it instead.

Notes
- 🔴 연습문제 1 (now 2) claimed "보통 바지(1번 부류)가 가장 밝은데, 옷감 화소가
  이미지의 넓은 부분을 덮기 때문이다." Ran the solution's own code on the
  training set: Trouser is −0.5542, the third *darkest* of the ten. The brightest
  is Coat at −0.2293, then Pullover at −0.2466. Only the "샌들이 가장 어둡다" half
  was right (−0.7265). Published the full ten-line table and rewrote the reading:
  brightness tracks how much of the 28×28 frame the object covers, and trousers
  land near the dark end *because* they are a narrow vertical shape — the
  opposite of the reason the old text gave. Only the two shoe classes are darker.
- 🔴 The published output carried twelve `Downloading …` / `Extracting …` lines.
  They appear only on a machine that has never fetched the dataset, so the page
  verified 20/32 — every one of the twelve counted as missing. The sibling
  `01_mnist_dataset.md` shows none. Removed them and put the fact in a `!!! note`
  so a first-run reader is not surprised by lines the page does not list.
- 🔴 `sample_images, sample_labels = next(iter(trainloader))` was dead — assigned
  and never read, so the page never stated a single fact about dataset size or
  tensor shape. Used it the way the sibling page does: the run now prints
  `Training images: 60000`, `Test images: 10000`, `Training batches: 938`, and
  `torch.Size([64, 1, 28, 28])` / `torch.Size([64])`. Carried over the sibling's
  comment warning that 938 × 64 = 60032 overcounts by the 32-image last batch —
  this page never had the `len(loader) * batch_size` bug, but it also had no
  count at all, and the shortcut is the obvious way to add one.
- 🟡 논의 asserted "숫자 MNIST에서 99%에 이르는 구조라도 Fashion-MNIST에서 학습하면
  대개 90~92%에 머문다. 이 7~9%p의 차이는 …". The book's own runs say otherwise:
  `ch03/mnist/04_cnn.md` gets 99.21% on MNIST and `05_fashion_mnist_classifier.md`
  — this section's CNN, three pages later — gets 85.84% on Fashion-MNIST, a
  13.4%p gap, not 7~9%p. Replaced the unsourced range with the two measured
  numbers, each linked, and flagged that the two runs differ in optimizer and
  epoch count so the reader does not over-read the decimals.
- 🟡 Particle disagreement inside one sentence of the same solution:
  "바지(1번 부류)**가**" against "샌들(5번 부류)**이**". 부류 ends in a vowel, and
  the page's own 논의 writes "셔츠(6번 부류)는", so 가 is the page's reading.
  Both rewritten with 가. Checked `image.squeeze()` for the sibling's other
  defect — this page has no comment about it at all, so nothing to correct; added
  the sibling's now-correct two-line comment ((H, W) or (H, W, C), C = 1 so the
  squeeze suffices) to keep the wrong reading from being invented later.
- 🟡 정리하며 repeated the page's own 논의 opening verbatim ("Fashion-MNIST는 MNIST와
  텐서의 짜임이 같아 … 훨씬 복잡하다", identical to the first line of 2절), so the
  closing section carried nothing new. Replaced with the three things the page
  actually establishes: the one-argument swap, the exactly-uniform 6000-per-class
  split, and the 99.21% → 85.84% gap.
- 🟡 논의 said the swap needs "깃발 하나만 바꾸면 된다" without naming it. Named it:
  `utils.load_data(...)` takes `fashion_mnist=True`, and inside `cnn_utils.py`
  (line 92) that is the only thing that changes — `datasets.MNIST` becomes
  `datasets.FashionMNIST`, transform and `DataLoader` untouched.
- 🟢 Added 연습문제 1 (쉬움). The page had 중간·중간·어려움 and three exercises
  against the 4–6 the writer spec asks for, and no easy entry point. The new one
  reads the page's own 10.00% column against the sibling's 9.04–11.24% column and
  asks for the constant-predictor accuracy on each. Verified every number:
  Fashion-MNIST is exactly 6000/class train and 1000/class test, so the
  constant predictor scores exactly 10.00%; MNIST test is 1135 for `1` and 892
  for `5`, so 11.35% and 8.92%.
- 🟢 연습문제 4 (was 3) asserted "옷은 대체로 좌우 대칭이므로 좌우 뒤집기는 거의 모든
  부류에 안전하지만" with nothing to back it, and Fashion-MNIST footwear all faces
  one way, which makes the flip unsafe for exactly three classes. Measured mean
  |x − fliplr(x)| per class on the raw [0, 1] images and published the table:
  Ankle boot 0.3269, Sandal 0.1554, Sneaker 0.1284 against 0.0707–0.0823 for the
  torso garments — the boot is 4.6× the T-shirt. Statement now asks for the split
  to be found numerically, and the solution says to run the two arms with several
  seeds rather than one.
- 🟢 Added the missing `---` before `## 정리하며` (exercises 1–3 had one).
- Verification: 20/32 lines before, **25/25 after**. `mkdocs build --strict`
  passes in 48s. Checked the built HTML: 4 `<details>` open and close cleanly, 4
  drillboxes render, no literal `$$` leaks, and the `정리하며` `<h2>` sits after
  the last `</details>`. Hand-checked every comment that asserts a number —
  938 × 64 = 60032, last batch 32, 60,000 / 10,000, 6,000 / 1,000 per class,
  (C, H, W) → (H, W) for C = 1, and `/2 + 0.5` inverting $[-1, 1] \to [0, 1]$ —
  all correct. No wall-clock number is published on this page.
