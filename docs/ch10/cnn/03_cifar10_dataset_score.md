┌───────────────┬──────────┬──────────┐
│               │    v1    │    v2    │
├───────────────┼──────────┼──────────┤
│ Math score    │ 5.5 / 10 │ 9.5 / 10 │
├───────────────┼──────────┼──────────┤
│ Writing score │ 6.5 / 10 │ 9.0 / 10 │
└───────────────┴──────────┴──────────┘

v1 → v2  2026-09-29
Math fixes: 6 (🔴 3, 🟡 3, 🟢 0)
Writing fixes: 5 (🔴 0, 🟡 4, 🟢 1)
Skipped: 🟢 `if __name__ == "__main__": pass` with the whole program at module
level — every dataset page in this section does it (`01_mnist_dataset.md`,
`02_fashion_mnist_dataset.md`), and the two siblings updated on 2026-09-29 both
left it, so fixing it here alone would make this page the odd one out. 🟢 3절
draws an 8×8 grid with `plt.show()` and the page never shows the resulting
image; adding one would mean creating `figures/`, which this run was told not
to do. 🟢 Particle after `$…$` math: `$(3, 32, 32)$으로`, `$32 \times 32$으로`,
`$[-1, 1]$으로`, `$(x / 2 + 0.5)$으로`. Read aloud these all end in a vowel or
ㄹ and want 로, but the book writes `$으로` 991 times against `$로` 350, and
`32$으로` 3 times against `32$로` 0 — it is a settled house habit, and
`01_mnist_dataset_score.md` already recorded the decision to leave it. Left
alone for consistency, not because it is right.

Notes
- 🔴 The published output opened with `Downloading https://cave.cs.toronto.edu/
  kriz/cifar-10-python.tar.gz …` and `Extracting …`. Those appear only on a
  machine that has never fetched the dataset, so the page verified 16/18 — both
  lines counted as missing, permanently. (The host in the URL was wrong too:
  torchvision fetches from `www.cs.toronto.edu/~kriz/`, not `cave.`.) Removed
  and replaced with the steady-state lines, plus a `!!! note` explaining what a
  first run prints instead. Also: the page showed `Files already downloaded and
  verified` **once**, but `load_data` builds a train and a test dataset each
  with `download=True`, so it prints **twice**. Ran the code — twice confirmed.
- 🔴 논의 said "첫 합성곱 층은 … 그 층의 매개변수가 세 배가 된다", which the
  page's own 연습문제 contradicted four paragraphs later with 896 / 320 = 2.8배.
  Both cannot be right. Weights alone are exactly 3× (288 → 864); the 32 biases
  do not scale with input channels, so the layer total is 2.8×. Rewrote 논의 to
  say weights 3× / layer 2.8× with all four numbers, and extended 연습문제 2 to
  ask for the bias-free ratio so the two readings are reconciled on the page.
- 🔴 연습문제 3 (was 2) quoted CIFAR-10's channel statistics as "평균 0.4914,
  0.4822, 0.4465; 표준편차 0.2023, 0.1994, 0.2010". Measured them on the 50,000
  training images: the means are exactly right, the standard deviations are not
  — over all pixels they are **0.2470, 0.2435, 0.2616**.
  I first recorded the quoted trio as matching no statistic and therefore simply
  wrong. That was itself wrong. 0.2023, 0.1994, 0.2010 is the mean of the
  PER-IMAGE standard deviations — `x.std(dim=(1,2)).mean(dim=0)` — agreeing to
  all four published decimals. It is the most-cited CIFAR-10 constant in
  circulation, and a reader will meet it constantly, so the page now derives both
  numbers instead of dismissing one: per-image scatter and image-to-image scatter
  are different quantities, they add in quadrature
  (√(0.2023² + 0.1284²) = 0.2396, close to 0.2470; the residual is because the
  per-image stds are averaged arithmetically, not in quadrature), and
  `transforms.Normalize` divides by one dataset-wide constant, which is the
  all-pixel std. ch04 uses 0.2470 as well, so the book is now self-consistent.
  The ratio is only 1.22×, so either trains — the point is that they measure
  different things, not that one is a typo. Published the measuring code and its
  output; reran it as printed.
- 🟡 The page never stated how many images CIFAR-10 has — it printed shapes and
  class counts but no totals, so 5만/1만 appeared only inside an exercise
  statement. Added `len(trainloader.dataset)` / `len(testloader.dataset)` /
  `len(trainloader)`, giving 50000 / 10000 / 782, with the sibling's warning
  comment worked out for *this* dataset: 782 × 64 = 50048 overcounts by 48
  because the last batch holds 16, not 64. (Checked: 50000 − 781 × 64 = 16.)
  `sample_labels` was assigned and never read; it now prints
  `torch.Size([64])`. Added a `Total: 50000` line under the class table.
- 🟡 논의 asserted "단순한 CNN은 MNIST에서 99%를 내지만 CIFAR-10에서는 60~70%에
  그친다" with nothing behind it. The 60~70% range is repeated unsourced on
  `06_cifar10_basic.md` too, and `07_cifar10_advanced.md` — which actually runs
  a CIFAR-10 model — prints **54.82%**, below the whole range. Replaced with the
  book's three measured numbers: 99.21% (`ch03/mnist/04_cnn.md`), 85.84%
  (`05_fashion_mnist_classifier.md`, the same two-conv `CNN`), 54.82%
  (`07_cifar10_advanced.md`, the four-conv `CNN_CIFAR10`, 14 epochs). Noted that
  the last uses a *deeper* net than the first two, which sharpens the point
  rather than weakening it. Verified the architectures: `ch03/mnist/04_cnn.md`
  line 301 and `cnn_utils.py` line 108 are the same Conv2d(1,32)/Conv2d(32,64)
  stack, so "같은 CNN" is accurate for the first two.
- 🟡 The `.permute(1, 2, 0)` line carried no comment at all, on the one page in
  this section where the transpose is real. The grayscale siblings use
  `.squeeze()`, which drops a singleton axis and transposes nothing, and
  `01_mnist_dataset.md` had shipped a comment miscalling that a (C,H,W) →
  (H,W,C) conversion. Added a three-line comment here spelling out
  (3, 32, 32) → (32, 32, 3), and a 논의 paragraph contrasting the two cases
  explicitly so a reader arriving from the grayscale pages cannot carry the
  wrong reading forward.
- 🟡 정리하며 repeated 논의's opening sentence verbatim ("MNIST에서 CIFAR-10으로
  넘어가면 컴퓨터 비전의 근본적인 어려움 몇 가지가 드러난다"), so the closing
  section carried nothing new. Replaced with the four things the page actually
  establishes: the one-argument swap, the permute that the grayscale pages did
  not need, the 5000-per-class split, and the 99.21 → 85.84 → 54.82 ladder.
- 🟡 Opening paragraph gave "6만 장" without the 5만/1만 split the rest of the
  page depends on, and linked to neither sibling. Added the split and links to
  `01_mnist_dataset.md` and `02_fashion_mnist_dataset.md`; "대략 네 배" became
  the exact 3072 / 784 = 3.92배.
- 🟢 Added 연습문제 1 (쉬움) and promoted the augmentation exercise to 어려움, so
  the page now runs 쉬움 → 중간 → 중간 → 어려움 over four exercises instead of
  three all marked 중간. The new one inverts the page's own printed channel means
  (0.376 / 0.381 / 0.324) back to [0, 1]. Checked the arithmetic: 0.688, 0.6905,
  0.662, each 0.197–0.216 above the dataset means, and the sample's G > R order
  is the reverse of the dataset's R > G, which the solution uses to warn against
  reading a dataset off one image.
- 🟢 연습문제 4 (was 3) claimed horizontal flip is label-preserving with nothing
  to back it — the exact shape of claim that was wrong on the Fashion-MNIST
  sibling. Measured mean |x − fliplr(x)| per class on the raw [0, 1] training
  images: 0.1191 (airplane) to 0.1898 (truck), a spread of only 1.59×. On
  Fashion-MNIST the same measurement spread 4.6× (ankle boot 0.3269 vs T-shirt
  0.0707) because its footwear all faces one way. So the claim holds for
  CIFAR-10 and the exercise now says so with the table, while pointing out that
  the *absolute* values are larger here (background texture, not asymmetry).
- 🟢 Backed the "야외 장면은 파랑 채널이 높다" hand-wave in 연습문제 3 with a
  measurement: per-class channel means over all 50,000 training images. Exactly
  two classes have B > R — airplane (0.5889 > 0.5257) and ship (0.5547 > 0.4902)
  — and they are also the two brightest (0.5583, 0.5234). Frog is the most
  colour-skewed, B = 0.3452 and a channel spread of ≈ 0.125.
- 🟢 Added the missing `---` before `## 정리하며`.
- Verification: **16/18 lines before, 21/21 after.** `mkdocs build --strict`
  passes. Hand-checked every comment that asserts a number — 782 × 64 = 50048,
  last batch 16, 48 overcounted, 50,000 / 10,000, 5,000 per class train and
  1,000 per class test (counted the test targets directly), (3, 32, 32) →
  (32, 32, 3), and `/2 + 0.5` inverting $[-1, 1] \to [0, 1]$ — all correct. No
  wall-clock number is published on this page, and none was measured for it.
