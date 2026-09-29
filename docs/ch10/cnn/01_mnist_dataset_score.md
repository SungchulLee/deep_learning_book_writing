┌───────────────┬──────────┬──────────┐
│               │    v1    │    v2    │
├───────────────┼──────────┼──────────┤
│ Math score    │ 6.0 / 10 │ 9.5 / 10 │
├───────────────┼──────────┼──────────┤
│ Writing score │ 7.0 / 10 │ 9.0 / 10 │
└───────────────┴──────────┴──────────┘

v1 → v2  2026-09-29
Math fixes: 5 (🔴 3, 🟡 2, 🟢 0)
Writing fixes: 4 (🔴 0, 🟡 3, 🟢 1)
Skipped: 🟢 `if __name__ == "__main__": pass` with the whole program at module
level. It reads like a style violation, but every sibling dataset page in this
section does the same (`02_fashion_mnist_dataset.md` line 117,
`03_cifar10_dataset.md` line 102), so fixing it here alone would make this page
the odd one out. 🟢 3절 draws an 8×8 grid of digits with `plt.show()` and the
page never shows the resulting image; adding one would mean creating
`figures/`, which this run was told not to do.

Notes
- 🔴 The page printed `Total training images: 60032` and it verified, because
  the code really did compute `len(trainloader) * cfg.batch_size` = 938 × 64.
  The dataset has 60,000 images — the last batch holds 32, not 64 — and the
  page said 60,000 in two other places (the code comment four lines above,
  "학습 집합: 이미지 60,000장", and the 논의 paragraph). So verification passed
  on a number the page itself contradicted twice. Switched to
  `len(trainloader.dataset)`, output now reads 60000, and the comment above the
  line spells out the 60032 trap so the shortcut is not re-introduced.
- 🔴 Line 112 comment claimed `image.squeeze()` converts "(C, H, W)에서
  (H, W, C)로". `squeeze()` transposes nothing — it drops the singleton channel
  axis, giving (28, 28). For grayscale there is no permute at all, and a reader
  who carries that reading to CIFAR-10 (where `.permute(1, 2, 0)` really is
  needed — see `cnn_utils.py` line 217) will write it wrong. Rewritten to say
  matplotlib accepts (H, W) or (H, W, C), and that C = 1 makes the squeeze
  enough.
- 🔴 "그 절을 시작으로 2장은 템플릿 학습 → …" — the four steps listed are
  sections 3.1–3.4 of chapter 3 (mkdocs.yml line 74: "3 MNIST로 보는 분류"),
  and the link immediately before is labelled 3.1. Changed to 3장 and tagged
  each step with its section number.
- 🟡 "`5`는 68.6%로 가장 나쁘며 오답의 대부분이 `3`으로 쏠린다(5 → 3이 118건)".
  Recomputed the template-matching confusion matrix: 5 has 388 errors, of which
  118 (30%) go to 3 and 63 go to 1. 118 is the largest single destination but
  nowhere near "대부분". Reworded to "오답 388건 가운데 가장 큰 몫", with the 63
  added. Everything else in that paragraph checked out exactly — 82.03%,
  1 = 96.2%, 5 = 68.6%, 4 → 9 = 116, 9 → 4 = 83 — so only the quantifier moved.
- 🟡 Exercise 3 (now 4) asked for 화소별 평균과 표준편차 — per-pixel, which for
  MNIST would be two 28×28 arrays — but its solution computes a single scalar
  pair over every pixel in the set, and the quoted 0.1307 / 0.3081 are those
  scalars. Statement changed to 전체 화소 so it matches the code. Reran the
  solution's code: Mean 0.1307, Std 0.3081, confirmed.
- 🟡 Particle agreement after Latin-script identifiers, four places:
  `ToTensor()`이 → 가, `DataLoader`이라는 → 라는, `DataLoader`은 → 는,
  `drop_last=True`을 → 를, `Normalize((0.5,), (0.5,))`이 → 가, and `$B$은` → `$B$는`.
  These are errors against the book's own usage, not a house style: `DataLoader`는
  / 라는 / 가 appears 11 times across `docs/`, `DataLoader`은 / 이라는 / 이 twice,
  both on this page. Left `$[0, 1]$으로` alone — the book is genuinely split
  there (18 vs 12), so it is a preference, not a mistake.
- 🟡 정리하며 repeated the page's own opening sentence verbatim ("MNIST 데이터셋은
  딥러닝의 'Hello World' 노릇을 한다", identical to line 3 and line 279), so the
  closing section carried no information. Replaced with the two things the page
  actually establishes — the (1, 28, 28) / $[-1, 1]$ pipeline and the 82.03%
  baseline — keeping the three-part shape the sibling dataset pages use.
- 🟢 Added 연습문제 1 (쉬움): invert the normalization on the printed
  −0.8363 to recover the raw pixel mean. The page had only 중간·중간·어려움 and
  three exercises against the 4–6 the writer spec asks for. This one is
  page-specific — it ties 5절's output block to the normalization formula in
  the 논의 and to the 0.1307 in the last exercise — rather than duplicating
  `ch03/mnist/01_template_learning.md`, which already owns the
  shifted-digit experiment (82.03% → 80.46% → 63.23%, its 연습문제 10). Checked
  the arithmetic against a real run: the sample image's raw mean is 0.081858,
  so 0.0819 is right, and 2 × 0.1307 − 1 = −0.7386.
- Two f-strings with no placeholders (`print(f"\nSample batch shape:")`) made
  plain strings. No output change.
- Page verifies 37/37 lines before and after; `mkdocs build --strict` passes in
  79s. Checked the rendered HTML for the new display-math block inside the
  `??? success` admonition — it comes out as a proper `<div class="arithmatex">`
  and 정리하며 stays outside the `<details>`.
