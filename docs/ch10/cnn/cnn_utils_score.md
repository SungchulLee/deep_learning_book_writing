┌───────────────┬──────────┬──────────┐
│               │    v1    │    v2    │
├───────────────┼──────────┼──────────┤
│ Math score    │ 5.5 / 10 │ 9.5 / 10 │
├───────────────┼──────────┼──────────┤
│ Writing score │ 6.5 / 10 │ 9.0 / 10 │
└───────────────┴──────────┴──────────┘

v1 → v2  2026-09-29
Math fixes: 6 (🔴 3, 🟡 3, 🟢 0)
Writing fixes: 5 (🔴 1, 🟡 3, 🟢 1)
Skipped: 🟢 `if __name__ == "__main__": pass` in the module. Here it is not a
style violation at all — the file *is* a library, and the guard is the
conventional way to say "nothing to run". Left as is. 🟢 `torch.load(path,
map_location=device)` in `load_model` omits `weights_only=`, which PyTorch 2.6
flipped to `True` by default; the call still works for these state dicts, but
the code block must stay byte-identical to `cnn_utils.py` and this run was told
not to touch the `.py`. Recorded below instead. 🟢 The module's author/date
header ("지은이: PyTorch CNN 실습 / 날짜: 2025년 11월") is meaningless but lives
in the `.py`, so it cannot be edited from here.

Notes
- 🔴 The page carried a 243-line module and **no output at all**, so
  `verify_outputs.py` skipped it outright ("코드나 출력 블록이 없다"). That is the
  worst place in this section for a blind spot: five pages
  (`01_mnist_dataset.md`, `02_fashion_mnist_dataset.md`,
  `03_cifar10_dataset.md`, `05_fashion_mnist_classifier.md`,
  `07_cifar10_advanced.md`) call `utils.load_data(...)`, and nothing on the page
  that documents it could go stale loudly. Added 2절 모듈 점검 — a short script
  that prints `inspect.signature(load_data)`, the three dataset branches, both
  models' flatten width and parameter count, and a `set_seed` repeat check. Ran
  it twice: byte-identical. The page now verifies **7/7 lines**, and a drifted
  signature breaks it immediately.
- 🔴 연습문제 1 (was 연습문제 1, the normalization one) gave CIFAR-10's standard
  deviation as (0.2023, 0.1994, 0.2010) as if it were the dataset statistic.
  Measured on the 50,000 training images: the all-pixel std is
  **0.2470 / 0.2435 / 0.2616**, and 0.2023 / 0.1994 / 0.2010 is the *mean of the
  per-image stds* (`x.std(dim=(1,2)).mean(dim=0)`) — agreeing to all four
  published decimals, so it is a different quantity, not a typo.
  `transforms.Normalize` divides by the all-pixel std. Corrected, and the widely
  circulated trio is now explained rather than dropped, matching what
  `03_cifar10_dataset.md` established on 2026-09-29. Ratio 0.2470 / 0.2023 =
  1.22×.
- 🔴 The same solution claimed "(거의 검은 화소인) MNIST에서는 차이가 작지만 …
  CIFAR-10에서는 차이가 더 뚜렷해진다". That is backwards on the mean, and the
  page's own parenthesis gives the reason it is backwards. Under (0.5, 0.5):
  MNIST lands at mean −0.7386 (from 0.1307), Fashion-MNIST at −0.4280 (0.2860),
  CIFAR-10 at −0.0172 / −0.0356 / −0.1070 (0.4914 / 0.4822 / 0.4465). MNIST is
  the one thrown 7× further off zero, precisely *because* it is mostly black.
  Replaced with a five-row table of measured numbers and moved CIFAR-10's real
  complaint to where it belongs — three channels with different spreads
  (0.2470 / 0.2435 / 0.2616) sharing one constant. All statistics measured here:
  MNIST 0.1307 / 0.3081, Fashion-MNIST 0.2860 / 0.3530.
- 🔴 연습문제 2 opened "`train` 함수는 매개변수를 통해 최적화기를 암묵적으로 새로
  만든다". `train(model, train_loader, loss_fn, optimizer, scheduler, device,
  epochs, ...)` creates neither — it receives both. The statement asserted the
  opposite of the code it was asking about. Restated as "스스로 만들지 않고 인자로
  받는다", and the solution now carries the arithmetic the exercise was gesturing
  at: `StepLR(step_size=1, gamma=0.7)` (what `05_` line 62 and `07_` line 42
  actually build, with `--gamma` default 0.7) leaves the learning rate at
  $0.01 \times 0.7^{14} \approx 6.8 \times 10^{-5}$ after one 14-epoch call, so
  re-creating the scheduler inside `train` restarts a second call at 147× that
  step.
- 🟡 연습문제 3 (now 5) said "혼동 행렬 방식으로 구현하라" and its own solution
  then said it does *not* build a confusion matrix — the statement and the
  answer contradicted each other on the page. Worse, the solution justified the
  counting approach as "온전한 혼동 행렬을 메모리에 만들지 않으므로 시험 집합이 커도
  효율적이다", which is false: a 10 × 10 confusion matrix is 100 integers no
  matter how large the test set is. Restated the exercise to ask for the
  counting form explicitly and to ask what it costs; the solution now says the
  cost is losing *where* the errors go, states the O(C²) fact, and shows the
  two-line confusion-matrix version with
  `confusion.diag() / confusion.sum(dim=1)`. Ran the published solution against
  an untrained `CNN` on one test batch — it executes and returns all ten keys.
- 🟡 논의 said `model.train()` "드롭아웃과 배치 정규화를 켜고", `model.eval()`
  "끈다". Neither model in this module has a single normalization layer — the
  only mode-sensitive layers are `nn.Dropout(0.25)` and `nn.Dropout(0.5)` — and
  `eval()` does not switch batch norm *off*, it switches it to the stored running
  statistics. Rewritten to describe inverted dropout ($p$, then $1/(1-p)$) as the
  only thing that changes here, with the batch-norm behaviour in a parenthesis
  marked as not applying to this module.
- 🟡 "PyTorch, CUDA, NumPy, 파이썬 내장 random의 씨앗을 모두 정하면 무작위성의 모든
  원천이 통제된다" overclaims, and the specific flag it praises does nothing on
  the hardware this section runs on. `torch.backends.cudnn.deterministic` is
  cuDNN-only, so it is inert on CPU and on Apple MPS. Added the two real gaps
  (`num_workers > 0` needs `worker_init_fn`; some GPU kernels need
  `torch.use_deterministic_algorithms(True)`) and stated the scope the module
  actually covers.
- 🟡 `load_data`'s contract was never written down on the page that documents
  it — the two flags, what the two kwargs dicts are for, and which normalization
  each branch applies. Added a 논의 paragraph, and noted the sharp edge: the
  flags are independent booleans, so `fashion_mnist=True, cifar10=True` is a
  legal call and `if cifar10:` silently wins. Checked by running it — returns
  `CIFAR10 50000`. Promoted to 연습문제 2 with a `raise ValueError` guard and the
  single-`dataset=` argument as the cleaner fix, plus the reason not to: five
  pages call the current signature.
- 🟡 정리하며 repeated 논의's opening sentence verbatim ("잘 설계된 유틸리티 모듈은
  손보기 좋은 기계 학습 프로젝트에 꼭 필요하다", identical to line 254 and line
  341) and then claimed "핵심 클래스는 `CNN`, `CNN_CIFAR10`" without one fact
  about either. Replaced with what the page now establishes: the `load_data`
  signature and its $[-1, 1]$ range, 3136 / 421,642 versus 4096 / 2,168,362, the
  95%-in-`fc1` split, and the real scope of `set_seed`.
- 🟢 Exercises went 중간 · 중간 · 어려움 with only three entries, against the 4–6
  the writer spec asks for and the 쉬움 → 어려움 order the book wants on new
  pages. Now five: 쉬움 (flatten arithmetic), 중간 (both flags true), 중간
  (normalization), 중간 (optimizer/scheduler), 어려움 (per-class accuracy).
  The new 연습문제 1 derives 3136 and 4096 from $28 \to 14 \to 7$ and
  $32 \to 16 \to 8$ and publishes the two real error messages for feeding a
  32 × 32 image to `CNN`, both captured from an actual run:
  `shape '[-1, 3136]' is invalid for input of size 4096` and
  `expected input[1, 3, 32, 32] to have 1 channels, but got 3 channels instead`.
- Hand-checked every number that is asserted rather than printed, since comments
  and prose are exactly what verification cannot see. Parameter counts summed by
  hand before running: CNN 320 + 18,496 + 401,536 + 1,290 = 421,642;
  CNN_CIFAR10 896 + 9,248 + 18,496 + 36,928 + 2,097,664 + 5,130 = 2,168,362 —
  both match the printed output. 2,168,362 / 421,642 = 5.14×; fc1 shares
  401,536 / 421,642 = 95.2% and 2,097,664 / 2,168,362 = 96.7%; the four
  convolutions total 65,568 = 3.0% of `CNN_CIFAR10`. The new comments
  `64 x 7 x 7 = 3136` and `64 x 8 x 8 = 4096` are correct, and both are also
  printed as `flatten 3136` / `flatten 4096` so they cannot drift silently.
- The section-1 code block is byte-identical to `docs/ch10/cnn/cnn_utils.py`
  (checked programmatically before and after the rewrite). The 2절 script is
  page-only and says so in its own docstring.
- The two `Files already downloaded and verified` lines in the new output are
  steady state, not a first-run artifact, and the page says which call produces
  each — `load_data` builds CIFAR-10's train and test sets with `download=True`
  apiece. MNIST and Fashion-MNIST print nothing once fetched.
- Verification: **skipped before (no output block on the page), 7/7 lines
  after.** `mkdocs build --strict` was NOT run — nine agents are working in this
  repo concurrently. Instead the page was rendered through python-markdown with
  the site's extension set: 5 `drillbox` divs, 5 `<details>` solutions, the
  statistics table renders inside its admonition, 19 arithmatex blocks, no
  literal `$$` leaks, headings h1 → h2 with no skipped level.
- No wall-clock number appears on this page and none was measured for it.
