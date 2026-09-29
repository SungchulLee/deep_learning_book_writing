┌───────────────┬──────────┬──────────┐
│               │    v1    │    v2    │
├───────────────┼──────────┼──────────┤
│ Math score    │ 3.5 / 10 │ 9.5 / 10 │
├───────────────┼──────────┼──────────┤
│ Writing score │ 5.5 / 10 │ 9.0 / 10 │
└───────────────┴──────────┴──────────┘

v1 → v2  2026-09-29
Math fixes: 8 (🔴 4, 🟡 3, 🟢 1)
Writing fixes: 7 (🔴 1, 🟡 5, 🟢 1)
Skipped: 🟢 Particle after `$…$` math: `$\sigma$은`, `$\delta$은`, `$\mathbf{z}$은`.
Read aloud 시그마/델타 end in a vowel and want 는. But the book writes
`$\sigma$은` on 16 pages against `$\sigma$는` on 3, and `$\delta$은` on 6 against
`$\delta$는` on 0 — a settled house habit, and `03_cifar10_dataset_score.md`
already recorded the decision to leave this class of thing alone. 🟢 The two
sibling pages in this section (`cnn_to_vit_bridge.md`, `non_local.md`) carry the
same "8.1.1절 / 9.3절" phantom cross-references that I fixed here; out of scope
for a single-file update, but they need the same pass. 🟢 No figure was added.
A per-stage bar of where the 2.53 M added parameters land would help, but
`figures/` does not exist in this directory and this run was not asked to
create one; the numbers are in the output block instead.

Notes
- 🔴 `$\text{Gate}_c = \sigma(s_c)$` applied sigmoid **twice** — $s_c$ was defined
  four lines earlier as $\sigma(W_1\delta(W_0\mathbf{z}))_c$, already a sigmoid
  output. The page then claimed the result lies in $(0,1)$. It does not:
  $\sigma(\sigma(t))$ is confined to **[0.5000, 0.7311]**, measured over
  $t \in [-20, 20]$ at 100,001 points. A gate that cannot go below 0.5 cannot
  turn a channel off, which is the one thing the section says it does. Fixed to
  $\text{Gate}_c = s_c$ and added a warning that prints the real range.
- 🔴 The backprop formula was wrong in two independent ways:

      ∂L/∂X_{c,i,j} = s_c ∂L/∂Y_{c,i,j} + X_{c,i,j} · ∂L/∂s_c · 1/HW

  (a) the stray factor $X_{c,i,j}$ — it belongs inside $\partial L/\partial s_c$,
  which is already a sum over $i,j$, so writing it again double-counts; (b) the
  missing sum over $c'$ — the excitation FCs are dense, so $z_c$ moves **every**
  channel's gate, not just its own. Checked against autograd in float64: the
  corrected form matches to 2.8e-17, the page's form is off by **2.7972** on a
  problem whose whole gradient has magnitude 1.4249 — larger than the quantity
  it approximates. Even the diagonal-only version (keeping the $c'$ sum's
  $c'=c$ term alone) is off by 0.0174 at $C = 8$. All three comparisons are now
  printed on the page.
- 🔴 "이 항등 연결이 … 기울기가 안정되게 흐르도록 보장한다" — an SE block has no
  identity connection. The direct path carries coefficient $s_c$, not 1, and
  $s_c < 1$ always, so SE *attenuates* the gradient rather than preserving it.
  Measured $s \in [0.3418, 0.6783]$ in the page's own example. Replaced with a
  warning saying so, and tied it to §5: what protects the gradient is the
  residual connection SE sits inside, not SE.
- 🔴 `$\mathbf{y} = \mathbf{x} + F_3(\text{SE}(F_2(F_1(\mathbf{x}))))$` put SE in the
  middle of the residual branch, while the prose above it said SE goes *after*
  the skip connection — two different placements, and neither is the one Hu et al.
  use. Corrected to $\mathbf{y} = \mathbf{x} + \text{SE}(F_3(F_2(F_1(\mathbf{x}))))$
  and explained why the alternative reading is worse: gating the sum scales the
  identity path too, and over ResNet-50's $3+4+6+3 = 16$ bottlenecks that is
  $0.5^{16} \approx 1.5 \times 10^{-5}$.
- 🔴 (parameter-overhead claim, the one this page is usually written around)
  연습문제 2 answered "$W_1$ 4096 + $W_2$ 4096 = 8192개". Built the module and
  counted: `sum(p.numel() for p in SEBlock(256, 16).parameters())` = **8,464**.
  The 272 missing are the biases $\mathbf{b}_1$ (16) and $\mathbf{b}_2$ (256).
  The trap is that both round to "약 1.4%" of a $3\times3$ conv (1.435% vs
  1.389%), so the wrong count never looks wrong — exactly the failure mode
  `03_cifar10_dataset_score.md` recorded for the 3× / 2.8× channel claim.
  The exercise now asks for both numbers and shows where they diverge: at
  $C = 64$, $r = 16$ the biases are 68 of 580 parameters, 11.7%.
- 🟡 "$r=16$에서는 여기 부분이 가장 크지만" is false for three of ResNet-50's four
  stages. $CHW$ vs $2C^2/r$ reduces to $HW$ vs $2C/r$; at $r=16$ pooling wins
  whenever $HW > C/8$. Measured: layer1 802,816 vs 8,192 (98×), layer2 401,408
  vs 32,768 (12×), layer3 200,704 vs 131,072 (1.5×), layer4 100,352 vs 524,288
  — the crossover happens once, at layer4. Replaced the sentence with the
  derivation, the table, and a new 연습문제 3 that asks for the condition.
- 🟡 The 축소 비율 table's 매개변수 column ($C^2$, $C^2/4$, $C^2/8$, $C^2/16$) is the
  right leading term but the 표현력 and 속도 columns were unsourced — and 속도 is
  contradicted by the page's own claim that the cost is negligible. Replaced both
  columns with counts from real modules at $C = 256$: 65,920 / 16,672 / 8,464 /
  4,360, i.e. 11.176% / 2.827% / 1.435% / 0.739% of the conv. Also recorded that
  doubling $r$ from 16 to 32 does **not** halve the total — 8,464 → 4,360 is
  1.94×, because $\mathbf{b}_2$ has $C$ entries regardless of $r$.
- 🟡 "ImageNet 정확도를 보통 1~2%p 올려 준다" asserted a gain with nothing behind it.
  Hu et al. (CVPR 2018) report 24.80% → 23.29% top-1 error, 1.51%p, but from a
  single training run with no seed-to-seed spread. Per this book's rule a
  difference smaller than the spread is not a difference, and no spread exists
  here to compare against. Nine other agents are training in this repo right now,
  so a measurement taken today would not be trustworthy either. The page now
  attributes the number to the paper and says plainly that it is not a number
  this book measured — while pointing out that the *cost* side, which the page
  does measure, is exact and seed-independent.
- 🟡 "계산 부담을 거의 늘리지 않고" conflated two different overheads. Built
  SE-ResNet-50 by wrapping every torchvision `Bottleneck` and counted:
  25,557,032 → **28,088,024** parameters, **+9.90%**, while conv+linear
  multiply-adds go 4.0892 G → 4.0917 G, **+0.062%**. The page now separates them
  in the opening paragraph, §4, §6 and 정리하며. 62.4% of the 2,530,992 added
  parameters (1,579,392) sit in layer4's three blocks, where $C = 2048$.
- 🟡 The MAC counter I first wrote reproduced the standard GFLOPs convention —
  conv and linear only — which silently omits the two things SE actually adds at
  every spatial position: the pooling and the channel-wise multiply. That is
  0.0110 G, **4.4× larger** than the 0.0025 G the FC layers add, and it turns
  +0.062% into **+0.331%**. Split the counter in two and added a warning, because
  this is the same omission the published GFLOPs tables make.
- 🟢 Notation: body used $W_0, W_1$ while 연습문제 1 used $W_1, W_2$ for the same
  two matrices. Unified on $W_1, W_2$ (the paper's names) and named the biases,
  which the parameter count needs.
- 🔴 (writing) The page had no runnable code and no output anywhere — every
  numeric claim on it was an assertion. Added one seeded program in §4 that
  checks each of them, with its full output. This is what let the four math
  🔴s above be found rather than argued about.
- 🟡 `$\odot$은 채널별 성분곱을 뜻한다` without saying that the two operands have
  different shapes. It is not a Hadamard product: $\mathbf{s}$ has length $C$ and
  broadcasts over $H\times W$. Added $Y_{c,i,j} = s_c X_{c,i,j}$ and a check that
  picks channel 3 and confirms output/input is one constant across all 49 cells
  ($s_3 = 0.4635$) — which distinguishes a multiply from an add.
- 🟡 "여기서 각 기호는 다음과 같다." was followed not by a symbol list but by
  $\delta(W_0\mathbf{z}) = \max(0, W_0\mathbf{z})$, a definition of something
  already defined. Rewritten.
- 🟡 관련 주제 pointed at "8.1.1절", "8.1.4절", "11장". This page is in 11.3, there
  is no 8.1.4, and none of the three was a link. Replaced with six working
  relative links (cnn_to_vit_bridge, non_local, appendix CBAM, ResNet 구현,
  항등 사상, 자기 어텐션), each with a clause saying why to go there.
- 🟡 정리하며 said only "이 마당은 핵심 개념, SE 블록의 구조, 수학적 성질, 계산량
  분석을 차례로 짚었다" — the section headings back in list form. Rewritten to
  carry the page's four results, and the missing `---` before it added.
- 🟡 연습문제 difficulty ran 어려움 → 쉬움 → 어려움 → 어려움, and 연습문제 3's
  solution was four lines of code with no answer to the question it asked
  ("ResNet 병목에 넣어라" — no bottleneck appeared). Reordered to 쉬움 → 중간 →
  중간 → 어려움 → 어려움, added the missing `SEBottleneck`, and added 연습문제 3
  on the pooling/FC crossover. No cross-reference in the book cites these by
  number (checked), so renumbering was safe.
- 🟢 연습문제 2's solution claimed a following batch norm undoes the shrinkage
  from gating. In the ResNet-SE placement SE comes *after* `bn3`, so no BN
  follows inside the block. Replaced with what is actually true: the gate only
  presses down (σ's ceiling is 1), so "excitation" means relatively-less-pressed,
  and the following layers' weights absorb the scale.
- Hand-checked every code comment that asserts a shape or a count, since
  verification cannot see comments: `(N, C, H, W) → (N, C)` (printed as
  `(2, 256)`), `여기: (N, C)` (printed), `J … # (C, C)` (jacobian of a
  $(1,C)\to(1,C)$ map is $(1,C,1,C)$; `[0,:,0,:]` is $(C,C)$ — confirmed by the
  formula matching autograd), `y = x + SE(F(x))`, `풀링 CHW + 채널별 곱 CHW`.
  Arithmetic in prose re-derived independently: 2,530,992 = 25,392 + 133,248 +
  792,960 + 1,579,392; 1,579,392/2,530,992 = 62.4%; 526,464/8,464 = 62.2 (not 64,
  because biases again); 8,464/4,360 = 1.94; $2C/r = C/8$ at $r = 16$;
  3,136 / 784 / 196 / 49 against 32 / 64 / 128 / 256.
- Verification: the v1 page had **no code and no output block at all**, so
  `verify_outputs.py` skipped it (0 lines checked, and nothing on it was
  checkable). v2 verifies **35/35 줄**. Runtime is 2.8 s; no download is needed
  (`resnet50(weights=None)`). `mkdocs build` was not run — nine agents are
  working in this repo concurrently and the build is done centrally.
