┌───────────────┬──────────┬──────────┐
│               │    v1    │    v2    │
├───────────────┼──────────┼──────────┤
│ Math score    │ 5.0 / 10 │ 9.5 / 10 │
├───────────────┼──────────┼──────────┤
│ Writing score │ 7.0 / 10 │ 9.0 / 10 │
└───────────────┴──────────┴──────────┘

v1 → v2  2026-09-28
Math fixes: 4 (🔴 2, 🟡 2, 🟢 0)
Writing fixes: 3 (🔴 1, 🟡 2, 🟢 0)
Skipped: 🟢 the page uses both `*` and `⋆` for cross-correlation. This is the
convention it states up front (line 7), so both readings are defensible and
changing it would touch a dozen formulas for no gain.

Notes
- 🔴 The cross-correlation definition contradicted the finite form four lines below
  it: `Σ_m x[m]·h[n+m]` vs `Σ_j x[i+j]·h[j]`. On x=[1,2,3,4,5], h=[1,10,100] these
  give [321,210,100] and [321,432,543]. Corrected to `Σ_m x[n+m]·h[m]`, which is
  what CNNs compute and what the finite form says.
- 🔴 The im2col reference implementation was wrong, and the page printed the proof:
  "Max difference: 3.38e+01" presented as a successful check. `unfold` returns
  (N,C,H_out,W_out,kH,kW) and the code viewed it straight to (N,C·kH·kW,L), which
  reinterprets the buffer in the wrong axis order — shapes still match, so it fails
  silently. Added the permute; verified 2.38e-06. Added a line telling the reader
  that tens instead of 1e-6 means the permute is missing, since that is exactly how
  this bug presents.
- 🟡 Two snippets asserted results without printing any: the equivariance demo
  computed out1/out2 and compared neither, and the edge-detection demo ended in a
  comment claiming which kernel catches which edge. Both now measure and print,
  and both gained the output block every other snippet on the page has.
  The equivariance output is the more useful of the two — it shows the outputs are
  NOT equal yet match exactly once shifted back, which is the actual content of
  "equivariant" and is easy to misread as "invariant".
- 🔴 Exercise 4's solution had no `---` before `## 정리하며`, so the closing summary
  rendered inside the collapsible solution.
- 🟡 84-word opening sentence split.
- Seeded three snippets that drew unseeded tensors; the asymmetric-kernel
  comparison's published 30.2844 was a value from an unseeded run and is now
  36.2866, which reproduces. Whole page verifies 19/19.
