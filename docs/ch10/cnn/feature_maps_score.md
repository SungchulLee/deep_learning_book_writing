┌───────────────┬──────────┬──────────┐
│               │    v1    │    v2    │
├───────────────┼──────────┼──────────┤
│ Math score    │ 6.5 / 10 │ 9.5 / 10 │
├───────────────┼──────────┼──────────┤
│ Writing score │ 8.5 / 10 │ 9.0 / 10 │
└───────────────┴──────────┴──────────┘

v1 → v2  2026-09-28
Math fixes: 3 (🔴 1, 🟡 2, 🟢 0)
Writing fixes: 1 (🔴 0, 🟡 0, 🟢 1)
Skipped: exercise difficulty runs med → easy → hard → easy rather than ascending.
CLAUDE.md explicitly says not to renumber existing pages for this, since the
cross-references would have to move with them.

Notes
- 🔴 visualize_feature_maps crashed for max_channels ≤ 4. subplots returns a (4,)
  array when there is one row and a (rows,4) array otherwise; the code wrapped the
  one-row case as `[axes]`, a list of length one, so `axes[1]` raised IndexError.
  That is precisely the case the max_channels argument exists to produce. Replaced
  with np.atleast_1d(axes).flatten(); confirmed working at 4 and 16 channels.
- 🟡 The prose said a 64×224×224 float32 map costs "약 12.3MB" while the table two
  screens below prints 12.85 for the same tensor. 12.3 is the value in MiB and 12.85
  in MB; the page was silently using both. Now states the byte count, gives 12.85,
  and says which convention the table uses.
- 🟡 plt.show() inside visualize_feature_maps blocks forever where there is no
  display, and the function already writes the figure to disk. Replaced with
  plt.close(fig).
- 🟢 Exercise 4's solution had no `---` before `## 정리하며`. Recorded as critical at
  first on the assumption that it broke rendering; it does not. The built HTML shows
  every <details> closing correctly, since Markdown ends the block at any unindented
  line. 1,096 pages book-wide share the pattern, which is itself evidence it is the
  house style rather than a defect. Separator kept for consistency only.
- Left visualize_feature_maps and feature_map_statistics without output blocks.
  Both need a trained model and a dataset to say anything, and WRITER.md forbids
  adding examples that were not in the original.
- Arithmetic checked and correct: layer parameter counts sum to 1,145,408, the
  memory column sums to 96.14, exercise 2's ⌊(224+6−7)/2⌋+1 = 112, and the
  channel-doubling FLOPs argument (4C²·K²·HW/4 = C²K²HW) holds.
