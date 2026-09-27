┌───────────────┬──────────┬──────────┐
│               │    v1    │    v2    │
├───────────────┼──────────┼──────────┤
│ Math score    │ 8.0 / 10 │ 8.0 / 10 │
├───────────────┼──────────┼──────────┤
│ Writing score │ 3.5 / 10 │ 9.0 / 10 │
└───────────────┴──────────┴──────────┘

v1 → v2  2026-09-28
Math fixes: 0 (🔴 0, 🟡 0, 🟢 0)
Writing fixes: 5 (🔴 2, 🟡 2, 🟢 1)
Skipped: none.

Notes
- 🔴 Heading numbers were "1. 1 / 2. 2 / 3. 3" — mangled forms of 11.1 / 11.2 / 11.3,
  which Markdown rendered as ordered-list artefacts in the TOC. Third heading also
  disagreed with the nav (비전 트랜스포머 vs CNN에서 ViT로); matched to the nav.
- 🔴 The overview described a chapter that does not exist: 22 of 33 bullets were
  unlinked and named absent pages, while 13 existing pages were missing from it.
  Now lists exactly the 24 pages the chapter contains, all linked, in nav order.
  11.1 grew large enough to warrant three sub-groupings (개념 / 변형 / 데이터셋과 분류기).
- 🟡 Opening 95-word sentence split, and given the chapter's through-line.
- 🟡 "정리하며" restated the broken numbering; now states the actual arc — each
  section assumes less about space than the one before — and points to ch12/ch13.
- 🟢 "배치 합성곱과 깊이별 분리 합성곱" → "깊이별 분리 합성곱", matching the nav
  and the page itself ("배치 합성곱" appears nowhere in it).
- Added a back-link to 3.4, where the reader first met convolution, since that is
  this chapter's starting point.
- No exercises added: index/overview pages are exempt per agents/SKILL.md.
