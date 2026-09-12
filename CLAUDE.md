# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

MkDocs Material documentation site for a mathematical textbook. Content uses LaTeX (MathJax)
and includes Python example scripts. Three specialized review/write agents live in `agents/`;
their full instructions are in `agents/SKILL.md`.

## Build Commands

```bash
mkdocs serve          # local dev with live reload
mkdocs build          # build static site
mkdocs build --strict # used in CI — all warnings are errors
pip install -r requirements.txt
```

## Deployment

GitHub Actions (`.github/workflows/deploy-mkdocs.yml`) auto-deploys to GitHub Pages on push
to `main`. Build uses `--strict` mode.

## Repository Structure

```
book_name/
├── CLAUDE.md
├── README.md
├── mkdocs.yml
├── requirements.txt
├── agents/
│   ├── SKILL.md            ← agent orchestration hub (read this first)
│   ├── MATH_REVIEWER.md
│   ├── WRITING_REVIEWER.md
│   └── WRITER.md
├── .github/workflows/deploy-mkdocs.yml
└── docs/
    ├── index.md
    ├── assets/favicon.ico
    ├── stylesheets/extra.css
    ├── javascripts/mathjax.js
    └── chapter_name/
        ├── index.md
        └── section_name/
            ├── topic.md
            ├── topic.py
            ├── module/__init__.py
            └── figures/
```

## Agent Commands (Quick Reference)

> Full command semantics, file conventions, and agent prompts: **read `agents/SKILL.md` first**.

| Command | What it does |
|---|---|
| `review <file\|folder\|all>` | Freeze vN snapshot, run both reviewers, print to stdout. No writes. |
| `write <file\|folder\|all> [if score < N]` | Requires prior `review`. Runs reviewers in-memory, writes improved file, updates score. |
| `update <file\|folder\|all> [if score < N]` | `review` + `write` in one step. |

**Path convention**: paths are relative to `docs/` — omit the `docs/` prefix.

**Execution rules**: sequential only · batch size = 1 · commit after each file.

## File Management

| File | Git | GitHub Pages |
|---|---|---|
| `<name>.md` | ✅ committed | ✅ published |
| `<name>_score.md` | ✅ committed | ❌ excluded via `mkdocs.yml` |
| `<name>.v[0-9]*.md` | ❌ gitignored | ❌ never built |

`.gitignore` entry: `docs/**/*.v[0-9]*.md`

> The snapshot separator is a **dot**, not an underscore. An underscore pattern
> (`*_v[0-9]*.md`) also matches real content pages such as `mobilenet_v2.md`,
> `inception_v3.md`, and `yolo_v3.md`, which silently excludes them from git and
> breaks the `--strict` CI build.

`mkdocs.yml` exclusion:
```yaml
exclude_docs: |
  *_score.md
  *.v[0-9]*.md
  # Real content pages that collide with the *_score.md pattern above.
  !inception_score.md
  !01_inception_score.md
```

`*_score.md` has the same collision hazard: it also matches content pages such as
`inception_score.md`, which drops them from the site silently (mkdocs logs this at
INFO, so `--strict` still passes and the page 404s in production). Any new content
page ending in `_score.md` needs a `!` negation line here.

Commit after `update` — stage only `<name>.md` and `<name>_score.md`:
```bash
git add docs/path/to/<name>.md docs/path/to/<name>_score.md
git commit -m "update: <name>"
```

## Navigation Structure

Nav hierarchy: **Parts → Chapters → Sections → Pages**. Nav entries point to `.md` and `.py`
files only.

```yaml
nav:
  - I Part Title:
    - 1 Chapter Title:
      - Chapter Overview: ch01/index.md
      - 1.1 Section Title:
        - Topic Title: ch01/section_title/topic_title.md
```

YAML quoting: quote any title containing `:`, `#`, `*`, `&`.

## Content Conventions (Summary)

Full MathJax/admonition rules are in `agents/SKILL.md`. Key points:

- `$...$` inline math, `$$...$$` display — always blank lines above and below display math
- No blank lines *inside* `$$...$$` blocks
- No LaTeX in `#` headings (breaks TOC)
- `\$` for currency, never bare `$`
- QED: `$\square$`
- Every content page ends with `## Exercises` (interleaved solutions, collapsible)
- Python: module docstring · `# ===` dividers · `if __name__ == "__main__":` guard

## Block Scheme (정의 · 정리 · 증명 · 보기 · 문제 · 연습문제 · 풀이)

Ported from `~/Desktop/book/high_school_math`. The required extensions
(`admonition`, `pymdownx.details`, `attr_list`, `md_in_html`) are already enabled;
the box styles live in `docs/stylesheets/extra.css`.

**모든 블록은 상자에 담는다** — 저마다 클래스를 가진 `<div>` 로 감싼다.

| 블록 | 클래스 | 색 |
|---|---|---|
| 정의 | `<div class="defn" markdown>` | 회색 |
| 정리 (제목 + 주장) | `<div class="thmbox" markdown>` | 보라 — 증명과 같은 색 |
| 보기 | `<div class="exbox" markdown>` | 초록 — 풀이와 같은 색 |
| 문제 | `<div class="probox" markdown>` | 초록 |
| 연습문제 | `<div class="drillbox" markdown>` | 초록 |

색이 곧 갈래다 — **보라는 책이 증명하는 것, 초록은 독자가 손을 대는 것**이다.

**증명과 풀이는 상자 밖**에 둔다 (`??? proof "증명"`, `??? success "풀이"`) — 이미
테두리를 가진 블록이라 안에 넣으면 상자가 겹친다. 반대로 **문항을 설명하는 그림은
상자 안**에 넣는다. 여는 태그와 본문 사이, 본문과 `</div>` 사이에는 빈 줄을 둔다
(`md_in_html` 이 안쪽을 마크다운으로 읽게 하는 조건이다).

정리는 오른쪽 목차에 뜨도록 `### 정리 N. 제목 { .thm }` 으로 적는다.

```text
<div class="thmbox" markdown>          <div class="drillbox" markdown>

### 정리 1. ... { .thm }               **연습문제 1.** <span class="diff easy" …></span> …

주장 ...                                </div>

</div>                                  ??? success "풀이"   ← 풀이는 상자 밖

??? proof "증명"   ← 증명은 상자 밖
```

**난이도 점** — 보기·문제·연습문제의 굵은 도입어 바로 뒤에 색 점 하나를 붙인다.
글자는 넣지 않고 **초록(쉬움) → 노랑(중간) → 빨강(어려움)** 색만으로 나타내며,
마우스를 올리면 뜨도록 `title` 을 적는다. 쪽마다 범례를 달지 않는다.

```html
**보기 2.** <span class="diff easy" title="쉬움"></span> ...
**연습문제 4.** <span class="diff hard" title="어려움"></span> ...
```

보기는 언제나 쉬움, 문제는 중간·어려움, 연습문제는 쉬움 → 중간 → 어려움 순서로 둔다.

## Common Tasks

### Add a new section
1. Create `docs/chapter_name/section_name/` with `.md`/`.py` files
2. Add nav entries to `mkdocs.yml` (`.md` and `.py` only)

### Add a new chapter
1. Create `docs/chapter_name/` with `index.md` and section subdirectories
2. Add chapter block to `mkdocs.yml` under the correct Part

### Add a Python example
1. Create `.py` in the relevant section directory
2. Educational style: module docstring, `# ===` dividers, `if __name__ == "__main__":` guard
