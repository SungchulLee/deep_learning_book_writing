# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MkDocs Material documentation site for a book. Content is mathematical with LaTeX (MathJax) and includes Python example scripts.

## Build Commands

```bash
# Local development server with live reload
mkdocs serve

# Build static site (used in CI with --strict)
mkdocs build
mkdocs build --strict

# Install dependencies
pip install -r requirements.txt
```

## Deployment

GitHub Actions (`.github/workflows/deploy-mkdocs.yml`) auto-deploys to GitHub Pages on push to `main`. Build uses `--strict` mode, so all warnings are errors.

## Repository Structure

```
book_name/
├── CLAUDE.md
├── README.md
├── mkdocs.yml
├── requirements.txt
├── .github/
│   └── workflows/
│       └── deploy-mkdocs.yml
└── docs/
    ├── index.md
    ├── assets/
    │   └── favicon.ico
    ├── stylesheets/
    │   └── extra.css
    ├── javascripts/
    │   └── mathjax.js
    └── chapter_name/
        ├── index.md
        └── section_name/
            ├── topic.md
            ├── topic.py
            ├── module/        
            │   ├── __init__.py
            │   └── module.py
            └── figures/
                ├── fig_name.png
                └── another_fig_name.svg
```

## Navigation Structure in mkdocs.yml

The nav is organized as: **Parts → Chapters → Sections → Pages**. Each chapter typically has numbered subsections. Nav entries must point to `.md` files and `.py` files only.

Example nav entry:
```yaml
nav:
  - I Part Title:
    - 1 Chapter Title:
      - Chapter Overview: ch01/index.md
      - 1.1 Section Title:
        - Topic Title: ch01/section_title/topic_title.md
        - Another Topic Title: ch01/section_title/another_topic_title.py
```

YAML quoting rules for nav titles:
- Quote any title containing `:`, `#`, `*`, `&`, or other special characters
- Example: `'Black-Scholes Model #1'` not `Black-Scholes Model #1`
- Hash symbols in titles must always be quoted: `'Greeks: Δ, Γ, #'`
- Use full words without abbreviations, and keep titles to no more than two lines when displayed

## Content Conventions

- **Math**: Use MathJax with `$...$` (inline) and `$$...$$` (display). Display math blocks must be surrounded by empty lines (blank line, then `$$...$$`, then blank line) for proper rendering
- **Python files**: Educational style with docstrings, section dividers (`# ===`), and `if __name__ == "__main__":` pattern
- **Markdown extensions available**: admonition, details, attr_list, md_in_html, superfences, arithmatex

## MathJax Conventions and Pitfalls

- **Currency vs math**: Use `\$` for literal dollar signs (e.g. `\$100` not `$100`), use `$...$` for math. Never use bare `$` for currency inside markdown.
- **`\boldsymbol`**: Requires the AMS extension. If rendering fails, use `\mathbf` for roman letters or ensure the MathJax config loads `boldsymbol`.
- **LaTeX in headings**: Avoid `$...$` in `#` headings — MathJax in headings breaks the auto-generated TOC anchor links. Use plain text or Unicode symbols in headings instead.
- **Display math**: Always wrap `$$...$$` blocks with blank lines above and below. Missing blank lines cause MathJax to fail silently.
- **Escaping in MkDocs**: Backslashes in MathJax sometimes need doubling (`\\alpha`) inside certain admonition or HTML blocks — test when in doubt.

## Admonitions and Details

Using `pymdownx.details` for collapsible blocks (non-obvious syntax):

```markdown
??? note "Title"
    Collapsed by default.

!!! note "Title"
    Always expanded.

??? example "Click to expand"
    Hidden content here.
```

Standard admonition types: `note`, `tip`, `warning`, `danger`, `info`, `example`, `quote`.

## Common Tasks

### Adding a new section to an existing chapter
1. Create a new `section_name/` directory under `docs/chapter_name/`
2. Add `.md` or `.py` files for each topic inside the section directory
3. Add entries to `mkdocs.yml` nav under the correct chapter, pointing to `.md` or `.py` files only
4. Ensure display math has blank lines above and below, and no LaTeX in headings

### Adding a new chapter
1. Create `docs/chapter_name/` directory with an `index.md`
2. Create section subdirectories with `.md` and `.py` files
3. Add the full chapter block to `mkdocs.yml` nav under the correct Part
4. Follow the Parts → Chapters → Sections → Pages hierarchy

### Adding a Python example
1. Create the `.py` script inside the relevant `section_name/` directory
2. Use the educational style: module docstring, `# ===` section dividers, `if __name__ == "__main__":` guard
3. If the script is a reusable module, create a proper package with `__init__.py`
