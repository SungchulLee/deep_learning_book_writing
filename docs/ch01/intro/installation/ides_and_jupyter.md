# IDEs and Jupyter Notebook

## Learning Objectives

By the end of this section, you will be able to:

- Understand the strengths and trade-offs of major Python IDEs
- Launch and navigate Jupyter Notebook and JupyterLab
- Configure Spyder, PyCharm, and VS Code for deep learning workflows
- Use Google Colab for GPU-accelerated experiments
- Choose the right tool for different stages of a project

---

## Overview

An **Integrated Development Environment** (IDE) combines a text editor, a runtime, a debugger, and often a file explorer into a single interface. The choice of IDE shapes your daily workflow more than almost any other tooling decision, so it is worth understanding the trade-offs.

This page covers the environments most commonly used for deep learning and data analysis: **Jupyter Notebook**, **Spyder**, **PyCharm**, **VS Code**, and **Google Colab**.

---

## Jupyter Notebook

Jupyter is the de facto standard for exploratory data analysis and teaching. Its *notebook* format (`.ipynb`) interleaves executable code cells, Markdown text, $\LaTeX$ equations, and inline plots in a single document.

### Launching

```bash
# from an activated conda environment
jupyter notebook
```

This starts a local server and opens the notebook dashboard in your default browser (typically at `http://localhost:8888`).

### Key Concepts

| Concept | Description |
|---|---|
| **Cell** | The atomic unit of a notebook — either *Code* or *Markdown* |
| **Kernel** | The Python process that executes code cells; one kernel per notebook |
| **Output** | Rendered below each code cell (text, tables, images, interactive widgets) |

### Workflow Tips

- **Shift + Enter** executes the current cell and advances to the next.
- **Restart & Run All** (Kernel menu) is the single best habit for reproducibility: it guarantees the notebook runs top-to-bottom without hidden state.
- Use `%matplotlib inline` (or `%matplotlib widget` for interactive plots) as the first code cell.
- Keep notebooks short and focused; refactor reusable logic into `.py` modules and `import` them.

### JupyterLab

**JupyterLab** is the next-generation interface that supersedes the classic Notebook. It provides a tabbed, IDE-like layout with a file browser, terminal, and multiple notebooks side by side.

```bash
jupyter lab
```

All `.ipynb` files are fully compatible between the classic interface and JupyterLab.

---

## Spyder

**Spyder** (Scientific Python Development Environment) ships with Anaconda and targets users who prefer a MATLAB-like experience.

| Feature | Detail |
|---|---|
| **Variable Explorer** | Inspect arrays, DataFrames, and scalars in a table view |
| **IPython Console** | Interactive REPL embedded in the IDE |
| **Debugger** | Breakpoints, step-through, conditional breakpoints |
| **Editor** | Syntax highlighting, code completion, real-time linting |

### Launching

```bash
spyder
```

Or open it from the Anaconda Navigator GUI.

### When to Choose Spyder

Spyder is a good fit when you want persistent variable inspection (like MATLAB's workspace) and prefer a single-window application over a browser-based interface.

---

## PyCharm

**PyCharm** (by JetBrains) is a full-featured professional IDE with deep Python support.

| Feature | Detail |
|---|---|
| **Intelligent completion** | Context-aware suggestions, type inference |
| **Refactoring** | Rename, extract method/variable, inline — across the whole project |
| **Integrated VCS** | Git, GitHub, GitLab built in |
| **Scientific mode** | Jupyter-like cell execution inside `.py` files (`# %%` markers) |
| **Remote interpreters** | SSH, Docker, WSL |

PyCharm comes in two editions:

- **Community** (free, open-source) — sufficient for most deep learning work.
- **Professional** (paid) — adds database tools, web frameworks, and remote development.

### When to Choose PyCharm

PyCharm excels for **large codebases**: multi-file projects, library development, production code that needs rigorous testing and refactoring.

---

## VS Code

**Visual Studio Code** deserves mention as the most popular general-purpose editor in the Python ecosystem. With the [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python) and the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter), VS Code supports:

- `.py` editing with IntelliSense and linting
- `.ipynb` notebook execution inside the editor
- Integrated terminal, Git, and debugging
- Remote development over SSH or in containers

VS Code is free and cross-platform.

---

## Google Colab

[Google Colab](https://colab.research.google.com/) is a hosted Jupyter environment that provides **free GPU and TPU access** (with usage limits).

| Advantage | Limitation |
|---|---|
| Zero local setup | Session timeout after ~90 min of idle / ~12 h total |
| Free GPU (T4, sometimes V100) | Limited RAM and disk on the free tier |
| Google Drive integration | Package installations are ephemeral (reset each session) |
| Easy sharing via link | No persistent environment across sessions |

Colab is ideal for quick experiments, sharing reproducible demos, and working from machines without a local GPU.

---

## Comparison Matrix

| | Jupyter | Spyder | PyCharm | VS Code | Colab |
|---|---|---|---|---|---|
| **Best for** | Exploration, teaching | MATLAB-style workflow | Large projects | Versatile all-rounder | GPU access, sharing |
| **Execution model** | Cell-based | Script / console | Script / cell markers | Script / notebook | Cell-based |
| **Debugger** | Limited (in classic) | Full | Full | Full | Limited |
| **GPU** | Local only | Local only | Local / remote | Local / remote | Cloud (free tier) |
| **Setup effort** | Minimal | Minimal (ships with Anaconda) | Moderate | Moderate | None |

---

## Recommendation for This Curriculum

Most code examples in this curriculum are presented as **standalone scripts** (`.py`) that can be pasted into any editor, and as **Jupyter-style cells** for interactive exploration. We recommend:

- **Jupyter Notebook / JupyterLab** for working through chapter examples and visualising results.
- **VS Code or PyCharm** for the capstone projects (Chapters 23–28), which involve multi-file codebases.
- **Google Colab** whenever you need a GPU but do not have one locally.
