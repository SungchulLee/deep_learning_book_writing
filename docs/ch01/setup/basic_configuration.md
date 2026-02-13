# Basic Configuration

## Learning Objectives

By the end of this section, you will be able to:

- Organise a clean, consistent project directory structure
- Install and verify essential deep learning libraries
- Configure Jupyter Notebook for a productive workflow
- Set up shell aliases and startup checks
- Configure Git for notebook version control

---

## Overview

With Python installed and an IDE chosen, a few configuration steps will make your day-to-day workflow noticeably smoother. This page covers directory organisation, essential library installation, Jupyter customisation, and common startup checks.

---

## Directory Structure

A clean, consistent project layout prevents the "which notebook was the latest?" problem. A minimal structure that works well for deep learning coursework:

```
project/
├── data/               # raw and processed datasets
│   ├── raw/
│   └── processed/
├── notebooks/          # Jupyter notebooks (exploration, experiments)
├── src/                # reusable Python modules
│   ├── models.py
│   ├── data_utils.py
│   └── train.py
├── outputs/            # saved models, figures, logs
│   ├── checkpoints/
│   └── figures/
├── environment.yml     # conda environment specification
└── README.md
```

!!! tip
    Keep notebooks in `notebooks/` and importable code in `src/`. A notebook that grows beyond ~300 lines is a sign that logic should be refactored into a module.

---

## Installing Essential Libraries

The following one-liner installs the packages used most frequently throughout this curriculum:

```bash
# inside your activated environment
conda install numpy pandas matplotlib seaborn scikit-learn jupyter -y
```

For PyTorch with CUDA support (adjust the CUDA version to match your driver):

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 \
    -c pytorch -c nvidia -y
```

Verify the installation:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

print(f"NumPy     : {np.__version__}")
print(f"Pandas    : {pd.__version__}")
print(f"Matplotlib: {plt.matplotlib.__version__}")
print(f"PyTorch   : {torch.__version__}")
print(f"CUDA      : {torch.cuda.is_available()}")
```

---

## Jupyter Notebook Configuration

### Default Matplotlib Backend

Add this to the **first cell** of every notebook so that plots render inline:

```python
%matplotlib inline
```

For interactive (zoomable, pannable) plots in JupyterLab:

```python
%matplotlib widget
```

### Auto-Reload External Modules

When you edit a `.py` file and want the running notebook to pick up changes without restarting the kernel:

```python
%load_ext autoreload
%autoreload 2
```

### Display Settings

```python
# show all columns of a DataFrame (useful for wide tables)
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# higher-resolution inline figures
%config InlineBackend.figure_format = 'retina'
```

### Jupyter Configuration File

For persistent settings, generate a config file:

```bash
jupyter notebook --generate-config
```

This creates `~/.jupyter/jupyter_notebook_config.py`. Common tweaks:

```python
# change the default browser
c.NotebookApp.browser = 'firefox'

# change the default notebook directory
c.NotebookApp.notebook_dir = '/home/user/projects'

# disable the token/password for local-only use
c.NotebookApp.token = ''
```

### Themes (Optional)

```bash
pip install jupyterthemes
jt -t onedork -fs 12 -nfs 13 -tfs 13   # dark theme example
jt -r                                    # reset to default
```

---

## Shell Aliases and Startup Checks

Adding a few aliases to your shell profile (`.bashrc`, `.zshrc`, etc.) reduces friction:

```bash
# activate the deep learning environment in one word
alias dl='conda activate dl_env'

# quick GPU check
alias gpu='nvidia-smi'

# start JupyterLab from the project root
alias jl='cd ~/projects && jupyter lab'
```

A useful startup sanity check, especially after a fresh install or system update:

```bash
# Python & conda
python --version
conda --version

# NVIDIA driver and GPU visibility
nvidia-smi

# PyTorch CUDA check
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## Git for Notebooks (Optional but Recommended)

Jupyter notebooks store outputs (images, cell results) inside the `.ipynb` JSON, which makes diffs noisy. Two tools that help:

| Tool | What it does |
|---|---|
| [**nbstripout**](https://github.com/kynan/nbstripout) | Git filter that automatically strips outputs before committing |
| [**nbdime**](https://nbdime.readthedocs.io/) | Provides notebook-aware `diff` and `merge` commands |

```bash
pip install nbstripout nbdime
nbstripout --install          # registers the filter in the current repo
nbdime config-git --enable    # enables notebook-aware diff/merge globally
```

---

## Summary

| Step | Command / Action |
|---|---|
| Organise directories | Create `data/`, `notebooks/`, `src/`, `outputs/` |
| Install core packages | `conda install numpy pandas matplotlib …` |
| Install PyTorch + CUDA | `conda install pytorch … -c pytorch -c nvidia` |
| Configure Jupyter | `%matplotlib inline`, `%autoreload 2`, retina figures |
| Verify GPU | `nvidia-smi` and `torch.cuda.is_available()` |
| Version-control notebooks | `nbstripout --install` |

With these steps complete, your environment is ready for the rest of the curriculum.
