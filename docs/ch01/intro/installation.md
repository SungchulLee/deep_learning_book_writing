# Installation and Setup

## Learning Objectives

By the end of this section, you will be able to:

- Choose the right Python distribution for deep learning work
- Set up a complete development environment on macOS, Windows, or Linux
- Manage packages and environments with conda and pip
- Configure GPU acceleration (NVIDIA CUDA or Apple Silicon MPS)
- Select and configure an IDE for deep learning workflows
- Verify your installation and troubleshoot common issues

---

## Overview

A well-configured development environment is the foundation of productive deep learning work. Unlike simple Python scripts, deep learning projects require careful management of complex dependencies — GPU drivers, CUDA toolkits, numerical libraries, and ML frameworks must all work together harmoniously. This section guides you through setting up a professional-grade environment from scratch.

### Development Stack Architecture

A modern deep learning environment consists of several layers, each serving a specific purpose:

```
┌─────────────────────────────────────────────────────────────┐
│                    IDE / Editor Layer                        │
│         VS Code + Python Extension + Jupyter Extension       │
├─────────────────────────────────────────────────────────────┤
│                  Python Environment Layer                    │
│          Conda Environment (dl) with PyTorch, NumPy, etc.   │
├─────────────────────────────────────────────────────────────┤
│                 Environment Manager Layer                    │
│              Miniforge / Miniconda / Anaconda               │
├─────────────────────────────────────────────────────────────┤
│                   Package Manager Layer                      │
│          Homebrew (macOS) / Chocolatey (Windows) / apt      │
├─────────────────────────────────────────────────────────────┤
│                    Operating System                          │
│              macOS / Windows / Linux                         │
└─────────────────────────────────────────────────────────────┘
```

The **package manager** installs system-level software, including the environment manager itself. The **environment manager** (conda) creates isolated Python environments with specific package versions and handles both Python and non-Python dependencies like CUDA libraries. The **Python environment** contains all the libraries for your project, and the **IDE layer** provides code editing, debugging, and notebook support.

### Why Local Development Matters

While cloud-based notebooks like Google Colab provide an excellent starting point, local development offers several advantages for serious deep learning work:

| Aspect | Cloud Notebooks | Local Environment |
|--------|----------------|-------------------|
| **Session Persistence** | Limited (12-hour timeout) | Unlimited |
| **Data Storage** | Temporary, requires re-upload | Persistent local storage |
| **GPU Control** | Shared, variable availability | Dedicated (if available) |
| **Customization** | Limited environment control | Full system access |
| **Cost** | Free tier limitations | One-time hardware investment |
| **Privacy** | Data uploaded to cloud | Data stays local |
| **Reproducibility** | Environment may change | Fully reproducible |

---

## Choosing a Python Distribution

A Python **distribution** bundles the interpreter with a curated set of packages, a package manager, and (often) environment-management tools. Choosing the right distribution at the outset avoids hours of dependency debugging later — especially in deep learning, where packages like PyTorch, CUDA toolkits, and numerical libraries must all agree on versions and ABIs.

### Anaconda

[Anaconda](https://www.anaconda.com/products/distribution) is the most widely recommended distribution for data science and deep learning work.

| Feature | Detail |
|---|---|
| **Package count** | Ships with 1,500+ data-science packages (NumPy, Pandas, Matplotlib, SciPy, scikit-learn, …) |
| **Package manager** | `conda` — resolves native (C/C++/Fortran) dependencies automatically |
| **Environment manager** | Built-in `conda env` for creating isolated per-project environments |
| **IDEs included** | Jupyter Notebook, JupyterLab, Spyder |
| **Platforms** | Windows, macOS, Linux (x86-64 and ARM where available) |
| **License** | Free for individual use; commercial tiers for large organisations |

Installation steps:

1. **Download** the installer for your OS from [anaconda.com/products/distribution](https://www.anaconda.com/products/distribution).
2. **Run the installer.** On the "Advanced Options" screen you can optionally add Anaconda to your system `PATH`.

    !!! tip
        Adding Anaconda to `PATH` lets you call `conda` and `python` from any terminal. If you decline, you can still use the **Anaconda Prompt** shortcut that the installer creates.

3. **Verify** the installation:

    ```bash
    conda --version      # e.g. conda 24.5.0
    python --version     # e.g. Python 3.11.9
    ```

### Miniconda

**Miniconda** is a minimal installer that includes only `conda`, Python, and a handful of essential packages. Everything else is installed on demand.

| | Anaconda | Miniconda |
|---|---|---|
| Disk footprint | ~4–5 GB | ~400 MB |
| Pre-installed packages | 1,500+ | ~70 |
| Best for | Beginners; full-featured local setups | CI/CD; Docker images; experienced users who want a lean base |

Install from [docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html), then add only what you need:

```bash
conda install numpy pandas matplotlib jupyter pytorch torchvision -c pytorch
```

### Miniforge

**Miniforge** is a community-maintained minimal installer that defaults to the `conda-forge` channel. It is the recommended choice for Apple Silicon Macs and for users who prefer open-source defaults. Installation via Homebrew (macOS) or direct download (Linux) is covered in the [Environment Setup](installation/environment_setup.md) page.

### Standard CPython + pip

If you prefer not to use `conda` at all, the official CPython interpreter from [python.org](https://www.python.org/) combined with `pip` and `venv` (or `virtualenv`) is a perfectly viable alternative:

```bash
# create a virtual environment
python -m venv .venv

# activate it
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows

# install packages
pip install numpy pandas matplotlib torch torchvision
```

The main trade-off: `pip` resolves only Python-level dependencies. Native libraries (BLAS, LAPACK, CUDA) must be installed separately via your OS package manager or NVIDIA's toolkit installer.

### Recommendation

For this curriculum we assume **Anaconda** (or Miniconda/Miniforge) because:

1. `conda` can install matching CUDA toolkit versions alongside PyTorch in a single command, avoiding the most common GPU-setup pitfall.
2. Environment isolation is built in — no extra tool required.
3. Jupyter Notebook is available immediately after installation.

All code examples in subsequent sections will work with any distribution, but installation commands are given in `conda` form unless noted otherwise.

---

## Prerequisites

Before beginning installation, verify your system meets these requirements:

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8 GB | 16+ GB |
| **Storage** | 20 GB free | 50+ GB SSD |
| **CPU** | Any modern 64-bit | Multi-core |
| **GPU** | Optional | NVIDIA with CUDA support |

### Software Requirements

| OS | Version |
|----|---------|
| **macOS** | 11.0 (Big Sur) or later |
| **Windows** | 10 (64-bit) or later |
| **Linux** | Ubuntu 20.04+, Fedora 36+, or equivalent |

### Architecture Check

Before proceeding, identify your system architecture:

**macOS:**
```bash
uname -m
# Output: x86_64 (Intel) or arm64 (Apple Silicon M1/M2/M3/M4)
```

**Windows (PowerShell):**
```powershell
$env:PROCESSOR_ARCHITECTURE
# Output: AMD64 (64-bit Intel/AMD)
```

**Linux:**
```bash
uname -m
# Output: x86_64 (Intel/AMD) or aarch64 (ARM)
```

---

## Section Guide

This installation section is organised into the following pages:

| Page | Contents |
|------|----------|
| [Environment Setup](installation/environment_setup.md) | OS-specific installation of system package managers, Miniforge/Miniconda, and VS Code |
| [Virtual Environments](installation/virtual_environments.md) | Creating, managing, and sharing isolated Python environments |
| [Package Management](installation/package_management.md) | Conda vs pip, channels, dependency resolution, and reproducible specs |
| [IDEs and Jupyter](installation/ides_and_jupyter.md) | Jupyter Notebook, JupyterLab, Spyder, PyCharm, VS Code, and Google Colab |
| [Basic Configuration](installation/basic_configuration.md) | Directory structure, Jupyter settings, shell aliases, and Git for notebooks |
| [GPU Configuration](installation/gpu_configuration.md) | NVIDIA drivers, CUDA toolkit, PyTorch GPU setup, Apple Silicon MPS, and memory management |

---

## Quick Start

For those who want to get started immediately, here is the minimal path:

```bash
# 1. Install Miniforge (macOS example)
brew install miniforge
conda init "$(basename "${SHELL}")"
# Close and reopen terminal

# 2. Create environment
conda create -n dl python=3.10 -y
conda activate dl

# 3. Install core packages
conda install -c conda-forge numpy pandas scipy scikit-learn matplotlib seaborn jupyterlab ipykernel -y
conda install -c pytorch pytorch torchvision torchaudio cpuonly -y

# 4. Register Jupyter kernel
python -m ipykernel install --user --name=dl --display-name "Python (dl)"

# 5. Verify
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

For GPU support, replace the PyTorch install line with:

```bash
# NVIDIA CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Apple Silicon — no special flags needed (MPS is built in)
conda install -c pytorch pytorch torchvision torchaudio -y
```

See the individual sub-pages for detailed, OS-specific instructions and troubleshooting.

---

## Quick Reference

```bash
# check conda is available
conda --version

# check Python version
python --version

# list installed packages
conda list

# update conda itself
conda update conda
```

---

## References

1. Conda Documentation. https://docs.conda.io
2. Anaconda Distribution. https://www.anaconda.com/products/distribution
3. Miniforge. https://github.com/conda-forge/miniforge
4. PyTorch Installation Guide. https://pytorch.org/get-started/locally/
