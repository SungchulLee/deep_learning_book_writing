# Python Development Environment — Installation Guides

> **Goal:** Set up a local Python development environment closely matching Google Colab, with conda, Jupyter, and VS Code.

---

## 🚀 Quick Start — Choose Your Operating System

Select your platform and follow the appropriate guide:

| Operating System | Installation Guide | Key Features |
|------------------|-------------------|--------------|
| 🍎 **macOS** | [MAC_Installation_Guide.md](MAC_Installation_Guide.md) | Homebrew + Miniforge + VS Code via terminal |
| 🪟 **Windows** | [WINDOWS_Installation_Guide.md](WINDOWS_Installation_Guide.md) | Chocolatey + Miniconda + VS Code via PowerShell |
| 🐧 **Linux** | [LINUX_Installation_Guide.md](LINUX_Installation_Guide.md) | Miniforge + VS Code via package managers |

---

## 📦 What's Included

Each guide will help you set up:

1. **Package Manager** (Homebrew/Chocolatey/apt-dnf-pacman)
2. **Conda Environment Manager** (Miniforge/Miniconda)
3. **Visual Studio Code** (installed via terminal/command-line)
4. **Jupyter Kernel** (integrated with VS Code)
5. **Python Environment** (matching Google Colab packages)

---

## 📁 Files in This Package

### Installation Guides (Choose Your OS)
- **`MAC_Installation_Guide.md`** — Complete macOS setup (Intel & Apple Silicon)
- **`WINDOWS_Installation_Guide.md`** — Complete Windows setup
- **`LINUX_Installation_Guide.md`** — Complete Linux setup (Ubuntu/Debian/Fedora/Arch)

### Environment Configuration Files
- **`01_Installation_03_env.yml`** — Conda environment specification (flexible, works on all platforms)
- **`01_Installation_04_env-explicit.txt`** — Exact hash-pinned environment (macOS Intel only)
- **`01_Installation_02_colab_raw.txt`** — Google Colab package snapshot (reference)
- **`01_Installation_06_requirements.txt`** — pip requirements (alternative method)
- **`01_Installation_05_pip-lock.txt`** — Locked pip dependencies (reference)

---

## 🎯 Installation Overview

### Common Steps Across All Platforms

1. **Install Package Manager** (platform-specific)
2. **Install Conda** (Miniforge/Miniconda)
3. **Install VS Code** (via terminal/CLI)
4. **Create Conda Environment** (from `env.yml`)
5. **Register Jupyter Kernel**
6. **Verify Installation**

### Estimated Time
- **First-time setup:** 30-45 minutes
- **With package managers already installed:** 15-20 minutes

---

## 🔍 Environment Details

Your conda environment will include:

### Core Python Stack
- **Python:** 3.10.x
- **NumPy:** 2.2.6 (with MKL optimization)
- **Pandas:** Latest from conda-forge
- **SciPy:** Latest scientific computing tools
- **Matplotlib & Seaborn:** Data visualization

### Machine Learning
- **PyTorch:** 2.6.0 (CPU by default, GPU optional)
- **Torchvision:** 0.21.0
- **Torchaudio:** 2.7.0
- **scikit-learn:** Latest ML algorithms
- **XGBoost & LightGBM:** Gradient boosting

### Deep Learning & Data Science
- **OpenCV:** Computer vision (4.10.x)
- **Jupyter Lab:** Interactive notebooks
- **IPython & IPykernel:** Enhanced Python shell

### Additional Tools
- **Visualization:** Graphviz, NetworkX
- **Geospatial:** GDAL, GeoPandas, Shapely
- **Statistics:** StatsModels, Patsy
- **Optimization:** CVXOPT, OSQP, SCS

---

## 💡 Key Features of These Guides

### ✅ What's New (vs. Original Guide)

1. **Homebrew Installation for Mac** — No more manual steps for new Macs
2. **VS Code via Terminal** — Fast, scriptable installation without web downloads
3. **Platform-Specific Instructions** — Optimized for Mac, Windows, and Linux
4. **Package Manager Integration** — Homebrew, Chocolatey, apt/dnf for easier management
5. **GPU Support Instructions** — Optional CUDA setup for NVIDIA GPUs
6. **Comprehensive Troubleshooting** — Common issues and solutions for each platform

### ✅ Best Practices Included

- Conda environment isolation
- Jupyter kernel registration
- VS Code extension setup via CLI
- Environment verification steps
- Optional GPU/CUDA setup
- Additional package installation

---

## 🛠️ Environment Creation Methods

### Method 1: From YAML (Recommended for Most Users)

```bash
conda env create -f 01_Installation_03_env.yml -n dl
conda activate dl
```

**Pros:**
- Works on all platforms (Mac, Windows, Linux)
- Flexible and human-readable
- Easy to modify and share
- Conda resolves compatible versions

**Best for:** Most users, especially on Windows, Linux, and Apple Silicon Macs

---

### Method 2: From Explicit Spec (Intel Mac Only)

```bash
conda create -n dl --file 01_Installation_04_env-explicit.txt
conda activate dl
```

**Pros:**
- Exact, byte-for-byte reproducibility
- Same package builds and hashes
- Matches original environment perfectly

**Cons:**
- Only works on macOS Intel (`osx-64`)
- Not portable to other platforms

**Best for:** Intel Mac users who need exact reproducibility

---

### Method 3: From Requirements.txt (Fallback)

```bash
conda create -n dl python=3.10 -y
conda activate dl
pip install -r 01_Installation_06_requirements.txt
```

**Pros:**
- Works when conda packages aren't available
- Familiar for pip users

**Cons:**
- May have dependency conflicts
- Slower than conda
- No binary package optimization

**Best for:** When conda installation fails or for pure-pip projects

---

## 🔄 Keeping Your Environment Updated

### Update All Packages

```bash
conda activate dl
conda update --all -y
```

### Update Specific Package

```bash
conda activate dl
conda update package-name -y
```

### Export Current Environment

```bash
conda activate dl
conda env export > my-current-env.yml
```

---

## 🆘 Getting Help

### If Installation Fails

1. Check the **Troubleshooting** section in your OS-specific guide
2. Verify all prerequisites are installed
3. Ensure you have a stable internet connection
4. Try creating the environment step-by-step instead of using the YAML

### Common Issues

- **Conda not found:** Restart terminal after installation
- **VS Code extensions not loading:** Reload VS Code window
- **Jupyter kernel not showing:** Reinstall kernel registration
- **Package conflicts:** Try conda-forge channel or use pip

### Resources

- Conda Documentation: https://docs.conda.io
- VS Code Python: https://code.visualstudio.com/docs/python/python-tutorial
- PyTorch Installation: https://pytorch.org/get-started/locally/

---

## 📊 Comparison: Your Local Environment vs. Google Colab

| Feature | Google Colab | Your Local Setup |
|---------|--------------|------------------|
| Python Version | 3.10.x | 3.10.x ✅ |
| NumPy | ~2.x | 2.2.6 ✅ |
| PyTorch | ~2.x (CUDA) | 2.6.0 (CPU/CUDA optional) ✅ |
| Jupyter | ✅ Built-in | ✅ VS Code + Jupyter |
| GPU Access | Free T4 GPU | Optional (Your NVIDIA GPU) |
| Session Time | 12-hour limit | Unlimited ✅ |
| Storage | Temporary | Persistent ✅ |
| Customization | Limited | Full control ✅ |

---

## 🎓 Next Steps After Installation

1. **Verify Installation** — Run the verification command in your guide
2. **Open VS Code** — `code .` in your project directory
3. **Create a Notebook** — Make a new `.ipynb` file
4. **Select Kernel** — Choose "Python (dl)" from kernel picker
5. **Start Coding!** — You're ready to go

### Example First Notebook

```python
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Test basic functionality
data = np.random.randn(100)
plt.hist(data, bins=20)
plt.title("Your First Plot!")
plt.show()
```

---

## 📝 Credits

This installation package provides a Google Colab-like environment for local development, with enhanced package manager integration and terminal-based VS Code installation for improved automation and ease of use.

**Environment based on:**
- Google Colab package snapshot (as of reference date)
- Conda-forge community packages
- PyTorch stable releases

---

## 📄 License

These installation guides and configuration files are provided as-is for educational and development purposes. Individual software packages are subject to their respective licenses.

---

**Happy Coding!** 🚀

For detailed installation instructions, select your operating system guide above.
