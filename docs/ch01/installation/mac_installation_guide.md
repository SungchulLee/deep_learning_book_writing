# Installation Guide for macOS — Python Development Environment

> This guide sets up a conda environment close to Google Colab, with Homebrew, Miniforge, and VS Code installed via terminal.

---

## Prerequisites Check

Before starting, check your Mac architecture:

```bash
uname -m
# x86_64 = Intel Mac
# arm64 = Apple Silicon (M1/M2/M3/M4)
```

---

## Step 1: Install Homebrew (Package Manager for macOS)

Homebrew is essential for managing software on macOS. Install it first:

### Installation

Open Terminal and run:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Post-Installation Setup

After installation completes, follow the "Next steps" instructions shown in Terminal. Typically:

**For Apple Silicon (M1/M2/M3/M4):**
```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

**For Intel Macs:**
```bash
echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/usr/local/bin/brew shellenv)"
```

### Verify Installation

```bash
brew --version
# Should show: Homebrew 4.x.x or newer
```

---

## Step 2: Install Miniforge (Conda Package Manager)

Miniforge is recommended for macOS (especially Apple Silicon) as it uses conda-forge by default.

### Install via Homebrew

```bash
brew install miniforge
```

### Initialize Conda

```bash
conda init "$(basename "${SHELL}")"
```

Then **close and reopen** your Terminal, or run:

```bash
source ~/.zshrc  # or source ~/.bash_profile for bash
```

### Verify Installation

```bash
conda --version
# Should show: conda 24.x.x or newer

which conda
# Should point to miniforge installation
```

---

## Step 3: Install Visual Studio Code via Terminal

Installing VS Code via terminal is fast and doesn't require manual download.

### Install VS Code

```bash
brew install --cask visual-studio-code
```

### Verify Installation

```bash
code --version
```

### Enable 'code' Command (if not working)

If the `code` command doesn't work, add it manually:

1. Open VS Code (from Applications or Spotlight)
2. Press `Cmd+Shift+P` to open Command Palette
3. Type "shell command" and select **"Shell Command: Install 'code' command in PATH"**
4. Restart Terminal

### Install Essential VS Code Extensions (via Terminal)

```bash
# Python support
code --install-extension ms-python.python

# Jupyter support
code --install-extension ms-toolsai.jupyter

# Pylance (Python language server)
code --install-extension ms-python.vscode-pylance

# Python debugger
code --install-extension ms-python.debugpy
```

---

## Step 4: Create Conda Environment (Google Colab-like)

Choose one of the following methods:

### Option A: Exact Replica (Recommended for Intel Macs)

This recreates the environment **exactly** using hash-pinned builds:

```bash
cd /path/to/your/project
conda create -n dl --file 01_Installation_04_env-explicit.txt
conda activate dl
```

**Note:** The explicit spec is for Intel Macs (`osx-64`). Apple Silicon users should use Option B.

---

### Option B: Solver-Built (Recommended for Apple Silicon)

Use the readable YAML file that conda will resolve for your architecture:

```bash
cd /path/to/your/project
conda env create -f 01_Installation_03_env.yml -n dl
conda activate dl
```

This creates a flexible environment close to Colab, automatically adapted for your Mac.

---

## Step 5: Register Jupyter Kernel with VS Code

After creating the environment, register it as a Jupyter kernel:

```bash
conda activate dl
python -m ipykernel install --user --name=dl --display-name "Python (dl)"
```

Now the `dl` environment will appear in VS Code's kernel picker.

---

## Step 6: Verify Installation

Test your environment:

```bash
conda activate dl
python -c "import platform, torch, numpy; print('Python:', platform.python_version()); print('PyTorch:', torch.__version__); print('NumPy:', numpy.__version__); print('CUDA available:', torch.cuda.is_available())"
```

**Expected output:**
- Python: 3.10.x
- PyTorch: 2.6.0
- NumPy: 2.2.6
- CUDA available: False (expected on macOS)

---

## Step 7: Using Jupyter in VS Code

1. Open VS Code: `code .` in your project directory
2. Open or create a `.ipynb` file
3. Click **"Select Kernel"** in the top-right
4. Choose **"Python (dl)"** from the list
5. Start coding!

---

## Step 8: Optional — Install Additional Packages

After activating the environment, add any extra packages:

```bash
conda activate dl
pip install tensorboard transformers wandb
```

Or via conda:

```bash
conda install -c conda-forge package-name
```

---

## Quick Troubleshooting

### Issue: `conda: command not found`

**Solution:** Restart Terminal or run:
```bash
source ~/.zshrc  # or ~/.bash_profile for bash
```

### Issue: VS Code doesn't show Python environments

**Solution:**
1. Install Python extension: `code --install-extension ms-python.python`
2. Reload VS Code: `Cmd+Shift+P` → "Developer: Reload Window"
3. Select interpreter: `Cmd+Shift+P` → "Python: Select Interpreter"

### Issue: Different performance on Apple Silicon

**Solution:** 
- Use Option B (YAML-based install)
- PyTorch on Apple Silicon uses **MPS** (Metal Performance Shaders) instead of CUDA
- Check MPS availability: `torch.backends.mps.is_available()`

### Issue: Homebrew installation fails

**Solution:**
- Check internet connection
- Ensure Xcode Command Line Tools are installed:
  ```bash
  xcode-select --install
  ```

---

## File Reference

* **`01_Installation_04_env-explicit.txt`** — Exact hash-pinned snapshot (Intel Mac only)
* **`01_Installation_03_env.yml`** — Flexible, readable environment (works on all Macs)
* **`01_Installation_02_colab_raw.txt`** — Colab package snapshot for reference
* **`01_Installation_06_requirements.txt`** — Pure pip requirements (if needed)

---

## Summary: Complete Installation Flow

```bash
# 1. Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Setup Homebrew in PATH (Apple Silicon example)
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"

# 3. Install Miniforge
brew install miniforge
conda init "$(basename "${SHELL}")"
source ~/.zshrc

# 4. Install VS Code
brew install --cask visual-studio-code

# 5. Install VS Code extensions
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
code --install-extension ms-python.vscode-pylance

# 6. Create conda environment
cd /path/to/project
conda env create -f 01_Installation_03_env.yml -n dl
conda activate dl

# 7. Register Jupyter kernel
python -m ipykernel install --user --name=dl --display-name "Python (dl)"

# 8. Verify
python -c "import torch, numpy; print('PyTorch:', torch.__version__, 'NumPy:', numpy.__version__)"
```

---

**You're all set!** 🎉 Your Mac now has a Python development environment closely matching Google Colab.
