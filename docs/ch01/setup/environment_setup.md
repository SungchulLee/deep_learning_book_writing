# Environment Setup

## Learning Objectives

By the end of this section, you will be able to:

- Install and configure system package managers for your operating system
- Set up Miniforge or Miniconda for Python environment management
- Install and configure Visual Studio Code with essential extensions
- Create an isolated Python environment for deep learning projects
- Verify your installation and troubleshoot common issues

---

## macOS Installation

### Step 1: Install Xcode Command Line Tools

Apple's command line tools provide essential compilers and development utilities:

```bash
xcode-select --install
```

A dialog will appear. Click "Install" and wait for completion (several minutes).

### Step 2: Install Homebrew

Homebrew is the standard package manager for macOS, making software installation trivial:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**Post-installation configuration** (follow the instructions shown in terminal):

For Apple Silicon (M1/M2/M3/M4):
```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

For Intel Macs:
```bash
echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/usr/local/bin/brew shellenv)"
```

Verify installation:
```bash
brew --version
# Expected: Homebrew 4.x.x or newer
```

### Step 3: Install Miniforge

Miniforge is the recommended conda distribution for macOS, especially for Apple Silicon:

```bash
brew install miniforge
```

Initialize conda for your shell:
```bash
conda init "$(basename "${SHELL}")"
```

**Close and reopen Terminal**, then verify:
```bash
conda --version
# Expected: conda 24.x.x or newer

which conda
# Expected: /opt/homebrew/Caskroom/miniforge/base/condabin/conda (Apple Silicon)
# or: /usr/local/Caskroom/miniforge/base/condabin/conda (Intel)
```

### Step 4: Install Visual Studio Code

```bash
brew install --cask visual-studio-code
```

Verify the `code` command works:
```bash
code --version
```

If `code` is not found, open VS Code manually, press `Cmd+Shift+P`, type "shell command", and select "Install 'code' command in PATH".

### Step 5: Install VS Code Extensions

```bash
# Python language support
code --install-extension ms-python.python

# Jupyter notebook support
code --install-extension ms-toolsai.jupyter

# Python language server (IntelliSense)
code --install-extension ms-python.vscode-pylance

# Python debugger
code --install-extension ms-python.debugpy
```

---

## Windows Installation

### Step 1: Install Chocolatey (Recommended)

Open **PowerShell as Administrator** (right-click → "Run as Administrator"):

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

**Close and reopen PowerShell as Administrator**, then verify:
```powershell
choco --version
# Expected: 2.x.x or newer
```

### Step 2: Install Miniconda

```powershell
choco install miniconda3 -y
```

**Alternative using winget** (built into Windows 10/11):
```powershell
winget install --id Anaconda.Miniconda3 -e
```

**Close and reopen PowerShell**, then initialize conda:
```powershell
conda init powershell
conda init cmd.exe
```

**Close and reopen PowerShell**, then verify:
```powershell
conda --version
# Expected: conda 24.x.x or newer
```

### Step 3: Install Visual Studio Code

```powershell
choco install vscode -y
```

**Alternative using winget:**
```powershell
winget install --id Microsoft.VisualStudioCode -e
```

**Close and reopen PowerShell**, then verify:
```powershell
code --version
```

### Step 4: Install VS Code Extensions

```powershell
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
code --install-extension ms-python.vscode-pylance
code --install-extension ms-python.debugpy
```

---

## Linux Installation

### Step 1: Update System and Install Prerequisites

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y wget curl git build-essential
```

**Fedora/RHEL/CentOS:**
```bash
sudo dnf update -y
sudo dnf install -y wget curl git gcc gcc-c++ make
```

**Arch Linux:**
```bash
sudo pacman -Syu --noconfirm
sudo pacman -S --noconfirm wget curl git base-devel
```

### Step 2: Install Miniforge

```bash
# Download installer
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh

# Install (batch mode, default location)
bash ~/miniforge.sh -b -p $HOME/miniforge3

# Clean up
rm ~/miniforge.sh
```

Initialize conda:
```bash
$HOME/miniforge3/bin/conda init bash
# For zsh users: $HOME/miniforge3/bin/conda init zsh
```

**Close and reopen Terminal**, then verify:
```bash
conda --version
which conda
```

### Step 3: Install Visual Studio Code

**Ubuntu/Debian (Official Repository):**
```bash
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg
sudo sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
rm -f packages.microsoft.gpg
sudo apt update
sudo apt install code -y
```

**Fedora/RHEL:**
```bash
sudo rpm --import https://packages.microsoft.com/keys/microsoft.asc
sudo sh -c 'echo -e "[code]\nname=Visual Studio Code\nbaseurl=https://packages.microsoft.com/yumrepos/vscode\nenabled=1\ngpgcheck=1\ngpgkey=https://packages.microsoft.com/keys/microsoft.asc" > /etc/yum.repos.d/vscode.repo'
sudo dnf check-update
sudo dnf install code -y
```

**Universal (Snap):**
```bash
sudo snap install --classic code
```

### Step 4: Install VS Code Extensions

```bash
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
code --install-extension ms-python.vscode-pylance
code --install-extension ms-python.debugpy
```

---

## Creating the Deep Learning Environment

With the base tools installed, create an isolated Python environment for deep learning work.

### Using the Environment Specification File

Create a file named `env.yml` with the following contents:

```yaml
name: dl
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.10
  - ipykernel
  - jupyterlab
  - numpy
  - pandas
  - scipy
  - scikit-learn
  - matplotlib
  - seaborn
  - pytorch
  - torchvision
  - torchaudio
  - cpuonly  # Remove this line for GPU support
  - tqdm
  - pillow
  - networkx
  - statsmodels
  - xgboost
  - lightgbm
```

Create the environment:
```bash
conda env create -f env.yml -n dl
```

Activate the environment:
```bash
conda activate dl
```

### Manual Environment Creation

Alternatively, create the environment step-by-step:

```bash
# Create base environment
conda create -n dl python=3.10 -y

# Activate
conda activate dl

# Install core scientific packages
conda install -c conda-forge numpy pandas scipy scikit-learn matplotlib seaborn jupyterlab ipykernel -y

# Install PyTorch (CPU version)
conda install -c pytorch pytorch torchvision torchaudio cpuonly -y

# Install additional ML packages
conda install -c conda-forge xgboost lightgbm statsmodels networkx tqdm pillow -y
```

### Register Jupyter Kernel

Register your environment as a Jupyter kernel for use in VS Code and JupyterLab:

```bash
conda activate dl
python -m ipykernel install --user --name=dl --display-name "Python (dl)"
```

Verify kernel registration:
```bash
jupyter kernelspec list
```

Expected output includes:
```
Available kernels:
  dl         /Users/username/.local/share/jupyter/kernels/dl
  python3    /Users/username/miniforge3/share/jupyter/kernels/python3
```

---

## Verification

### Comprehensive Environment Test

Run this verification script to ensure all components are working:

```python
import sys
import platform

print("=" * 60)
print("System Information")
print("=" * 60)
print(f"Python Version: {platform.python_version()}")
print(f"Python Path: {sys.executable}")
print(f"Platform: {platform.platform()}")
print(f"Architecture: {platform.machine()}")
print()

print("=" * 60)
print("Package Versions")
print("=" * 60)

packages = [
    ('numpy', 'np'),
    ('pandas', 'pd'),
    ('scipy', 'scipy'),
    ('sklearn', 'sklearn'),
    ('matplotlib', 'matplotlib'),
    ('seaborn', 'sns'),
    ('torch', 'torch'),
    ('torchvision', 'torchvision'),
    ('torchaudio', 'torchaudio'),
]

for name, alias in packages:
    try:
        module = __import__(name)
        print(f"{name:15} {module.__version__}")
    except ImportError:
        print(f"{name:15} NOT INSTALLED")
    except AttributeError:
        print(f"{name:15} (version unavailable)")

print()
print("=" * 60)
print("PyTorch Configuration")
print("=" * 60)
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Count: {torch.cuda.device_count()}")

# Check for Apple Silicon MPS
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print(f"MPS Available: True (Apple Silicon GPU)")
else:
    print(f"MPS Available: False")

print()
print("=" * 60)
print("Quick Functionality Test")
print("=" * 60)
import numpy as np
import torch

# NumPy test
arr = np.random.randn(1000, 1000)
result = np.dot(arr, arr.T)
print(f"NumPy matrix multiplication: OK (result shape: {result.shape})")

# PyTorch test
tensor = torch.randn(1000, 1000)
result = torch.mm(tensor, tensor.T)
print(f"PyTorch matrix multiplication: OK (result shape: {tuple(result.shape)})")

print()
print("✅ Environment verification complete!")
```

Save this as `verify_env.py` and run:
```bash
conda activate dl
python verify_env.py
```

### Expected Output

A successful installation should show:

- Python version 3.10.x
- All listed packages with version numbers
- CUDA Available: False (for CPU-only installation)
- MPS Available: True (on Apple Silicon Macs)
- Matrix multiplication tests passing

---

## Using VS Code with Your Environment

### Opening a Project

```bash
# Navigate to your project directory
cd ~/projects/deep-learning

# Open VS Code
code .
```

### Selecting the Python Interpreter

1. Open any `.py` file or create a new `.ipynb` notebook.
2. Look at the bottom-right corner of VS Code for the Python version indicator.
3. Click it and select "Python (dl)" from the list.
4. For notebooks, click "Select Kernel" in the top-right and choose "Python (dl)".

### VS Code Settings for Deep Learning

Create `.vscode/settings.json` in your project directory:

```json
{
    "python.defaultInterpreterPath": "~/miniforge3/envs/dl/bin/python",
    "python.terminal.activateEnvironment": true,
    "jupyter.askForKernelRestart": false,
    "editor.formatOnSave": true,
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true
}
```

---

## Troubleshooting

### "conda: command not found"

**Cause**: Shell not initialized after conda installation.

**Solution**:
```bash
# macOS/Linux
source ~/.bashrc  # or ~/.zshrc for zsh

# Or reinitialize
conda init "$(basename "${SHELL}")"
```

### VS Code doesn't show Python environments

**Solution**:

1. Install Python extension: `code --install-extension ms-python.python`
2. Reload VS Code: `Cmd/Ctrl+Shift+P` → "Developer: Reload Window"
3. Select interpreter: `Cmd/Ctrl+Shift+P` → "Python: Select Interpreter"

### Kernel not showing in Jupyter

**Solution**:
```bash
conda activate dl
python -m ipykernel install --user --name=dl --display-name "Python (dl)"
```

Verify:
```bash
jupyter kernelspec list
```

### Permission errors on Linux

**Solution**: Never use `sudo` with conda. Fix ownership:
```bash
sudo chown -R $USER:$USER ~/miniforge3
```

### Homebrew installation fails on macOS

**Solution**: Install Xcode Command Line Tools first:
```bash
xcode-select --install
```

### PowerShell execution policy error on Windows

**Solution**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Quick Reference Commands

### Environment Management

```bash
# List all environments
conda env list

# Activate environment
conda activate dl

# Deactivate environment
conda deactivate

# Remove environment
conda env remove -n dl

# Export environment to file
conda env export > environment.yml

# Update all packages
conda update --all -y

# Clean cached packages
conda clean --all -y
```

### VS Code Commands

```bash
# Open current directory
code .

# Open specific file
code file.ipynb

# List installed extensions
code --list-extensions

# Install extension
code --install-extension extension-id
```

### Jupyter Commands

```bash
# Start JupyterLab
jupyter lab

# List installed kernels
jupyter kernelspec list

# Remove a kernel
jupyter kernelspec uninstall kernel-name
```

---

## Summary

In this section, you learned to:

1. **Install platform-specific tools** — Homebrew for macOS, Chocolatey for Windows, and native package managers for Linux provide system-level software installation.

2. **Set up Miniforge/Miniconda** — Conda manages Python environments and dependencies, including non-Python libraries like CUDA.

3. **Configure VS Code** — The Python and Jupyter extensions transform VS Code into a powerful deep learning IDE.

4. **Create and verify environments** — Isolated environments ensure reproducibility and avoid dependency conflicts.

---

## Next Steps

With your environment configured, proceed to:

- [Virtual Environments](virtual_environments.md) — Learn environment isolation in depth
- [Package Management](package_management.md) — Master conda and pip for managing dependencies
- [GPU Configuration](../../ch02/intro/gpu_configuration.md) — Set up CUDA for GPU-accelerated training

---

## References

1. Conda Documentation. https://docs.conda.io
2. VS Code Python Tutorial. https://code.visualstudio.com/docs/python/python-tutorial
3. PyTorch Installation Guide. https://pytorch.org/get-started/locally/
4. Homebrew Documentation. https://docs.brew.sh
5. Chocolatey Documentation. https://docs.chocolatey.org
