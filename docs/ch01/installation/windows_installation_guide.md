# Installation Guide for Windows — Python Development Environment

> This guide sets up a conda environment close to Google Colab, with Miniconda and VS Code installed via terminal/PowerShell.

---

## Prerequisites Check

Before starting, ensure you have:
- Windows 10 or later (64-bit)
- Administrator access
- Internet connection

---

## Step 1: Install Chocolatey (Optional but Recommended)

Chocolatey is a package manager for Windows that makes software installation easier.

### Installation

Open **PowerShell as Administrator** (Right-click → "Run as Administrator"):

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

### Verify Installation

```powershell
choco --version
# Should show: 2.x.x or newer
```

Close and reopen PowerShell as Administrator for changes to take effect.

---

## Step 2: Install Miniconda

### Option A: Install via Chocolatey (Recommended)

In PowerShell (as Administrator):

```powershell
choco install miniconda3 -y
```

### Option B: Manual Download and Install

If you prefer not to use Chocolatey:

1. Download Miniconda installer:
   ```powershell
   Invoke-WebRequest -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile "$env:TEMP\Miniconda3-latest-Windows-x86_64.exe"
   ```

2. Run the installer:
   ```powershell
   Start-Process -FilePath "$env:TEMP\Miniconda3-latest-Windows-x86_64.exe" -ArgumentList "/S /D=$env:USERPROFILE\Miniconda3" -Wait
   ```

### Initialize Conda

Close all PowerShell windows, then open a **new** PowerShell (as Administrator) and run:

```powershell
conda init powershell
conda init cmd.exe
```

**Close and reopen** PowerShell for changes to take effect.

### Verify Installation

```powershell
conda --version
# Should show: conda 24.x.x or newer

where.exe conda
# Should show path to conda installation
```

---

## Step 3: Install Visual Studio Code via Terminal

### Option A: Install via Chocolatey (Recommended)

In PowerShell (as Administrator):

```powershell
choco install vscode -y
```

### Option B: Install via winget (Built-in Windows Package Manager)

Windows 10/11 comes with `winget`:

```powershell
winget install --id Microsoft.VisualStudioCode -e
```

### Option C: Manual Download via PowerShell

```powershell
Invoke-WebRequest -Uri "https://code.visualstudio.com/sha/download?build=stable&os=win32-x64-user" -OutFile "$env:TEMP\VSCodeUserSetup.exe"
Start-Process -FilePath "$env:TEMP\VSCodeUserSetup.exe" -ArgumentList "/VERYSILENT /MERGETASKS=!runcode" -Wait
```

### Verify Installation

Close and reopen PowerShell, then:

```powershell
code --version
```

### Install Essential VS Code Extensions (via Terminal)

```powershell
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

Open PowerShell (or Anaconda Prompt) and navigate to your project directory:

```powershell
cd C:\path\to\your\project
```

### Option A: Solver-Built (Recommended for Windows)

Create a flexible environment that conda will resolve for Windows:

```powershell
conda env create -f 01_Installation_03_env.yml -n dl
conda activate dl
```

**Note:** The explicit spec file (`01_Installation_04_env-explicit.txt`) is for macOS Intel. Windows users should use the YAML method.

---

### Option B: Create from Scratch (If YAML doesn't work)

If you encounter platform-specific issues:

```powershell
# Create base environment
conda create -n dl python=3.10 -y
conda activate dl

# Install core packages
conda install -c conda-forge numpy=2.2.6 pandas scipy scikit-learn matplotlib seaborn jupyterlab ipykernel -y

# Install PyTorch (CPU version for Windows)
conda install -c pytorch pytorch=2.6.0 torchvision=0.21.0 torchaudio=2.7.0 cpuonly -y

# Install additional packages
conda install -c conda-forge opencv networkx pillow statsmodels xgboost lightgbm -y
```

---

## Step 5: Register Jupyter Kernel with VS Code

After creating the environment:

```powershell
conda activate dl
python -m ipykernel install --user --name=dl --display-name "Python (dl)"
```

---

## Step 6: Verify Installation

Test your environment:

```powershell
conda activate dl
python -c "import platform, torch, numpy; print('Python:', platform.python_version()); print('PyTorch:', torch.__version__); print('NumPy:', numpy.__version__); print('CUDA available:', torch.cuda.is_available())"
```

**Expected output:**
- Python: 3.10.x
- PyTorch: 2.6.0
- NumPy: 2.2.6
- CUDA available: False (unless you have NVIDIA GPU and install CUDA toolkit)

---

## Step 7: Using Jupyter in VS Code

1. Open VS Code in your project directory:
   ```powershell
   code .
   ```

2. Open or create a `.ipynb` file

3. Click **"Select Kernel"** in the top-right

4. Choose **"Python (dl)"** from the list

5. Start coding!

---

## Step 8: Optional — Install GPU Support (NVIDIA GPU Only)

If you have an NVIDIA GPU and want CUDA support:

### Install NVIDIA CUDA Toolkit

```powershell
# Via Chocolatey
choco install cuda -y
```

### Install PyTorch with CUDA

```powershell
conda activate dl
# Uninstall CPU version
conda remove pytorch torchvision torchaudio

# Install CUDA version (example: CUDA 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

### Verify CUDA

```python
import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))  # Shows your GPU name
```

---

## Step 9: Optional — Install Additional Packages

After activating the environment:

```powershell
conda activate dl
pip install tensorboard transformers wandb
```

Or via conda:

```powershell
conda install -c conda-forge package-name
```

---

## Quick Troubleshooting

### Issue: `conda: command not found` or not recognized

**Solution:**
1. Ensure conda is initialized:
   ```powershell
   conda init powershell
   ```
2. Close and reopen PowerShell
3. Check if conda is in PATH:
   ```powershell
   $env:PATH -split ';' | Select-String conda
   ```

### Issue: PowerShell execution policy blocks scripts

**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: VS Code doesn't show Python environments

**Solution:**
1. Install Python extension: `code --install-extension ms-python.python`
2. Reload VS Code: `Ctrl+Shift+P` → "Developer: Reload Window"
3. Select interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter"

### Issue: Conda environment not showing in VS Code

**Solution:**
1. Make sure kernel is registered:
   ```powershell
   conda activate dl
   python -m ipykernel install --user --name=dl --display-name "Python (dl)"
   ```
2. Restart VS Code

### Issue: Package conflicts during installation

**Solution:**
- Update conda: `conda update -n base conda`
- Try installing in smaller batches
- Use `conda install package --no-deps` to skip dependency resolution

---

## File Reference

* **`01_Installation_03_env.yml`** — Flexible environment configuration (use this for Windows)
* **`01_Installation_02_colab_raw.txt`** — Colab package snapshot for reference
* **`01_Installation_06_requirements.txt`** — Pure pip requirements (alternative method)

---

## Summary: Complete Installation Flow

```powershell
# 1. Install Chocolatey (in PowerShell as Administrator)
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Close and reopen PowerShell as Administrator

# 2. Install Miniconda
choco install miniconda3 -y

# Close and reopen PowerShell as Administrator

# 3. Initialize conda
conda init powershell
conda init cmd.exe

# Close and reopen PowerShell

# 4. Install VS Code
choco install vscode -y

# Close and reopen PowerShell

# 5. Install VS Code extensions
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
code --install-extension ms-python.vscode-pylance

# 6. Create conda environment
cd C:\path\to\project
conda env create -f 01_Installation_03_env.yml -n dl
conda activate dl

# 7. Register Jupyter kernel
python -m ipykernel install --user --name=dl --display-name "Python (dl)"

# 8. Verify
python -c "import torch, numpy; print('PyTorch:', torch.__version__, 'NumPy:', numpy.__version__)"
```

---

## Alternative: Using Anaconda Prompt

If you prefer using Anaconda Prompt instead of PowerShell:

1. Search for "Anaconda Prompt" in Start Menu
2. All conda commands work the same
3. No need to initialize conda in PowerShell

---

**You're all set!** 🎉 Your Windows machine now has a Python development environment closely matching Google Colab.
