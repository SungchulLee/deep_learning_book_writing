# Installation Guide for Linux — Python Development Environment

> This guide sets up a conda environment close to Google Colab, with Miniforge/Miniconda and VS Code installed via terminal.

---

## Prerequisites Check

Before starting, identify your Linux distribution:

```bash
cat /etc/os-release
# or
lsb_release -a
```

This guide covers:
- **Ubuntu/Debian** (apt-based)
- **Fedora/RHEL/CentOS** (dnf/yum-based)
- **Arch Linux** (pacman-based)

---

## Step 1: Update System and Install Prerequisites

### Ubuntu/Debian

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y wget curl git build-essential
```

### Fedora/RHEL/CentOS

```bash
sudo dnf update -y
sudo dnf install -y wget curl git gcc gcc-c++ make
```

### Arch Linux

```bash
sudo pacman -Syu --noconfirm
sudo pacman -S --noconfirm wget curl git base-devel
```

---

## Step 2: Install Miniforge (Recommended) or Miniconda

**Miniforge** is recommended for Linux as it uses conda-forge by default and supports multiple architectures.

### Download and Install Miniforge

```bash
# Download Miniforge installer
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh

# Make executable and install
bash ~/miniforge.sh -b -p $HOME/miniforge3

# Clean up
rm ~/miniforge.sh
```

### Initialize Conda

```bash
# Initialize for your shell
$HOME/miniforge3/bin/conda init bash
# or for zsh users:
# $HOME/miniforge3/bin/conda init zsh

# Reload shell configuration
source ~/.bashrc
# or for zsh:
# source ~/.zshrc
```

### Verify Installation

```bash
conda --version
# Should show: conda 24.x.x or newer

which conda
# Should point to ~/miniforge3/bin/conda
```

---

## Step 3: Install Visual Studio Code via Terminal

### Option A: Using Official Microsoft Repository (Ubuntu/Debian)

```bash
# Add Microsoft GPG key
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg
sudo sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
rm -f packages.microsoft.gpg

# Update and install
sudo apt update
sudo apt install code -y
```

### Option B: Using Snap (Universal)

Works on most modern Linux distributions:

```bash
sudo snap install --classic code
```

### Option C: Using Flatpak (Universal)

```bash
flatpak install flathub com.visualstudio.code -y
```

### Option D: Download and Install .deb Package (Ubuntu/Debian)

```bash
wget -O ~/vscode.deb "https://code.visualstudio.com/sha/download?build=stable&os=linux-deb-x64"
sudo dpkg -i ~/vscode.deb
sudo apt-get install -f -y  # Fix dependencies if needed
rm ~/vscode.deb
```

### Option E: Fedora/RHEL/CentOS

```bash
# Add Microsoft repository
sudo rpm --import https://packages.microsoft.com/keys/microsoft.asc
sudo sh -c 'echo -e "[code]\nname=Visual Studio Code\nbaseurl=https://packages.microsoft.com/yumrepos/vscode\nenabled=1\ngpgcheck=1\ngpgkey=https://packages.microsoft.com/keys/microsoft.asc" > /etc/yum.repos.d/vscode.repo'

# Update and install
sudo dnf check-update
sudo dnf install code -y
```

### Option F: Arch Linux

```bash
# VS Code is in AUR, install using yay or paru
yay -S visual-studio-code-bin

# or
paru -S visual-studio-code-bin
```

### Verify Installation

```bash
code --version
```

### Install Essential VS Code Extensions

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

Navigate to your project directory:

```bash
cd /path/to/your/project
```

### Option A: Solver-Built (Recommended)

Create a flexible environment using the YAML file:

```bash
conda env create -f 01_Installation_03_env.yml -n dl
conda activate dl
```

---

### Option B: Create from Scratch (Alternative)

If you need to customize or the YAML has issues:

```bash
# Create base environment
conda create -n dl python=3.10 -y
conda activate dl

# Install core packages
conda install -c conda-forge numpy=2.2.6 pandas scipy scikit-learn matplotlib seaborn jupyterlab ipykernel -y

# Install PyTorch (CPU version)
conda install -c pytorch pytorch=2.6.0 torchvision=0.21.0 torchaudio=2.7.0 cpuonly -y

# Install additional packages
conda install -c conda-forge opencv networkx pillow statsmodels xgboost lightgbm tqdm -y
conda install -c conda-forge shapely geopandas pyproj gdal -y
```

---

## Step 5: Register Jupyter Kernel with VS Code

After creating the environment:

```bash
conda activate dl
python -m ipykernel install --user --name=dl --display-name "Python (dl)"
```

Verify kernel registration:

```bash
jupyter kernelspec list
# Should show 'dl' in the list
```

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
- CUDA available: False (unless you have NVIDIA GPU and install CUDA toolkit)

---

## Step 7: Using Jupyter in VS Code

1. Open VS Code in your project directory:
   ```bash
   code .
   ```

2. Open or create a `.ipynb` file

3. Click **"Select Kernel"** in the top-right

4. Choose **"Python (dl)"** from the list

5. Start coding!

---

## Step 8: Optional — Install GPU Support (NVIDIA GPU Only)

If you have an NVIDIA GPU and want CUDA support:

### Install NVIDIA Drivers

**Ubuntu/Debian:**
```bash
# Check available drivers
ubuntu-drivers devices

# Install recommended driver
sudo ubuntu-drivers autoinstall

# Or install specific version
sudo apt install nvidia-driver-535 -y

# Reboot
sudo reboot
```

**Fedora/RHEL:**
```bash
# Enable RPM Fusion repository
sudo dnf install https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm -y
sudo dnf install https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm -y

# Install driver
sudo dnf install akmod-nvidia -y

# Reboot
sudo reboot
```

### Verify NVIDIA Driver

```bash
nvidia-smi
# Should display GPU information
```

### Install CUDA Toolkit

**Ubuntu/Debian:**
```bash
# Install CUDA from conda (recommended)
conda activate dl
conda install -c conda-forge cudatoolkit=11.8 -y
```

### Install PyTorch with CUDA

```bash
conda activate dl

# Uninstall CPU version
conda remove pytorch torchvision torchaudio --force

# Install CUDA version (CUDA 11.8 example)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

### Verify CUDA Support

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
```

---

## Step 9: Optional — Install Additional Packages

After activating the environment:

```bash
conda activate dl
pip install tensorboard transformers wandb accelerate
```

Or via conda:

```bash
conda install -c conda-forge package-name
```

---

## Quick Troubleshooting

### Issue: `conda: command not found`

**Solution:**
```bash
# Reinitialize conda
$HOME/miniforge3/bin/conda init bash
source ~/.bashrc

# Or add to PATH manually
export PATH="$HOME/miniforge3/bin:$PATH"
echo 'export PATH="$HOME/miniforge3/bin:$PATH"' >> ~/.bashrc
```

### Issue: VS Code doesn't detect conda environments

**Solution:**
1. Install Python extension: `code --install-extension ms-python.python`
2. Reload VS Code: `Ctrl+Shift+P` → "Developer: Reload Window"
3. Select interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. Ensure conda base is activated when starting VS Code

### Issue: Permission denied when installing packages

**Solution:**
```bash
# Never use sudo with conda
# If you get permission errors, check conda ownership:
ls -la ~/miniforge3

# Fix ownership if needed:
sudo chown -R $USER:$USER ~/miniforge3
```

### Issue: Jupyter kernel not showing in VS Code

**Solution:**
```bash
conda activate dl
python -m ipykernel install --user --name=dl --display-name "Python (dl)"

# List installed kernels
jupyter kernelspec list

# If still not showing, reinstall Jupyter extension in VS Code
code --uninstall-extension ms-toolsai.jupyter
code --install-extension ms-toolsai.jupyter
```

### Issue: OpenGL/display issues with graphical packages

**Solution:**
```bash
# Install display dependencies (Ubuntu/Debian)
sudo apt install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev

# Fedora/RHEL
sudo dnf install -y mesa-libGL glib2 libSM libXext libXrender
```

---

## File Reference

* **`01_Installation_03_env.yml`** — Flexible environment configuration (recommended for Linux)
* **`01_Installation_02_colab_raw.txt`** — Colab package snapshot for reference
* **`01_Installation_06_requirements.txt`** — Pure pip requirements (alternative method)

---

## Summary: Complete Installation Flow

```bash
# 1. Update system and install prerequisites
sudo apt update && sudo apt upgrade -y
sudo apt install -y wget curl git build-essential

# 2. Install Miniforge
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh
bash ~/miniforge.sh -b -p $HOME/miniforge3
rm ~/miniforge.sh

# 3. Initialize conda
$HOME/miniforge3/bin/conda init bash
source ~/.bashrc

# 4. Install VS Code (Ubuntu/Debian via official repo)
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg
sudo sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
rm -f packages.microsoft.gpg
sudo apt update
sudo apt install code -y

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

## Additional Tips for Linux Users

### Using tmux/screen for Long-Running Processes

```bash
# Install tmux
sudo apt install tmux -y  # Ubuntu/Debian
sudo dnf install tmux -y  # Fedora

# Start a session
tmux new -s dl-session

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t dl-session
```

### Setting up SSH for Remote Development

```bash
# Enable VS Code Remote-SSH
code --install-extension ms-vscode-remote.remote-ssh

# Now you can connect to remote Linux machines from VS Code
```

### Performance Monitoring

```bash
# Install htop for better process monitoring
sudo apt install htop -y

# For GPU monitoring (if NVIDIA GPU)
watch -n 1 nvidia-smi
```

---

**You're all set!** 🎉 Your Linux machine now has a Python development environment closely matching Google Colab.
