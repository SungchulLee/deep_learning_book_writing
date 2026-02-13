# Linux Quick Reference Card

## One-Line Install Commands

### Ubuntu/Debian

```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y && sudo apt install -y wget curl git build-essential

# 2. Install Miniforge
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && bash ~/miniforge.sh -b -p $HOME/miniforge3 && rm ~/miniforge.sh

# 3. Initialize Conda
$HOME/miniforge3/bin/conda init bash && source ~/.bashrc

# 4. Install VS Code
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg && sudo install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg && sudo sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list' && rm -f packages.microsoft.gpg && sudo apt update && sudo apt install code -y

# 5. Install VS Code Extensions
code --install-extension ms-python.python && code --install-extension ms-toolsai.jupyter && code --install-extension ms-python.vscode-pylance

# 6. Create Environment (from project directory)
conda env create -f 01_Installation_03_env.yml -n dl && conda activate dl

# 7. Register Kernel
python -m ipykernel install --user --name=dl --display-name "Python (dl)"
```

### Fedora/RHEL/CentOS

```bash
# 1. Update system
sudo dnf update -y && sudo dnf install -y wget curl git gcc gcc-c++ make

# 2. Install Miniforge
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && bash ~/miniforge.sh -b -p $HOME/miniforge3 && rm ~/miniforge.sh

# 3. Initialize Conda
$HOME/miniforge3/bin/conda init bash && source ~/.bashrc

# 4. Install VS Code
sudo rpm --import https://packages.microsoft.com/keys/microsoft.asc && sudo sh -c 'echo -e "[code]\nname=Visual Studio Code\nbaseurl=https://packages.microsoft.com/yumrepos/vscode\nenabled=1\ngpgcheck=1\ngpgkey=https://packages.microsoft.com/keys/microsoft.asc" > /etc/yum.repos.d/vscode.repo' && sudo dnf check-update && sudo dnf install code -y

# 5. Install VS Code Extensions
code --install-extension ms-python.python && code --install-extension ms-toolsai.jupyter && code --install-extension ms-python.vscode-pylance

# 6. Create Environment
conda env create -f 01_Installation_03_env.yml -n dl && conda activate dl

# 7. Register Kernel
python -m ipykernel install --user --name=dl --display-name "Python (dl)"
```

### Arch Linux

```bash
# 1. Update system
sudo pacman -Syu --noconfirm && sudo pacman -S --noconfirm wget curl git base-devel

# 2. Install Miniforge
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && bash ~/miniforge.sh -b -p $HOME/miniforge3 && rm ~/miniforge.sh

# 3. Initialize Conda
$HOME/miniforge3/bin/conda init bash && source ~/.bashrc

# 4. Install VS Code (requires yay or paru)
yay -S visual-studio-code-bin

# 5. Install VS Code Extensions
code --install-extension ms-python.python && code --install-extension ms-toolsai.jupyter && code --install-extension ms-python.vscode-pylance

# 6. Create Environment
conda env create -f 01_Installation_03_env.yml -n dl && conda activate dl

# 7. Register Kernel
python -m ipykernel install --user --name=dl --display-name "Python (dl)"
```

## Essential Commands

### Conda
```bash
conda activate dl                    # Activate environment
conda deactivate                     # Deactivate environment
conda env list                       # List all environments
conda list                           # List packages in active environment
conda update --all                   # Update all packages
conda clean --all                    # Clean cached packages
```

### VS Code
```bash
code .                              # Open current directory in VS Code
code file.ipynb                     # Open specific notebook
code --list-extensions              # List installed extensions
code --disable-gpu                  # Disable GPU acceleration (if issues)
```

### Jupyter
```bash
jupyter lab                         # Start Jupyter Lab
jupyter notebook                    # Start Jupyter Notebook
jupyter kernelspec list             # List installed kernels
jupyter notebook --generate-config # Generate config file
```

## Common Issues

| Problem | Solution |
|---------|----------|
| `conda: command not found` | `source ~/.bashrc` or add to PATH: `export PATH="$HOME/miniforge3/bin:$PATH"` |
| Permission denied | Never use `sudo` with conda; fix ownership: `sudo chown -R $USER:$USER ~/miniforge3` |
| Kernel not showing | `python -m ipykernel install --user --name=dl --display-name "Python (dl)"` |
| OpenGL errors | Install: `sudo apt install -y libgl1-mesa-glx libglib2.0-0` |
| Display issues | Set: `export DISPLAY=:0` |

## Alternative VS Code Installation

### Via Snap (Universal)
```bash
sudo snap install --classic code
```

### Via Flatpak (Universal)
```bash
flatpak install flathub com.visualstudio.code -y
flatpak run com.visualstudio.code
```

## GPU Setup (NVIDIA Only)

### Ubuntu/Debian
```bash
# Install NVIDIA driver
ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
sudo reboot

# Verify
nvidia-smi

# Install CUDA-enabled PyTorch
conda activate dl
conda remove pytorch torchvision torchaudio --force
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Verify CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0))"
```

### Fedora/RHEL
```bash
# Enable RPM Fusion
sudo dnf install https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm -y
sudo dnf install https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm -y

# Install NVIDIA driver
sudo dnf install akmod-nvidia -y
sudo reboot

# Verify
nvidia-smi
```

## Useful System Commands

```bash
# Check system info
uname -a                            # Kernel version
lsb_release -a                      # Distribution info
cat /etc/os-release                 # OS details
nproc                               # Number of CPU cores
free -h                             # Memory usage
df -h                               # Disk usage

# Process monitoring
htop                                # Interactive process viewer
nvidia-smi                          # GPU monitoring (NVIDIA)
watch -n 1 nvidia-smi              # Real-time GPU monitoring
```

## Environment Variables

```bash
# Add conda to PATH permanently
echo 'export PATH="$HOME/miniforge3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Set CUDA environment (if using GPU)
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## Paths to Remember

- Miniforge: `~/miniforge3/`
- Conda envs: `~/miniforge3/envs/`
- Jupyter kernels: `~/.local/share/jupyter/kernels/`
- VS Code extensions: `~/.vscode/extensions/`

## Performance Tips

```bash
# Use all CPU cores for conda operations
echo 'export CONDA_THREADS=$(nproc)' >> ~/.bashrc

# Disable conda auto-activation
conda config --set auto_activate_base false

# Enable conda-libmamba-solver for faster dependency resolution
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```
