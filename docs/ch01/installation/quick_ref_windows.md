# Windows Quick Reference Card

## One-Line Install Commands (PowerShell as Administrator)

```powershell
# 1. Install Chocolatey
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Close and reopen PowerShell as Administrator

# 2. Install Miniconda
choco install miniconda3 -y

# Close and reopen PowerShell as Administrator

# 3. Initialize Conda
conda init powershell

# Close and reopen PowerShell

# 4. Install VS Code
choco install vscode -y

# Close and reopen PowerShell

# 5. Install VS Code Extensions
code --install-extension ms-python.python; code --install-extension ms-toolsai.jupyter; code --install-extension ms-python.vscode-pylance

# 6. Create Environment (from project directory)
conda env create -f 01_Installation_03_env.yml -n dl; conda activate dl

# 7. Register Kernel
python -m ipykernel install --user --name=dl --display-name "Python (dl)"
```

## Essential Commands

### Conda
```powershell
conda activate dl                    # Activate environment
conda deactivate                     # Deactivate environment
conda env list                       # List all environments
conda list                           # List packages in active environment
conda update --all                   # Update all packages
```

### VS Code
```powershell
code .                              # Open current directory in VS Code
code file.ipynb                     # Open specific notebook
code --list-extensions              # List installed extensions
```

### Jupyter
```powershell
jupyter lab                         # Start Jupyter Lab
jupyter notebook                    # Start Jupyter Notebook
jupyter kernelspec list             # List installed kernels
```

## Common Issues

| Problem | Solution |
|---------|----------|
| `conda: command not found` | Close and reopen PowerShell after `conda init` |
| Execution policy error | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| Kernel not showing | `python -m ipykernel install --user --name=dl` |
| Choco not recognized | Close and reopen PowerShell as Admin |

## Alternative: Using winget (Built-in)

```powershell
# Install VS Code
winget install --id Microsoft.VisualStudioCode -e

# Install Git
winget install --id Git.Git -e
```

## GPU Setup (NVIDIA Only)

```powershell
# Install CUDA toolkit
choco install cuda -y

# Install PyTorch with CUDA
conda activate dl
conda remove pytorch torchvision torchaudio
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Verify
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## Paths to Remember

- Miniconda: `C:\Users\<YourName>\Miniconda3\`
- VS Code: `C:\Users\<YourName>\AppData\Local\Programs\Microsoft VS Code\`
- Conda envs: `C:\Users\<YourName>\Miniconda3\envs\`
