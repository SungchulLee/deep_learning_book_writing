# macOS Quick Reference Card

## One-Line Install Commands

```bash
# 1. Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Setup Homebrew (Apple Silicon)
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile && eval "$(/opt/homebrew/bin/brew shellenv)"

# 3. Install Miniforge
brew install miniforge && conda init "$(basename "${SHELL}")" && source ~/.zshrc

# 4. Install VS Code
brew install --cask visual-studio-code

# 5. Install VS Code Extensions
code --install-extension ms-python.python && code --install-extension ms-toolsai.jupyter && code --install-extension ms-python.vscode-pylance

# 6. Create Environment (from project directory)
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
```

### VS Code
```bash
code .                              # Open current directory in VS Code
code file.ipynb                     # Open specific notebook
code --list-extensions              # List installed extensions
```

### Jupyter
```bash
jupyter lab                         # Start Jupyter Lab
jupyter notebook                    # Start Jupyter Notebook
jupyter kernelspec list             # List installed kernels
```

## Common Issues

| Problem | Solution |
|---------|----------|
| `conda: command not found` | `source ~/.zshrc` |
| `code: command not found` | Open VS Code → `Cmd+Shift+P` → "Install 'code' command in PATH" |
| Kernel not showing | `python -m ipykernel install --user --name=dl` |
| Homebrew warnings | `brew doctor` |

## Architecture-Specific Notes

**Intel Mac:** Use `01_Installation_04_env-explicit.txt` for exact reproduction
**Apple Silicon:** Use `01_Installation_03_env.yml` (will resolve native arm64 packages)

Check your architecture:
```bash
uname -m
# x86_64 = Intel
# arm64 = Apple Silicon
```
