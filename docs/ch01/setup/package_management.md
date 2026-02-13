# Package Management

## Learning Objectives

By the end of this section, you will be able to:

- Understand the distinction between conda and pip package managers
- Master conda channels and environment management
- Handle dependency conflicts and version constraints
- Create reproducible environment specifications
- Implement best practices for deep learning project dependencies
- Troubleshoot common package management issues

---

## Introduction

Package management is a critical skill for deep learning practitioners. Unlike traditional software development, deep learning projects involve complex dependency graphs spanning numerical computing libraries, GPU drivers, CUDA toolkits, and machine learning frameworks. A single version mismatch can lead to cryptic errors or silent numerical bugs. This section provides a comprehensive guide to managing these dependencies professionally.

### The Deep Learning Dependency Challenge

Consider a typical deep learning project's dependency structure:

```
Your Project
    ├── PyTorch 2.6.0
    │   ├── Requires NumPy >= 1.19
    │   ├── Requires CUDA 11.8 or 12.1 (GPU)
    │   └── Requires Python 3.8-3.12
    ├── TensorFlow 2.15
    │   ├── Requires NumPy < 2.0
    │   ├── Requires CUDA 11.8 (GPU)
    │   └── Requires Python 3.9-3.11
    ├── Transformers 4.35
    │   ├── Requires PyTorch OR TensorFlow
    │   └── Requires tokenizers >= 0.14
    └── OpenCV 4.10
        ├── Requires NumPy
        └── Optional GPU support
```

Notice the potential conflicts: TensorFlow requires NumPy < 2.0, while modern PyTorch works with NumPy 2.x. Without proper management, installing both frameworks in the same environment can lead to runtime errors.

---

## Conda vs Pip: Understanding the Difference

### Fundamental Architecture

**Conda** and **pip** serve similar purposes but operate fundamentally differently:

| Aspect | Conda | Pip |
|--------|-------|-----|
| **Package Source** | Conda channels (binary) | PyPI (wheels/source) |
| **Language Support** | Any (Python, R, C, etc.) | Python only |
| **Environment Scope** | Full environment isolation | Python packages only |
| **Dependency Solver** | SAT solver (libmamba) | Simple resolution (pip-tools) |
| **System Libraries** | Managed | Requires system install |
| **Virtual Environment** | Built-in | Requires venv/virtualenv |

### When to Use Each

**Use Conda for:**

- Creating environments
- Installing packages with complex binary dependencies (OpenCV, GDAL, CUDA)
- Managing non-Python dependencies
- Ensuring reproducibility across platforms

**Use Pip for:**

- PyPI-only packages
- Packages not available on conda-forge
- Development installs (`pip install -e .`)
- Quick prototyping

### The Golden Rule

```bash
# CORRECT: Conda first, pip second
conda install numpy pandas scipy
pip install some-pypi-only-package

# WRONG: Mixing randomly
pip install numpy
conda install pandas  # May override pip numpy
pip install scipy     # Confusion!
```

---

## Conda Deep Dive

### Channels: Package Sources

Channels are repositories hosting conda packages. Understanding channels is crucial for finding packages and resolving conflicts.

```bash
# List configured channels
conda config --show channels

# Add a channel
conda config --add channels conda-forge

# Set channel priority
conda config --set channel_priority strict
```

**Major Channels:**

| Channel | Description | Use Case |
|---------|-------------|----------|
| `defaults` | Anaconda's curated packages | Commercial, stable |
| `conda-forge` | Community-maintained | Most comprehensive |
| `pytorch` | Official PyTorch builds | PyTorch installation |
| `nvidia` | NVIDIA GPU packages | CUDA, cuDNN |

**Recommended Configuration for Deep Learning:**

```bash
# Set conda-forge as primary (best for scientific computing)
conda config --add channels conda-forge
conda config --set channel_priority strict
```

### Package Operations

```bash
# Search for packages
conda search pytorch
conda search 'numpy>=1.20'

# Install packages
conda install numpy pandas scipy

# Install specific version
conda install numpy=1.24.0
conda install 'numpy>=1.20,<2.0'

# Install from specific channel
conda install -c pytorch pytorch

# Update packages
conda update numpy
conda update --all

# Remove packages
conda remove numpy

# List installed packages
conda list
conda list numpy  # Filter by name
```

### Version Constraints Syntax

```yaml
# Exact version
numpy=1.24.0

# Minimum version
numpy>=1.20

# Version range
numpy>=1.20,<2.0

# Compatible release (e.g., 1.20.x)
numpy~=1.20.0

# Build string specification
numpy=1.24.0=py310h5f9d8c6_0
```

---

## Environment Specification Files

### YAML Format (Recommended)

The `environment.yml` file provides cross-platform environment specifications:

```yaml
name: dl
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  # Core Python
  - python=3.10
  
  # Scientific computing
  - numpy>=2.0
  - pandas>=2.0
  - scipy>=1.10
  
  # Machine learning
  - scikit-learn>=1.3
  - xgboost>=2.0
  - lightgbm>=4.0
  
  # Deep learning (CPU)
  - pytorch>=2.0
  - torchvision>=0.15
  - torchaudio>=2.0
  - cpuonly  # Remove for GPU
  
  # Visualization
  - matplotlib>=3.7
  - seaborn>=0.12
  
  # Development tools
  - jupyterlab>=4.0
  - ipykernel>=6.0
  
  # Pip-only packages
  - pip
  - pip:
    - wandb>=0.15
    - transformers>=4.30
    - accelerate>=0.20
```

**Creating the environment:**
```bash
conda env create -f environment.yml

# Update existing environment
conda env update -f environment.yml --prune
```

### Explicit Specification (Exact Reproducibility)

For exact byte-for-byte reproducibility on the same platform:

```bash
# Export
conda list --explicit > spec-file.txt

# Create
conda create -n dl --file spec-file.txt
```

The explicit file contains URLs with hashes:

```
# This file may be used to create an environment using:
# $ conda create --name <env> --file <this file>
# platform: osx-64
@EXPLICIT
https://conda.anaconda.org/conda-forge/osx-64/python-3.10.13-h5f1b46c_0_cpython.conda#sha256=abc123...
https://conda.anaconda.org/conda-forge/osx-64/numpy-1.24.0-py310h7451ae0_0.conda#sha256=def456...
```

**Limitation**: Explicit specs are platform-specific (osx-64, linux-64, win-64).

### Requirements.txt (Pip Fallback)

When conda packages aren't available or for pip-based workflows:

```
# requirements.txt
numpy>=1.24.0,<2.0
pandas>=2.0.0
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
wandb>=0.15.0

# Pinned versions for reproducibility
# numpy==1.24.3
# pandas==2.0.2
```

```bash
# Install
pip install -r requirements.txt

# Generate from current environment
pip freeze > requirements.txt
```

---

## Dependency Resolution

### Understanding Conflicts

Dependency conflicts occur when packages require incompatible versions:

```
Package A requires numpy>=2.0
Package B requires numpy<2.0
```

Conda's SAT solver attempts to find a compatible combination, but sometimes no solution exists.

### Diagnosing Conflicts

```bash
# Dry run to see what would happen
conda install package --dry-run

# Verbose output
conda install package -v

# Check package dependencies
conda search package --info
```

### Resolution Strategies

**Strategy 1: Relax Version Constraints**

```bash
# Instead of exact version
conda install numpy=1.24.0

# Try range
conda install 'numpy>=1.20,<1.25'
```

**Strategy 2: Use conda-forge Consistently**

```bash
# Reset channels
conda config --remove channels defaults
conda config --add channels conda-forge
conda config --set channel_priority strict
```

**Strategy 3: Separate Environments**

When packages truly conflict, create separate environments:

```bash
# PyTorch environment
conda create -n pytorch python=3.10 pytorch torchvision -c pytorch

# TensorFlow environment
conda create -n tensorflow python=3.10 tensorflow
```

**Strategy 4: Use libmamba Solver**

The newer libmamba solver is faster and sometimes finds solutions the classic solver misses:

```bash
# Install solver
conda install -n base conda-libmamba-solver

# Use for all operations
conda config --set solver libmamba

# Or per-command
conda install package --solver=libmamba
```

---

## Pip Integration

### Installing Pip Packages in Conda Environments

When packages are only available on PyPI:

```bash
# Activate conda environment first
conda activate dl

# Install pip package
pip install transformers

# With constraints
pip install 'transformers>=4.30,<5.0'
```

### Best Practices for Mixing

1. **Always install conda packages first**
2. **Use `--no-deps` carefully** for pip packages that duplicate conda dependencies
3. **Track pip packages in environment.yml**:

```yaml
dependencies:
  - python=3.10
  - numpy
  - pip
  - pip:
    - transformers
    - wandb
```

### Avoiding Common Pitfalls

```bash
# WRONG: pip installing something conda manages
conda install numpy
pip install numpy  # Creates duplicate, version conflicts

# BETTER: Install from one source
conda install numpy  # Use conda
# OR
pip install numpy    # Use pip only (in venv, not conda)
```

---

## Practical Workflows

### Starting a New Project

```bash
# 1. Create project directory
mkdir my_dl_project
cd my_dl_project

# 2. Create environment file
cat > environment.yml << 'EOF'
name: my_project
channels:
  - pytorch
  - conda-forge
dependencies:
  - python=3.10
  - pytorch>=2.0
  - torchvision
  - numpy
  - pandas
  - matplotlib
  - jupyterlab
  - ipykernel
  - pip
  - pip:
    - wandb
EOF

# 3. Create environment
conda env create -f environment.yml

# 4. Activate
conda activate my_project

# 5. Register Jupyter kernel
python -m ipykernel install --user --name=my_project --display-name "My Project"

# 6. Initialize git (track environment.yml)
git init
git add environment.yml
git commit -m "Initial environment setup"
```

### Adding Dependencies

```bash
# 1. Install the package
conda activate my_project
conda install new_package
# Or: pip install new_package

# 2. Update environment.yml
# Edit manually to add the package

# 3. Verify environment can be recreated
conda env remove -n test_env
conda env create -f environment.yml -n test_env
```

### Updating Dependencies

```bash
# Update all packages
conda update --all

# Update specific package
conda update pytorch

# Update and regenerate environment.yml
conda env export --from-history > environment.yml
```

### Sharing with Collaborators

```bash
# For cross-platform sharing
conda env export --from-history > environment.yml

# Include in repository
git add environment.yml
git commit -m "Update environment dependencies"
git push
```

Collaborator workflow:
```bash
git clone https://github.com/user/project.git
cd project
conda env create -f environment.yml
conda activate project_name
```

---

## Deep Learning Package Configuration

### PyTorch Installation Variants

```bash
# CPU only
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### TensorFlow Installation

```bash
# CPU
conda install -c conda-forge tensorflow

# GPU (automatically detects CUDA)
pip install tensorflow[and-cuda]
```

### Common Package Combinations

**Computer Vision:**
```yaml
dependencies:
  - pytorch
  - torchvision
  - opencv
  - pillow
  - albumentations
  - pip:
    - timm
    - segmentation-models-pytorch
```

**Natural Language Processing:**
```yaml
dependencies:
  - pytorch
  - pip:
    - transformers
    - tokenizers
    - datasets
    - sentencepiece
    - accelerate
```

**Reinforcement Learning:**
```yaml
dependencies:
  - pytorch
  - pip:
    - gymnasium
    - stable-baselines3
    - wandb
```

---

## Troubleshooting

### "Solving environment: failed"

**Cause**: No compatible package combination exists.

**Solutions**:
```bash
# Try fewer packages at once
conda install package1
conda install package2

# Use libmamba solver
conda install package --solver=libmamba

# Check for channel conflicts
conda config --show channels

# Create minimal environment and add incrementally
conda create -n test python=3.10
conda activate test
conda install package
```

### "PackagesNotFoundError"

**Cause**: Package not in configured channels.

**Solutions**:
```bash
# Search all channels
conda search package --channel conda-forge

# Install from specific channel
conda install -c conda-forge package

# Use pip
pip install package
```

### Import Errors After Installation

**Cause**: Wrong environment active or package installed in different location.

**Solutions**:
```bash
# Verify environment
which python
conda list package

# Reinstall
conda remove package
conda install package

# Check for pip/conda conflicts
pip list | grep package
conda list package
```

### Slow Solving

**Cause**: Classic solver with complex dependencies.

**Solutions**:
```bash
# Use libmamba
conda config --set solver libmamba

# Reduce package count
# Use environment.yml with --from-history

# Clean cache
conda clean --all
```

---

## Performance Optimization

### Faster Package Installation

```bash
# Use libmamba solver
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

# Parallel downloads
conda config --set concurrent.downloads 5

# Use mamba directly (alternative)
conda install -n base mamba
mamba install pytorch
```

### Reducing Environment Size

```bash
# Remove unused packages
conda clean --all

# Use --no-deps for packages with complex dependencies you don't need
pip install package --no-deps

# Create minimal environments
conda create -n minimal python=3.10 --no-default-packages
```

### Caching

```bash
# View cache location
conda info

# Package cache (shared across environments)
~/miniforge3/pkgs/

# Clear cache
conda clean --packages
conda clean --tarballs
```

---

## Summary

Effective package management is essential for reproducible deep learning research. Key takeaways:

1. **Use conda for environment management** — It handles Python and non-Python dependencies together.

2. **Prefer conda-forge channel** — Most comprehensive for scientific computing.

3. **Create environment.yml files** — Track dependencies in version control for reproducibility.

4. **Install conda packages first, pip second** — Avoid mixing that causes conflicts.

5. **Use libmamba solver** — Faster and more robust dependency resolution.

6. **Create project-specific environments** — Avoid polluting the base environment.

7. **Test environment recreation** — Regularly verify that `environment.yml` creates a working environment.

---

## Exercises

### Exercise 1: Create a Minimal Environment

Create an environment with only PyTorch and its dependencies. Export it and count the total number of packages installed.

### Exercise 2: Resolve a Conflict

Try installing both `tensorflow` and `pytorch` in the same environment. Document the conflicts you encounter and propose solutions.

### Exercise 3: Cross-Platform Sharing

Create an `environment.yml` that works on both macOS and Linux. Test it by creating the environment on both platforms (or using Docker).

---

## References

1. Conda Documentation. https://docs.conda.io
2. Conda-Forge. https://conda-forge.org
3. PyTorch Installation. https://pytorch.org/get-started/locally/
4. Pip Documentation. https://pip.pypa.io
5. Mamba Documentation. https://mamba.readthedocs.io
