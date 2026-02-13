# Virtual Environments

## Learning Objectives

By the end of this section, you will be able to:

- Understand why environment isolation is essential for deep learning projects
- Create and manage conda environments for project-specific work
- Use pip + venv as an alternative isolation strategy
- Export and reproduce environments across machines
- Apply best practices for environment management

---

## Why Isolate Environments?

A virtual environment is a self-contained directory tree that holds its own Python interpreter and its own set of installed packages. Isolation solves three recurring problems:

1. **Dependency conflicts** — Project A needs `torch 2.0` while Project B requires `torch 2.3`. Without isolation, installing one breaks the other.
2. **Reproducibility** — An environment file records the exact package versions used in an experiment, letting collaborators (or your future self) recreate the setup.
3. **System protection** — Installing packages into the base/system Python can corrupt OS-level tools that depend on specific library versions.

---

## Conda Environments

### Create

```bash
# minimal environment (inherits nothing from base)
conda create --name dl_env python=3.11

# with initial packages
conda create --name dl_env python=3.11 numpy pandas matplotlib jupyter
```

`--name` (or `-n`) sets the environment name. You can also use `--prefix` to place the environment in an arbitrary directory:

```bash
conda create --prefix ./envs/dl_env python=3.11
```

### Activate / Deactivate

```bash
conda activate dl_env      # prompt changes to (dl_env)

# ... do your work ...

conda deactivate           # returns to (base) or system Python
```

!!! important
    Always activate the target environment **before** running `conda install` or `pip install`. Otherwise the package lands in `base`.

### Install Packages

```bash
# conda-hosted packages
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# pip-only packages (inside the activated environment)
pip install transformers datasets
```

### List Packages

```bash
conda list                 # everything in the active environment
conda list numpy           # filter by name
```

### Export / Reproduce

```bash
# export to a YAML file
conda env export > environment.yml

# recreate on another machine
conda env create -f environment.yml
```

A minimal cross-platform alternative uses `--from-history`, which records only the packages you explicitly requested (not their transitive dependencies):

```bash
conda env export --from-history > environment_minimal.yml
```

### Clone and Remove

```bash
# clone an existing environment
conda create -n dl_backup --clone dl_env

# rename (clone + remove)
conda create -n new_name --clone old_name
conda env remove -n old_name

# remove environment
conda env remove --name dl_env
```

---

## pip + venv (Alternative)

If you use standard CPython rather than Anaconda, the built-in `venv` module provides lightweight isolation:

```bash
# create
python -m venv .venv

# activate
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# install
pip install numpy torch

# freeze requirements
pip freeze > requirements.txt

# recreate elsewhere
pip install -r requirements.txt

# deactivate
deactivate
```

`venv` environments do **not** manage native libraries (BLAS, CUDA, cuDNN). Those must be installed at the system level.

---

## Best Practices

| Practice | Rationale |
|---|---|
| **One environment per project** | Prevents cross-project contamination |
| **Pin Python version** in `conda create` | Avoids surprises when `conda` defaults change |
| **Commit `environment.yml`** (or `requirements.txt`) to version control | Ensures reproducibility |
| **Prefer `conda` for packages with native deps** (PyTorch, OpenCV, CUDA) | `conda` handles C/C++ linking automatically |
| **Use `pip` only for packages absent from conda channels** | Mixing managers is fine, but let `conda` resolve first |
| **Never install into `base`** | Keep `base` clean so `conda` itself stays updatable |

---

## Common Commands Cheat-Sheet

```bash
conda env list                        # list all environments
conda activate dl_env                 # switch into dl_env
conda deactivate                      # leave current environment
conda create -n dl_env python=3.11    # create with specific Python
conda env export > environment.yml    # snapshot
conda env create -f environment.yml   # recreate from snapshot
conda env remove -n dl_env           # delete
conda clean --all                     # free cached package tarballs
```
