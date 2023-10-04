---
icon: octicons/package-16
---

# Install

First prepare a python virtual environment, for example via :simple-anaconda: conda
```
conda create -n tcbench python=3.10 pip
conda activate tcbench
```

tcbench is [availabe on pypi](https://pypi.org/project/tcbench/) so you install it via pip
```
python -m pip install tcbench
```

All dependecies are automatically pulled.

Verify the installation was successful by running
```
tcbench --version
```

!!! note "Output"
    ```
    version: 0.0.21
    ```

# Developer

For developing your own projects or contributing
to tcbench fork/clone the [official repository](https://github.com/tcbenchstack/tcbench)
and install the developer version.

```
python -m pip install .[dev]
```

The only difference with respect to the base version
is the installation of extra dependencies.
