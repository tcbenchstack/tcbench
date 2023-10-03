---
icon: octicons/package-16
---

# Install

First prepare a python virtual environment, for example via :simple-anaconda: conda
```
conda create -n tcbench python=3.10 pip
conda activate tcbench
```

Then simply install via pip
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
    version: 0.0.17
    ```
