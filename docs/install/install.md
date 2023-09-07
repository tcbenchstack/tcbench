# Install and config

## Download code and artifacts

If you see this documentation it means
you downloaded the file from figshare so you already have the code
in your hand :)

!!! note
    It is our intent to push all the code into a proper repository


## Configure a python environment :material-language-python:

We first create a `conda` environment to install
all required dependencies

```
conda create -n replicating-imc22-flowpic python=3.10 pip
conda activate replicating-imc22-flowpic
python -m pip install -r ./requirements.txt
```

The code artifacts are also a python package
that can be installed inside the environment.
From inside `/replicate_imc22_flowpic` run

```
python -m pip install .
```
