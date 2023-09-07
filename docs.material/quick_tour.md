# Quick tour

The code base is collected into a python package named `tcbench`
which is designed to cover two functionalities

1. Easy install and access to a curated set of traffic classification datasets
2. Use the datasets to train/test ML and DL models


## :material-language-python: Install `tcbench`

!!! note "If you unpacked the artifacts..."
    When [unpacking the artifacts](/artifacts/#unpack-artifacts), `tcbench` has been already installed

First prepare a python virtual environment, for example via :simple-anaconda: conda
```
conda create -n tcbench python=3.10 pip
conda activate tcbench
```

Grab the latest `code_artifacts_paper132.tgz` from [:simple-figshare: figshare](https://figshare.com/s/cab23f730cfbc5172f78) and unpack it.
It contains a folder `/code_artifacts_paper132` from which you can trigger the installation.

```
cd code_artifacts_paper132
python -m pip install .
```

All dependecies are automatically installed.

## :octicons-terminal-16: `tcbench` cli

When installing the package you also install
a `tcbench` CLI script which 
acts as a universal entry
point to interact with the framework
via a nested commands structure.

For instance
```
tcbench --help
```

!!! info "Output"
    ```bash
     Usage: tcbench [OPTIONS] COMMAND [ARGS]...

    ╭─ Options ────────────────────────────────────────────────────────────────────────────────╮
    │ --version      Show tcbench version and exit.                                            │
    │ --help         Show this message and exit.                                               │
    ╰──────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ Commands ───────────────────────────────────────────────────────────────────────────────╮
    │ aimrepo         Investigate AIM repository content.                                      │
    │ campaign        Triggers a modeling campaign.                                            │
    │ datasets        Install/Remove traffic classification datasets.                          │
    │ run             Triggers a modeling run.                                                 │
    ╰──────────────────────────────────────────────────────────────────────────────────────────╯
    ```

For instance command `datasets` offers the following sub-commands

```
tcbench datasets --help
```

!!! info "Output"
    ```
	Usage: tcbench datasets [OPTIONS] COMMAND [ARGS]...

	 Install/Remove traffic classification datasets

	╭─ Options ────────────────────────────────────────────────────────────────────────────────╮
	│ --help      Show this message and exit.                                                  │
	╰──────────────────────────────────────────────────────────────────────────────────────────╯
	╭─ Commands ───────────────────────────────────────────────────────────────────────────────╮
	│ delete                Delete a dataset                                                   │
	│ import                Import datasets                                                    │
	│ info                  Show the meta-data related to supported datasets                   │
	│ install               Install a dataset                                                  │
	│ lsparquet             Tree view of the datasets parquet files                            │
	│ samples-count         Show report on number of samples per class                         │
	│ schema                Show datasets schemas                                              │
	╰──────────────────────────────────────────────────────────────────────────────────────────╯
    ```

Those sub-commands in turn offers different options. For instance for `install`

```
tcbench datasets install --help
```

!!! info "Output"
    ```
     Usage: tcbench datasets install [OPTIONS]

     Install a dataset

    ╭─ Options ────────────────────────────────────────────────────────────────────────────────╮
    │ *  --name          -n  [ucdavis-icdm19|utmobilenet21|m  Dataset to install. [required]   │
    │                        irage19|mirage22]                                                 │
    │    --input-folder  -i  PATH                             Folder where to find             │
    │                                                         pre-downloaded tarballs.         │
    │    --help                                               Show this message and exit.      │
    ╰──────────────────────────────────────────────────────────────────────────────────────────╯
    ```
