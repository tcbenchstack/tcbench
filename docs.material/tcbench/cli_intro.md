---
icon: octicons/terminal-16
title: CLI Intro
---

# CLI Introduction

tcbench can be used for as SDK and
from the command line.

When installing tcbench you install
also a `tcbench` command line script
created via [:material-cursor-default: click](https://click.palletsprojects.com/en/8.1.x/) 
and [:material-language-python: rich](https://github.com/Textualize/rich).

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
    │ tree            show the command tree of your CLI.                                       │
    ╰──────────────────────────────────────────────────────────────────────────────────────────╯
    ```

The commands are organized in a nested structure which
you can visualize using

```
tcbench tree
```

!!! info "Output"
    ```bash
    main
    ├── aimrepo - Investigate AIM repository content.
    │   ├── ls - List a subset of properties of each run.
    │   ├── merge - Coalesce different AIM repos into a single new repo.
    │   ├── properties - List properties across all runs.
    │   └── report - Summarize runs performance metrics.
    ├── campaign - Triggers a modeling campaign.
    │   ├── augment-at-loading - Modeling by applying data augmentation when loading the training set.
    │   └── contralearn-and-finetune - Modeling by pre-training via constrative learning and then finetune the final classifier from the pre-trained model.
    ├── datasets - Install/Remove traffic classification datasets.
    │   ├── delete - Delete a dataset.
    │   ├── import - Import datasets.
    │   ├── info - Show the meta-data related to supported datasets.
    │   ├── install - Install a dataset.
    │   ├── lsparquet - Tree view of the datasets parquet files.
    │   ├── samples-count - Show report on number of samples per class.
    │   └── schema - Show datasets schemas
    ├── run - Triggers a modeling run.
    │   ├── augment-at-loading - Modeling by applying data augmentation when loading the training set.
    │   └── contralearn-and-finetune - Modeling by pre-training via constrative learning and then finetune the final classifier from the pre-trained model.
    └── tree - show the command tree of your CLI
    ```
