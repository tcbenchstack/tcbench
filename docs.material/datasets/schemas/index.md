---
icon: material/table
title: Schemas
---

# Datasets schemas

Despite the [curation](/tcbench/datasets/curation_and_metadata/), datasets can have intrinsically
different schemas.

You can investigate those on the command line via
the `datasets schema` sub-command.

```
tcbench datasets schema --help

 Usage: tcbench datasets schema [OPTIONS]

 Show datasets schemas

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --name  -n  [ucdavis-icdm19|utmobilenet21|mirage19|mirage22]  Dataset to install                                         │
│ --type  -t  [unfiltered|filtered|splits]                      Schema type (unfiltered: original raw data; filtered:      │
│                                                               curated data; splits: train/val/test splits)               │
│ --help                                                        Show this message and exit.                                │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

Beside the dataset name `--name`, the selection
of the schema is simplified via a single parameter `--type`
which matches the parquet files as follows

* `"unfiltered"` corresponds to the monolithic 
before any filtering (i.e., the files under `/preprocessed`)

* `"filtered"` corresponds to the filtered 
version of the monolithic files (i.e., the files
having `minpkts<N>` in the filename).

* `"splits"` corresponds to the split files
(i.e., the files having `xyz_split.parquet`
in the filename).


