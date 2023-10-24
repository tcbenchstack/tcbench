---
icon: material/arrow-down-bold-box
---

# Datasets installation

Dataset installation is triggered with the `datasets install` subcommand

```
tcbench datasets install --help
```

!!! info "Output"
	```
	 Usage: tcbench datasets install [OPTIONS]

	 Install a dataset.

	╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
	│ *  --name          -n  [ucdavis-icdm19|utmobilenet21|mirage19|mirage22]  Dataset to install. [required]                         │
	│    --input-folder  -i  PATH                                              Folder where to find pre-downloaded tarballs.          │
	│    --help                                                                Show this message and exit.                            │
	╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
	```

The raw data of the datasets is either hosted on websites
or cloud environments. The automatic download from
those locations is available only for some of the datasets.

| Name | Auto download | 
|:----:|:-------------:|
|[`ucdavis-icdm19`](/tcbench/datasets/install/ucdavis-icdm19/)| :octicons-x-24: |
|[`mirage19`](/tcbench/datasets/install/mirage19/)| :material-check: |
|[`mirage22`](/tcbench/datasets/install/mirage22/)| :material-check: |
|[`utmobilenet21`](/tcbench/datasets/install/utmobilenet21/)| :octicons-x-24: |

If auto download is not possible, to install the dataset
you need to manually fetch the related archives, place them
in a folder, e.g., `/download`, and provide the `--input-folder`
option when triggering installation.

When installing a dataset, `tcbench` also
shows two types of reports as formatted tables.

* __Samples count__: This tables collect
the number of samples (i.e., flows)
available.

* __Stats__: The curation process
can filter out flows (e.g., based
on a minum number of packets
or remove classes without a minimum
number of flows). As such, when 
installing, `tcbench` is showing
general stats (mean, std, percentiles)
about number of packets
for each flow across classes.

Please check the specific install page for each dataset for more details.


## Datasets deletion

The datasets files are installed within the 
python environment where tcbench is installed.

You can delete a dataset using the following command

```
tcbench datasets delete --name <dataset-name>
```

