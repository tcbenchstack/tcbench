---
icon: material/cloud-download-outline
title: Import
---

# Import curated datasets

The `datasets` command offers also the option
to import a pre-computed curation of datasets.

This is 

* To avoid spending computation. 
Some of the preprocessing requires ingenuity and
multiprocessing/multicore architecture. 

* Further strength replicability (although
the curation process of tcbench is deterministic).

The [datasets summary table](/datasets/#table-datasets-properties) indicates that the
not all datasets have the curated data already available.
This is because some datasets (namely MIRAGE) has
tighter licensing. For these datasets
please refer to the related installation page.

## The `import` subcommand

For datasets which licensing allows to redistribute
modified version, the curated data is stored
in a public [:simple-figshare: figshare collection](https://figshare.com/collections/IMC23_artifacts_-_Replication_Contrastive_Learning_and_Data_Augmentation_in_Traffic_Classification_Using_a_Flowpic_Input_Representation/6849252).

You can manually fetch the datasets from the collection or use
automate their installation with the `datasets import` subcommand.

```
tcbench datasets import --name ucdavis-icdm19
```

!!! info Output
	```
	Downloading... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 554.2 MB / 554.2 MB eta 0:00:00
	opening: /tmp/tmpb586lqhh/42438621

	Files installed
	Datasets
	└── ucdavis-icdm19
		└── 📁 preprocessed/
			├── ucdavis-icdm19.parquet
			├── LICENSE
			└── 📁 imc23/
				├── test_split_human.parquet
				├── test_split_script.parquet
				├── train_split_0.parquet
				├── train_split_1.parquet
				├── train_split_2.parquet
				├── train_split_3.parquet
				└── train_split_4.parquet
	```


Notice that `installed` is not set. Indeed
the prepared curated datasets do NOT repack
the original datasets, just the preprocessed ones 
(see the [meta-data](datasets/metadata/#samples-count-reports) page).

You can also import the curated data by downloading the individual
archives from figshare and use the `--archive` option

```
tcbench datasets import \
	--name ucdavis-icdm19 \
	--archive <tarball>
```

!!! warning ":simple-figshare: Figshare versioning"
	
	Figshare updates the version of a published entry for any modification
    to any of the elements related to the entry (including changes to 
    description). 

	tcbench is configured to automatically fetch the latest version of
    the curated datasets. But if you download them manually make
    sure to download the latest versions


## The `delete` subcommand

You can use the `delete` subcommand to remove installed/imported datasets.

For instance, continuing the example above

```
tcbench datasets delete --name ucdavis-icdm19
```

...now `info` stats all data for `ucdavis-icdm19` has been removed

```
tcbench datasets info --name ucdavis-icdm19
```
!!! info "Output"
	```
	Datasets
	└── ucdavis-icdm19
		└──  🚩 classes:           5
			 🔗 paper_url:         https://arxiv.org/pdf/1812.09761.pdf
			 🔗 website:           https://github.com/shrezaei/Semi-supervised-Learning-QUIC-
			 🔗 data:              https://drive.google.com/drive/folders/1Pvev0hJ82usPh6dWDlz7Lv8L6h3JpWhE
	         🔗 curated data:      https://figshare.com/ndownloader/files/42437043
	         ➕ curated data MD5:  9828cce0c3a092ff19ed77f9e07f317c
			 📁 installed:         None
			 📁 preprocessed:      None
			 📁 data splits:       None
	```
