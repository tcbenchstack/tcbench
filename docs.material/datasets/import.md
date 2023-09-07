The `datasets` command offers also the option
to import a pre-computed curation of datasets.

This is 

* To avoid spending computation. 
Some of the preprocessing requires ingenuity and
multiprocessing/multicore architecture. 

* Further strength replicability (although
the curation process is deterministic).

## The `import` subcommand

To `import` sub-command enables to 
add to `tcbench` a pre-created curated datasets, e.g.,
the artifacts [available on figshare]().

It requires the data to be in a folder (so 
unpack the tarball if you use prepared artifacts).
For instance, assuming the data is stored under `./curated-datasets`

```
tcbench datasets import --input-folder ./curated-datasets
```

!!! info Output
	```
	tcbench datasets info --name ucdavis-icdm19
	Datasets
	└── ucdavis-icdm19
		└──  🚩 classes:       5
			 🔗 paper_url:     https://arxiv.org/pdf/1812.09761.pdf
			 🔗 website:       https://github.com/shrezaei/Semi-supervised-Learning-QUIC-
			 🔗 data:          https://drive.google.com/drive/folders/1Pvev0hJ82usPh6dWDlz7Lv8L6h3JpWhE
			 📁 installed:     None
			 📁 preprocessed:  /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm1
							   9/preprocessed
			 📁 data splits:   /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm1
							   9/preprocessed/imc23
	```

Notice that `installed` is not set. Indeed
the prepared curated datasets do NOT repack
the original datasets, just the preprocessed ones (see the [meta-data](datasets/metadata/#samples-count-reports) page).

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
		└──  🚩 classes:       5
			 🔗 paper_url:     https://arxiv.org/pdf/1812.09761.pdf
			 🔗 website:       https://github.com/shrezaei/Semi-supervised-Learning-QUIC-
			 🔗 data:          https://drive.google.com/drive/folders/1Pvev0hJ82usPh6dWDlz7Lv8L6h3JpWhE
			 📁 installed:     None
			 📁 preprocessed:  None
			 📁 data splits:   None
	```
