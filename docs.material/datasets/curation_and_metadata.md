---
icon: material/medical-bag
title: Curation & Meta-data
---

## Datasets curation

The curation process operated by tcbench aims at the following objectives.

!!! quote "Curation objectives"

    1. Transform all datasets into a common format,
    apply some clearning if/where needed and 
    create a reference set of data split for modeling.

    2. All data should reflect per-flow records where each flow is
    associated to packet time series input.


To do so, tcbench applies an *opinionated* preprocessing
starting from the original *raw* version of the data
(i.e., the data offered by the authors of each dataset).

The raw data go through the following steps:

1. __Install__: tcbench can fetch the raw data from the Internet or 
can take as input a folder where the raw data were already downloaded.
This data is then "installed" by unpacking it into
the python environment were tcbench is installed.

2. __Preprocess__: Once unpacked, the raw data 
is converted into __monolithic packet files__.
Such files are left *untouched*, i.e., they simply
serve as a re-organization of the original data
(with a per-flow view where needed) with an
homogeneous format across datasets.

3. __Filter and split__: The monolithic parquet files
are first filtered (e.g., removing very short
flows or flow related to invalid IP addresses) and
then used to train/validation/test splits.

We reiterate that all steps are a necessity
to enable ML/DL modeling. Yet, they are *opinionated* 
processes so tcbench embodies just on option to
perform them.

## Datasets Meta-data

As part of the curation process, 
tcbench can easily show the meta-data
related to the datasets.

For instance, you can see the 
[datasets summary table](/tcbench/datasets/#table-datasets-properties)
by running

```
tcbench datasets info
```

!!! info "Output"
	```
	Datasets
	├── ucdavis-icdm19
	│   └──  🚩 classes:           5
	│        🔗 paper_url:         https://arxiv.org/pdf/1812.09761.pdf
	│        🔗 website:           https://github.com/shrezaei/Semi-supervised-Learning-QUIC-
	│        🔗 data:              https://drive.google.com/drive/folders/1Pvev0hJ82usPh6dWDlz7Lv8L6h3JpWhE
	│        🔗 curated data:      https://figshare.com/ndownloader/files/42437043
	│        ➕ curated data MD5:  9828cce0c3a092ff19ed77f9e07f317c
	│        📁 installed:         None
	│        📁 preprocessed:      None
	│        📁 data splits:       None
	├── mirage19
	│   └──  🚩 classes:       20
	│        🔗 paper_url:     http://wpage.unina.it/antonio.montieri/pubs/MIRAGE_ICCCS_2019.pdf
	│        🔗 website:       https://traffic.comics.unina.it/mirage/mirage-2019.html
	│        🔗 data:          https://traffic.comics.unina.it/mirage/MIRAGE/MIRAGE-2019_traffic_dataset_downloadable_v2.tar.gz
	│        📁 installed:     None
	│        📁 preprocessed:  None
	│        📁 data splits:   None
	├── mirage22
	│   └──  🚩 classes:       9
	│        🔗 paper_url:     http://wpage.unina.it/antonio.montieri/pubs/_C__IEEE_CAMAD_2021___Traffic_Classification_Covid_app.pdf
	│        🔗 website:       https://traffic.comics.unina.it/mirage/mirage-covid-ccma-2022.html
	│        🔗 data:          https://traffic.comics.unina.it/mirage/MIRAGE/MIRAGE-COVID-CCMA-2022.zip
	│        📁 installed:     None
	│        📁 preprocessed:  None
	│        📁 data splits:   None
	└── utmobilenet21
		└──  🚩 classes:           17
			 🔗 paper_url:         https://ieeexplore.ieee.org/abstract/document/9490678/
			 🔗 website:           https://github.com/YuqiangHeng/UTMobileNetTraffic2021
			 🔗 data:              https://utexas.app.box.com/s/okrimcsz1mn9ec4j667kbb00d9gt16ii
			 🔗 curated data:      https://figshare.com/ndownloader/files/42436353
			 ➕ curated data MD5:  e1fdcffa41a0f01d63eaf57c198485ce
			 📁 installed:         None
			 📁 preprocessed:      None
			 📁 data splits:       None
	```

Notice the three properties `installed`, `preprocessed` and `data_splits`.
These show the *absolute* path where the data is stored.
As mentioned before, notice that the data is installed inside
the python environment where tcbench is installed.

The example above refers to an environment with no dataset installed yet.

The following instead show a case where all datasets are installed.

!!! info "Output"
	```
	Datasets
	├── ucdavis-icdm19
	│   └──  🚩 classes:           5
	│        🔗 paper_url:         https://arxiv.org/pdf/1812.09761.pdf
	│        🔗 website:           https://github.com/shrezaei/Semi-supervised-Learning-QUIC-
	│        🔗 data:              https://drive.google.com/drive/folders/1Pvev0hJ82usPh6dWDlz7Lv8L6h3JpWhE
	│        🔗 curated data:      https://figshare.com/ndownloader/files/42437043
	│        ➕ curated data MD5:  9828cce0c3a092ff19ed77f9e07f317c
	│        📁 installed:         ./envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19/raw
	│        📁 preprocessed:      ./envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19/preprocessed
	│        📁 data splits:       ./envs/tcbenchlib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19/preprocessed/imc23
	├── mirage19
	│   └──  🚩 classes:       20
	│        🔗 paper_url:     http://wpage.unina.it/antonio.montieri/pubs/MIRAGE_ICCCS_2019.pdf
	│        🔗 website:       https://traffic.comics.unina.it/mirage/mirage-2019.html
	│        🔗 data:          https://traffic.comics.unina.it/mirage/MIRAGE/MIRAGE-2019_traffic_dataset_downloadable_v2.tar.gz
	│        📁 installed:     None
	│        📁 preprocessed:  ./envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/mirage19/preprocessed
	│        📁 data splits:   ./envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/mirage19/preprocessed/imc23
	├── mirage22
	│   └──  🚩 classes:       9
	│        🔗 paper_url:     http://wpage.unina.it/antonio.montieri/pubs/_C__IEEE_CAMAD_2021___Traffic_Classification_Covid_app.pdf
	│        🔗 website:       https://traffic.comics.unina.it/mirage/mirage-covid-ccma-2022.html
	│        🔗 data:          https://traffic.comics.unina.it/mirage/MIRAGE/MIRAGE-COVID-CCMA-2022.zip
	│        📁 installed:     None
	│        📁 preprocessed:  ./envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/mirage22/preprocessed
	│        📁 data splits:   ./envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/mirage22/preprocessed/imc23
	└── utmobilenet21
		└──  🚩 classes:           17
			 🔗 paper_url:         https://ieeexplore.ieee.org/abstract/document/9490678/
			 🔗 website:           https://github.com/YuqiangHeng/UTMobileNetTraffic2021
			 🔗 data:              https://utexas.app.box.com/s/okrimcsz1mn9ec4j667kbb00d9gt16ii
			 🔗 curated data:      https://figshare.com/ndownloader/files/42436353
			 ➕ curated data MD5:  e1fdcffa41a0f01d63eaf57c198485ce
			 📁 installed:         None
			 📁 preprocessed:      ./envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/utmobilenet21/preprocessed
			 📁 data splits:       ./envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/utmobilenet21/preprocessed/imc23
	```
	
Notice that

* __installed__ may or may not be `None` depending on
    if the dataset was installed from the raw data
    or via tcbench artifacts (see XXX).

* __preprocessed__ and __data splits__ are always
    reported unless the dataset was not installed.

## Dataset files

As mentioned, the datasets files are installed within the 
python environment where tcbench is installed.

Generally speaking, tcbench API are designed to
ease the process of loading the data without
the need to master the internal organization
of a dataset. Check the XXX tutorial.

In case of need, you can inspect the internal
organization of a dataset using the 
`datasets lsfiles` subcommand.

For instance, for the [`ucdavis-icdm19`](/tcbench/datasets/install/ucdavis-icdm19/) datasets
```
tcbench datasets lsfiles --name ucdavis-icdm19
```

!!! note "Output"

	```
	tcbench datasets lsfiles --name ucdavis-icdm19
	Datasets
	└── ucdavis-icdm19
		└── 📁 preprocessed/
			├── ucdavis-icdm19.parquet
			└── 📁 imc23/
				├── test_split_human.parquet
				├── test_split_script.parquet
				├── train_split_0.parquet
				├── train_split_1.parquet
				├── train_split_2.parquet
				├── train_split_3.parquet
				└── train_split_4.parquet
	```

While the `info` subcommand show the absolute root path
where datasets are installed, the `lsparquet` shows
the relative paths and internal hierarchical structure
of how the data is locally installed.

In the sample above

* the `raw/` folder where the original raw data
of the dataset is never displayed.

* the `preprocessed/ucdavis-icdm19.parquet` corresponds
to the monolithic parquet obtained reformatting
(but not curating) the raw data.

* the files under the `preprocessed/imc23` 
corresponds to the curation operated to the
dataset based on our [IMC23](/tcbench/papers/imc23) paper.
