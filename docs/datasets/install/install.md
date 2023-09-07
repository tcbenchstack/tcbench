## Supported datasets

TCBench integrates and *curates* the following traffic classification datasets

##### Table : Datasets properties
| Name | No. Classes | Links | Auto-download |
|:----:|:-----------:|:-----:|:-------------:|
|`ucdavis-icdm19`|5|[:fontawesome-regular-file-pdf:](https://arxiv.org/pdf/1812.09761.pdf)[:material-package-down:](https://drive.google.com/drive/folders/1Pvev0hJ82usPh6dWDlz7Lv8L6h3JpWhE)[:material-github:](https://github.com/shrezaei/Semi-supervised-Learning-QUIC-)|:octicons-x-12:|
|`mirage19`|20|[:fontawesome-regular-file-pdf:](http://wpage.unina.it/antonio.montieri/pubs/MIRAGE_ICCCS_2019.pdf)[:material-package-down:](https://traffic.comics.unina.it/mirage/MIRAGE/MIRAGE-2019_traffic_dataset_downloadable_v2.tar.gz)[:material-web:](https://traffic.comics.unina.it/mirage/mirage-2019.html)|:heavy_check_mark:|
|`mirage22`|9|[:fontawesome-regular-file-pdf:](http://wpage.unina.it/antonio.montieri/pubs/_C__IEEE_CAMAD_2021___Traffic_Classification_Covid_app.pdf)[:material-package-down:](https://traffic.comics.unina.it/mirage/MIRAGE/MIRAGE-COVID-CCMA-2022.zip)[:material-web:](https://traffic.comics.unina.it/mirage/mirage-covid-ccma-2022.html)|:heavy_check_mark:|
|`utmobilenet21`|17|[:fontawesome-regular-file-pdf:](https://ieeexplore.ieee.org/abstract/document/9490678/)[:material-package-down:](https://github.com/YuqiangHeng/UTMobileNetTraffic2021)[:material-github:](https://github.com/YuqiangHeng/UTMobileNetTraffic2021)|:octicons-x-12:|

Unfortunately, there is no single format used when preparing datasets for public release.

* Datasets come as __either CSV or JSON files collections__ with either
per-packet or per-flow records. 

* __Files can be organized in subfolders__, namely __partitions__,
named to reflect the related measurement campaign
(see `ucdavis-icdm19`, `utmobilenet21`).

* __File names can carry semantic__ and/or the classification 
require preprocessing to be obtained by separating
it from background (see `mirage19` and `mirage22`).

* Datasets typically __do not have native splits__
(i.e., train/validation/test splits) nor are ready to use
for app modeling (e.g., ICMP packets, short flows, etc., are not filtered).

## Datasets curation at-glance

:material-target:
==*We target per-flow classification where each flow is
associated to packet time series input.*==
:material-target:

To do so, we take an opinionated view on how
datasets should be formatted and handled.

Specifically, when installing a dataset, the dataset raw
data goes through the following steps:

1. __Download and unpack__: `tcbench` enables you
to directly fetch the raw data from the Internet or install it
from a folder where data was pre-downloaded, and
unpack it into a predefined destination folder which, 
for simplicity, corresponds to a
subfolder of the python environment where 
[`tcbench` is installed](/modeling_framework/install).
This folder is mantained and can accessed for futher
ad-hoc processing outside the functionalities of `tcbench`.

2. __Preprocess__: Once unpacked, the dataset
raw data is converted into __monolithic packet files__.
Such files are left *untouched*, i.e., they simply
serve as a re-organization of the original data
(with a per-flow view where needed) with an
homogeneous format across datasets.

3. __Filter and split__: The monolithic parquet files
are first filtered (e.g., removing very short
flows or flow related to invalid IP addresses) and
then used to train/validation/test splits.
Both steps are necessary to enable traffic modeling
and replicability; yet they reflect our opinionated view
on how to handle dataset for traffic classification.

## Datasets meta-data

As part of the curation process, 
`tcbench` enables you to show meta-data
related to the datasets.

For instance, the refences collected
in the [summary table](#table-datasets-properties) reported
at the top of this page
can be visualized issuing

```
tcbench datasets info
```

!!! info "Output"
	```
	Datasets
	├── ucdavis-icdm19
	│   └──  🚩 classes:       5
	│        🔗 paper_url:     https://arxiv.org/pdf/1812.09761.pdf
	│        🔗 website:       https://github.com/shrezaei/Semi-supervised-Learning-QUIC-
	│        🔗 data:          https://drive.google.com/drive/folders/1Pvev0hJ82usPh6dWDlz7Lv8L6h3JpWhE
	│        📁 installed:     None
	│        📁 preprocessed:  None
	│        📁 data splits:   None
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
		└──  🚩 classes:       17
			 🔗 paper_url:     https://ieeexplore.ieee.org/abstract/document/9490678/
			 🔗 website:       https://github.com/YuqiangHeng/UTMobileNetTraffic2021
			 🔗 data:          https://utexas.app.box.com/s/okrimcsz1mn9ec4j667kbb00d9gt16ii
			 📁 installed:     None
			 📁 preprocessed:  None
			 📁 data splits:   None
	```


Beside showing the a set of static properties (e.g., URL links), 
the 3 properties `installed`, `preprocessed` nd `data_splits` 
reports the absolute path where the related data is stored.
The example refers to the initial setup where no dataset is yet
installed.

However, when [unpacking artifacts with the 
provided scripts](/artifacts/#unpack-artifacts), 
the curated datasets are automatically installed

```
tcbench datasets info
```

!!! info "Output"
	```
	Datasets
	├── ucdavis-icdm19
	│   └──  🚩 classes:       5
	│        🔗 paper_url:     https://arxiv.org/pdf/1812.09761.pdf
	│        🔗 website:       https://github.com/shrezaei/Semi-supervised-Learning-QUIC-
	│        🔗 data:          https://drive.google.com/drive/folders/1Pvev0hJ82usPh6dWDlz7Lv8L6h3JpWhE
	│        📁 installed:     None
	│        📁 preprocessed:  /home/johndoe/.conda/envs/tcbench-johndoe/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19/preprocessed
	│        📁 data splits:   /home/johndoe/.conda/envs/tcbench-johndoe/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19/preprocessed/imc23
	├── mirage19
	│   └──  🚩 classes:       20
	│        🔗 paper_url:     http://wpage.unina.it/antonio.montieri/pubs/MIRAGE_ICCCS_2019.pdf
	│        🔗 website:       https://traffic.comics.unina.it/mirage/mirage-2019.html
	│        🔗 data:          https://traffic.comics.unina.it/mirage/MIRAGE/MIRAGE-2019_traffic_dataset_downloadable_v2.tar.gz
	│        📁 installed:     None
	│        📁 preprocessed:  /home/johndoe/.conda/envs/tcbench-johndoe/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/mirage19/preprocessed
	│        📁 data splits:   /home/johndoe/.conda/envs/tcbench-johndoe/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/mirage19/preprocessed/imc23
	├── mirage22
	│   └──  🚩 classes:       9
	│        🔗 paper_url:     http://wpage.unina.it/antonio.montieri/pubs/_C__IEEE_CAMAD_2021___Traffic_Classification_Covid_app.pdf
	│        🔗 website:       https://traffic.comics.unina.it/mirage/mirage-covid-ccma-2022.html
	│        🔗 data:          https://traffic.comics.unina.it/mirage/MIRAGE/MIRAGE-COVID-CCMA-2022.zip
	│        📁 installed:     None
	│        📁 preprocessed:  /home/johndoe/.conda/envs/tcbench-johndoe/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/mirage22/preprocessed
	│        📁 data splits:   /home/johndoe/.conda/envs/tcbench-johndoe/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/mirage22/preprocessed/imc23
	└── utmobilenet21
		└──  🚩 classes:       17
			 🔗 paper_url:     https://ieeexplore.ieee.org/abstract/document/9490678/
			 🔗 website:       https://github.com/YuqiangHeng/UTMobileNetTraffic2021
			 🔗 data:          https://utexas.app.box.com/s/okrimcsz1mn9ec4j667kbb00d9gt16ii
			 📁 installed:     None
			 📁 preprocessed:  /home/johndoe/.conda/envs/tcbench-johndoe/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/utmobilenet21/preprocessed
			 📁 data splits:   /home/johndoe/.conda/envs/tcbench-johndoe/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/utmobilenet21/preprocessed/imc23
	```
	
	Notice that
	* __installed__ is `None` because this refers to the
		original raw data of the datasets (which is not
		provided with the curated artifacts).

	* __preprocessed__ and __data splits__ are two specific types
		of curation of the datasets.

	In fact, the artifacts unpacking uses a [dataset import](/datasets/import) process.


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

Please check out the [datasets meta-data](/datasets/metadata) page for more details.


## `ucdavis-icdm19`

This dataset cannot be downloaded directly
(as it is stored on Google Drive). So,
it needs to be manually downloaded.

More specifically, the dataset is composed
of 3 zip files that need to be kept in
a single folder.

For instance
```
downloads/
├── pretraining.zip
├── Retraining(human-triggered).zip
└── Retraining(script-triggered).zip
```

#### Original structure

The 3 files correspond to 3 *partitions*
with different scopes: `pretraining` is 
meant for training while the other two
for testing.

When all zips are unpacked, the folder structure becomes
```
downloads/
├── pretraining
│   ├── Google Doc
│   ├── Google Drive
│   ├── Google Music
│   ├── Google Search
│   └── Youtube
├── Retraining(human-triggered)
│   ├── Google Doc
│   ├── Google Drive
│   ├── Google Music
│   ├── Google Search
│   └── Youtube
└── Retraining(script-triggered)
    ├── Google Doc
    ├── Google Drive
    ├── Google Music
    ├── Google Search
    └── Youtube
```

Inside each nested folder there is a collection of CSV files.
Each file corresponds to a different flow, 
where each row represents individual packet information.

#### Curation

* __No filtering__ is applied to the original data.

* The only processing applied is to "transpose" 
the data with respect to the original representation.
Specifically, original CSV are per-packet while we
create pandas DataFrames where one row 
represents one flow and we gather
all packets of the flow into numpy arrays.

* During the conversion the original folder
structure is preserved via extra columns
(`partition` and `flow_id`).

#### Splits

This dataset was used in [:material-file-document-outline:`imc22-paper`](/index.md), hence we follow the splits described in the paper

* From `pretraining` we generate 5 random splits, each with 100 samples per class.
* The other two partitions are left as is and are used for testing.

Both training and testing splits are "materialized", i.e.,
differently from `mirage19`, `mirage22`, and `utmobilenet21`,
the splits are NOT collection or row indexes but
rather already filtered views of the monolithic 
parquet files.

Hence, all splits have the same columns.

| Field | Description |
|:------|:------------|
|`row_id`| A unique row id|
|`app`| The label of the flow, encoded as pandas `category`|
|`flow_id`| The original filename|
|`partition`| The partition related to the flow|
|`num_pkts`| Number of packets in the flow|
|`duration`| The duration of the flow|
|`bytes`| The number of bytes of the flow|
|`unixtime`| Numpy array with the absolute time of each packet|
|`timetofirst`| Numpy array with the delta between a packet the first packet of the flow|
|`pkts_size`| Numpy array for the packet size time series|
|`pkts_dir`| Numpy array for the packet direction time series|
|`pkts_iat`| Numpy array for the packet inter-arrival time series|

#### Install

To install the dataset run (assuming data was pre-downloaded under `/downloads`)

```
tcbench datasets install \
	--name ucdavis-icdm19 \
	--input-folder ./downloads/
```


!!! info "Output"
	```
    ╭──────╮
    │unpack│
    ╰──────╯
    opening: downloads/pretraining.zip
    opening: downloads/Retraining(human-triggered).zip
    opening: downloads/Retraining(script-triggered).zip

    ╭──────────╮
    │preprocess│
    ╰──────────╯
    found 6672 CSV files to load
    Converting CSVs... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
    concatenating files
    saving: /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19/preprocessed/ucdavis-icdm19.parquet
    samples count : unfiltered
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━┓
    ┃ partition                   ┃ app           ┃ samples ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━┩
    │ pretraining                 │ google-doc    │    1221 │
    │                             │ google-drive  │    1634 │
    │                             │ google-music  │     592 │
    │                             │ google-search │    1915 │
    │                             │ youtube       │    1077 │
    │                             │ __total__     │    6439 │
    ├─────────────────────────────┼───────────────┼─────────┤
    │ retraining-human-triggered  │ google-doc    │      15 │
    │                             │ google-drive  │      18 │
    │                             │ google-music  │      15 │
    │                             │ google-search │      15 │
    │                             │ youtube       │      20 │
    │                             │ __total__     │      83 │
    ├─────────────────────────────┼───────────────┼─────────┤
    │ retraining-script-triggered │ google-doc    │      30 │
    │                             │ google-drive  │      30 │
    │                             │ google-music  │      30 │
    │                             │ google-search │      30 │
    │                             │ youtube       │      30 │
    │                             │ __total__     │     150 │
    └─────────────────────────────┴───────────────┴─────────┘

    ╭───────────────╮
    │generate splits│
    ╰───────────────╯
    saving: /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19/preprocessed/imc23/train_split_0.parquet
    saving: /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19/preprocessed/imc23/train_split_1.parquet
    saving: /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19/preprocessed/imc23/train_split_2.parquet
    saving: /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19/preprocessed/imc23/train_split_3.parquet
    saving: /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19/preprocessed/imc23/train_split_4.parquet
    samples count : train_split = 0 to 4
    ┏━━━━━━━━━━━━━━━┳━━━━━━━━━┓
    ┃ app           ┃ samples ┃
    ┡━━━━━━━━━━━━━━━╇━━━━━━━━━┩
    │ google-doc    │     100 │
    │ google-drive  │     100 │
    │ google-music  │     100 │
    │ google-search │     100 │
    │ youtube       │     100 │
    ├───────────────┼─────────┤
    │ __total__     │     500 │
    └───────────────┴─────────┘

    saving: /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19/preprocessed/imc23/test_split_human.parquet
    samples count : test_split_human
    ┏━━━━━━━━━━━━━━━┳━━━━━━━━━┓
    ┃ app           ┃ samples ┃
    ┡━━━━━━━━━━━━━━━╇━━━━━━━━━┩
    │ youtube       │      20 │
    │ google-drive  │      18 │
    │ google-doc    │      15 │
    │ google-music  │      15 │
    │ google-search │      15 │
    ├───────────────┼─────────┤
    │ __total__     │      83 │
    └───────────────┴─────────┘

    saving: /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19/preprocessed/imc23/test_split_script.parquet
    samples count : test_split_script
    ┏━━━━━━━━━━━━━━━┳━━━━━━━━━┓
    ┃ app           ┃ samples ┃
    ┡━━━━━━━━━━━━━━━╇━━━━━━━━━┩
    │ google-doc    │      30 │
    │ google-drive  │      30 │
    │ google-music  │      30 │
    │ google-search │      30 │
    │ youtube       │      30 │
    ├───────────────┼─────────┤
    │ __total__     │     150 │
    └───────────────┴─────────┘
	```

!!! note "Focusing on the reports..."

	Notice the following:

	* There is a monolithic parquet file containing the original partitions. For readability, the samples count report is groupped by partition
	* There are 5 train splits (with 100 samples per class)
	* There are 2 test splits (i.e., `human` and `script`) matching the related partitions from the raw dataset.




## `mirage19`

The dataset is a collection of JSON files gathering per-flow
data from 20 Android apps.

!!! Warning "...but why not 40 apps?"
    Despite the [`mirage19` website](https://traffic.comics.unina.it/mirage/mirage-2019.html) and the [related paper](http://wpage.unina.it/antonio.montieri/pubs/MIRAGE_ICCCS_2019.pdf) mention
	the availability of 40 applications, the public version
    has only 20. With separate communication with the authors
    of the dataset, we understood that the remaining 20
    are available only upon request (altough not explicitly
    specified). As result, we considered only the 20 publicly
    available.

The dataset can be downloaded directly by `tcbench`
or can be pre-downloaded and placed into a folder.

For instance
```
downloads/
└── MIRAGE-2019_traffic_dataset_downloadable_v2.tar.gz 
```

#### Original structure

Once unpacked the dataset has
the following structure
```
downloads/
└─ MIRAGE-2019_traffic_dataset_downloadable
   ├── Mi5_38_a4_ed_18_cc_bf
   └── Nexus7_bc_ee_7b_a4_09_47
```

The subfolders contain collections of JSON files,
each representing a different experiment.
The application is encoded both in the filenames
as well as extra metadata in the JSON structure
(`flow` :material-arrow-right: `metadata` :material-arrow-right: `bf` :material-arrow-right: `label`).
    
Each JSON file is already per-flow, but they have
a fairly complicated nested structure.
For example, aggregate flow metrics are
hierarchially separated from packet time series,
which are further separated from other metadata.

#### Curation

1. Combine all JSON into a monolithic parquet file.

2. __Flatten the JSON nested structure__. 
	For instance, the nested input dictionary
    `{"layer1":{"col1":1, "col2":2}}` 
	would be flattened into a table
	with columns "layer1_col1" and "layer1_col2" 
	with the respective values "1" and "2".

3. Add a __`"background"` class__. More specifically,
    each JSON file details the Android 
	app name in the file name. But the traffic in an experiment
    can be related to a different app/service running in parallel.
	However, the dataset offers the column
    `flow_metadata_bf_label` which contains the
	Android app name that `netstat` linked to each
    network socket during an experiment.
	This implies that, by knowing the expected app
	of an experiment, one can define as "background" 
	:material-arrow-right: `flow_metadata_bf_label` != 
    expected Android app name.

4. The dataset contains raw packet bytes across multiple packets
    of a flow. We process these series to search for 
	__ASCII strings__.
    This can be usefull for extract (in a lazy way) TLS
    handshake information (e.g., SNI or certificate info).

The final parquet files has 127 columns, and most of
them comes from the original dataset itself.
They are not documented but fairly easy to understand
based on the name. Thus, __we prune some
columns in the final filtered files__.

The most important ones are

|Field|Description|
|:----|:----------|
|`packet_data_packet_dir`|The time series of the packet direction|
|`packet_data_l4_payload_bytes`|The time series of the packet size|
|`packet_data_iat`|The time series of the packet inter-arrival time|
|`flow_metadata_bf_label`|The label gathered via netstat|
|`strings`|The ASCII string recovered from the payload analysis|
|`android_name`|The app used for an experiment|
|`app`|The final label encoded as a pandas `category`|
|`row_id`|A unique row identifier|

Please refer to the [datasets schema](/datasets/metadata/#schemas) page for more details.

#### Splits

Once preprocessed, the monolithic dataset is further processed to:

* Remove ACK packets from time series.
* Remove flows with < 10 samples.
* Remove apps with < 100 samples.

From the remaining traffic we define 5 train/val/test splits with the following logic

1. Shuffle the rows.
2. Perform a 90/10 split where the 10-part is used for testing.
3. From the 90-part, do a second 90/10 to define train and validation.

The splits are NOT materialized, i.e., 
splits are a collection of row indexes
that needs to be applied on the filtered monolithic
parquet in order to obtain the data for modeling

The structure of the splits table is as follows

|Field|Description|
|:----|:----------|
|`train_indexes`|A numpy array with the `row_id` related to the train split|
|`val_indexes`| ... validation split|
|`test_indexes`| ... test split|
|`split_index`| The index of the split (0..4)|


#### Install

To install the dataset run

```
tcbench datasets install --name mirage19
```

!!! info "Output"
	```
	╭─────────────────╮
	│download & unpack│
	╰─────────────────╯
	Downloading... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.5 GB / 1.5 GB eta 0:00:00
	opening: /tmp/tmpxcdzy8tw/MIRAGE-2019_traffic_dataset_downloadable_v2.tar.gz

	╭──────────╮
	│preprocess│
	╰──────────╯
	found 1642 JSON files to load
	Converting JSONs... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1642/1642 0:00:11
	merging files...
	saving: /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/mirage19/preprocessed/mirage19.parquet

	╭────────────────────────╮
	│filter & generate splits│
	╰────────────────────────╯
	loading: /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/mirage19/preprocessed/mirage19.parquet
	samples count : unfiltered
	┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
	┃ app                         ┃ samples ┃
	┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
	│ com.waze                    │   11737 │
	│ de.motain.iliga             │   10810 │
	│ com.accuweather.android     │   10631 │
	│ com.duolingo                │    8319 │
	│ it.subito                   │    8167 │
	│ com.contextlogic.wish       │    6507 │
	│ com.spotify.music           │    6431 │
	│ com.joelapenna.foursquared  │    6399 │
	│ com.google.android.youtube  │    6346 │
	│ com.iconology.comics        │    5516 │
	│ com.facebook.katana         │    5368 │
	│ com.dropbox.android         │    4815 │
	│ com.twitter.android         │    4734 │
	│ background                  │    4439 │
	│ com.pinterest               │    4078 │
	│ com.facebook.orca           │    4018 │
	│ com.tripadvisor.tripadvisor │    3572 │
	│ air.com.hypah.io.slither    │    3088 │
	│ com.viber.voip              │    2740 │
	│ com.trello                  │    2306 │
	│ com.groupon                 │    1986 │
	├─────────────────────────────┼─────────┤
	│ __total__                   │  122007 │
	└─────────────────────────────┴─────────┘
	stats : number packets per-flow (unfiltered)
	┏━━━━━━━┳━━━━━━━━━━┓
	┃ stat  ┃    value ┃
	┡━━━━━━━╇━━━━━━━━━━┩
	│ count │ 122007.0 │
	│ mean  │    23.11 │
	│ std   │     9.73 │
	│ min   │      1.0 │
	│ 25%   │     17.0 │
	│ 50%   │     26.0 │
	│ 75%   │     32.0 │
	│ max   │     32.0 │
	└───────┴──────────┘

	filtering min_pkts=10...
	saving: /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/mirage19/preprocessed/imc23/mirage19_filtered_minpkts10.parquet
	samples count : filtered (min_pkts=10)
	┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
	┃ app                         ┃ samples ┃
	┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
	│ de.motain.iliga             │    7505 │
	│ com.waze                    │    7214 │
	│ com.duolingo                │    4583 │
	│ it.subito                   │    4299 │
	│ com.contextlogic.wish       │    3927 │
	│ com.accuweather.android     │    3737 │
	│ com.joelapenna.foursquared  │    3627 │
	│ com.spotify.music           │    3300 │
	│ com.dropbox.android         │    3189 │
	│ com.facebook.katana         │    2878 │
	│ com.iconology.comics        │    2812 │
	│ com.twitter.android         │    2805 │
	│ com.google.android.youtube  │    2728 │
	│ com.pinterest               │    2450 │
	│ com.tripadvisor.tripadvisor │    2052 │
	│ com.facebook.orca           │    1783 │
	│ com.viber.voip              │    1618 │
	│ com.trello                  │    1478 │
	│ com.groupon                 │    1174 │
	│ air.com.hypah.io.slither    │    1013 │
	├─────────────────────────────┼─────────┤
	│ __total__                   │   64172 │
	└─────────────────────────────┴─────────┘
	stats : number packets per-flow (min_pkts=10)
	┏━━━━━━━┳━━━━━━━━━┓
	┃ stat  ┃   value ┃
	┡━━━━━━━╇━━━━━━━━━┩
	│ count │ 64172.0 │
	│ mean  │   17.01 │
	│ std   │    4.43 │
	│ min   │    11.0 │
	│ 25%   │    14.0 │
	│ 50%   │    17.0 │
	│ 75%   │    19.0 │
	│ max   │    32.0 │
	└───────┴─────────┘
	saving: /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/mirage19/preprocessed/imc23/mirage19_filtered_minpkts10_splits.parquet
	┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
	┃ app                         ┃ train_samples ┃ val_samples ┃ test_samples ┃ all_samples ┃
	┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
	│ de.motain.iliga             │          6079 │         675 │          751 │        7505 │
	│ com.waze                    │          5844 │         649 │          721 │        7214 │
	│ com.duolingo                │          3712 │         413 │          458 │        4583 │
	│ it.subito                   │          3482 │         387 │          430 │        4299 │
	│ com.contextlogic.wish       │          3181 │         353 │          393 │        3927 │
	│ com.accuweather.android     │          3027 │         336 │          374 │        3737 │
	│ com.joelapenna.foursquared  │          2938 │         326 │          363 │        3627 │
	│ com.spotify.music           │          2673 │         297 │          330 │        3300 │
	│ com.dropbox.android         │          2583 │         287 │          319 │        3189 │
	│ com.facebook.katana         │          2331 │         259 │          288 │        2878 │
	│ com.iconology.comics        │          2278 │         253 │          281 │        2812 │
	│ com.twitter.android         │          2272 │         252 │          281 │        2805 │
	│ com.google.android.youtube  │          2209 │         246 │          273 │        2728 │
	│ com.pinterest               │          1984 │         221 │          245 │        2450 │
	│ com.tripadvisor.tripadvisor │          1662 │         185 │          205 │        2052 │
	│ com.facebook.orca           │          1444 │         161 │          178 │        1783 │
	│ com.viber.voip              │          1310 │         146 │          162 │        1618 │
	│ com.trello                  │          1197 │         133 │          148 │        1478 │
	│ com.groupon                 │           951 │         106 │          117 │        1174 │
	│ air.com.hypah.io.slither    │           821 │          91 │          101 │        1013 │
	├─────────────────────────────┼───────────────┼─────────────┼──────────────┼─────────────┤
	│ __total__                   │         51978 │        5776 │         6418 │       64172 │
	└─────────────────────────────┴───────────────┴─────────────┴──────────────┴─────────────┘
	```

!!! note "Focusing on the reports..."

	Notice the following:

	* The unfiltered dataset has many flows but observe (see packets stats) that those are mostly short.
	* The splits are formed so to have roughly 100 samples for validation/test.

## Install `mirage22`

The dataset is from the same authors of `mirage19` so 
the two datasets have many commonalities.

The major difference is the dataset aim: It focuses
on video meeting Android apps with experiments
annotated with respect to different interactions
the the apps (voice, chat, etc.)


#### Original structure

The dataset can be downloaded automatically 
(or can be pre-downloaded into a folder).

For instance
```
downloads/
└── MIRAGE-COVID-CCMA-2022.zip
```

Once unpacked it has the following structure
```
downloads/
└── MIRAGE-COVID-CCMA-2022
   ├── Preprocessed_pickle
   └── Raw_JSON
       ├── Discord
       ├── GotoMeeting
       ├── Meet
       ├── Messenger
       ├── Skype
       ├── Slack
       ├── Teams
       ├── Webex
       └── Zoom
```

Notice the two subfolders:

* `Raw_JSON` gathers the nested JSON files for each experiment.

* `Preprocessed_pickle` is a pickle serialization of the 
data (undocumented by the authors).

#### Curation

Same as for [`mirage19` curation](datasets/install/#curation_2)


#### Splits

Again, similar to [`mirage19` splits](datasets/install/#splits_2).

The only difference is that we apply two
filtering on flow length (at least 10 packets and at least 1000 packets).


#### Install

To install the dataset run

```
tcbench datasets install --name mirage22
```

!!! note "Output"
	```
	╭─────────────────╮
	│download & unpack│
	╰─────────────────╯
	Downloading... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.1 GB / 3.1 GB eta 0:00:00
	opening: /tmp/tmp3marsp7l/MIRAGE-COVID-CCMA-2022.zip
	opening: /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/libtcdatasets/datasets/mirage22/raw/MIRAGE-COVID-CCMA-2022/Raw_JSON/Discord.zip
	opening: /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/libtcdatasets/datasets/mirage22/raw/MIRAGE-COVID-CCMA-2022/Raw_JSON/GotoMeeting.zip
	opening: /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/libtcdatasets/datasets/mirage22/raw/MIRAGE-COVID-CCMA-2022/Raw_JSON/Meet.zip
	opening: /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/libtcdatasets/datasets/mirage22/raw/MIRAGE-COVID-CCMA-2022/Raw_JSON/Messenger.zip
	opening: /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/libtcdatasets/datasets/mirage22/raw/MIRAGE-COVID-CCMA-2022/Raw_JSON/Skype.zip
	opening: /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/libtcdatasets/datasets/mirage22/raw/MIRAGE-COVID-CCMA-2022/Raw_JSON/Slack.zip
	opening: /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/libtcdatasets/datasets/mirage22/raw/MIRAGE-COVID-CCMA-2022/Raw_JSON/Teams.zip
	opening: /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/libtcdatasets/datasets/mirage22/raw/MIRAGE-COVID-CCMA-2022/Raw_JSON/Webex.zip
	opening: /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/libtcdatasets/datasets/mirage22/raw/MIRAGE-COVID-CCMA-2022/Raw_JSON/Zoom.zip

	╭──────────╮
	│preprocess│
	╰──────────╯
	found 998 JSON files to load
	Converting JSONs... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 998/998 0:00:28
	merging files...
	saving: /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/libtcdatasets/datasets/mirage22/preprocessed/mirage22.parquet

	╭────────────────────────╮
	│filter & generate splits│
	╰────────────────────────╯
	loading: /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/mirage22/preprocessed/mirage22.parquet
	samples count : unfiltered
	┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
	┃ app                              ┃ samples ┃
	┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
	│ background                       │   18882 │
	│ com.microsoft.teams              │    6541 │
	│ com.skype.raider                 │    6203 │
	│ us.zoom.videomeetings            │    5066 │
	│ com.cisco.webex.meetings         │    4789 │
	│ com.discord                      │    4337 │
	│ com.facebook.orca                │    4321 │
	│ com.gotomeeting                  │    3695 │
	│ com.Slack                        │    2985 │
	│ com.google.android.apps.meetings │    2252 │
	├──────────────────────────────────┼─────────┤
	│ __total__                        │   59071 │
	└──────────────────────────────────┴─────────┘
	stats : number packets per-flow (unfiltered)
	┏━━━━━━━┳━━━━━━━━━━━┓
	┃ stat  ┃     value ┃
	┡━━━━━━━╇━━━━━━━━━━━┩
	│ count │   59071.0 │
	│ mean  │   3068.32 │
	│ std   │  25416.43 │
	│ min   │       1.0 │
	│ 25%   │      20.0 │
	│ 50%   │      27.0 │
	│ 75%   │      42.0 │
	│ max   │ 1665842.0 │
	└───────┴───────────┘

	filtering min_pkts=10...
	saving: /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/mirage22/preprocessed/imc23/mirage22_filtered_minpkts10.parquet
	samples count : filtered (min_pkts=10)
	┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
	┃ app                              ┃ samples ┃
	┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
	│ com.cisco.webex.meetings         │    4437 │
	│ com.skype.raider                 │    4117 │
	│ com.microsoft.teams              │    3857 │
	│ us.zoom.videomeetings            │    3587 │
	│ com.discord                      │    3387 │
	│ com.facebook.orca                │    2623 │
	│ com.gotomeeting                  │    2557 │
	│ com.google.android.apps.meetings │    1238 │
	│ com.Slack                        │     970 │
	├──────────────────────────────────┼─────────┤
	│ __total__                        │   26773 │
	└──────────────────────────────────┴─────────┘
	stats : number packets per-flow (min_pkts=10)
	┏━━━━━━━┳━━━━━━━━━━━┓
	┃ stat  ┃     value ┃
	┡━━━━━━━╇━━━━━━━━━━━┩
	│ count │   26773.0 │
	│ mean  │   6598.23 │
	│ std   │  37290.08 │
	│ min   │      11.0 │
	│ 25%   │      15.0 │
	│ 50%   │      21.0 │
	│ 75%   │     186.0 │
	│ max   │ 1665842.0 │
	└───────┴───────────┘
	saving: /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/mirage22/preprocessed/imc23/mirage22_filtered_minpkts10_splits.parquet
	┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
	┃ app                              ┃ train_samples ┃ val_samples ┃ test_samples ┃ all_samples ┃
	┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
	│ com.cisco.webex.meetings         │          3594 │         399 │          444 │        4437 │
	│ com.skype.raider                 │          3334 │         371 │          412 │        4117 │
	│ com.microsoft.teams              │          3124 │         347 │          386 │        3857 │
	│ us.zoom.videomeetings            │          2905 │         323 │          359 │        3587 │
	│ com.discord                      │          2743 │         305 │          339 │        3387 │
	│ com.facebook.orca                │          2125 │         236 │          262 │        2623 │
	│ com.gotomeeting                  │          2072 │         230 │          255 │        2557 │
	│ com.google.android.apps.meetings │          1002 │         112 │          124 │        1238 │
	│ com.Slack                        │           786 │          87 │           97 │         970 │
	├──────────────────────────────────┼───────────────┼─────────────┼──────────────┼─────────────┤
	│ __total__                        │         21685 │        2410 │         2678 │       26773 │
	└──────────────────────────────────┴───────────────┴─────────────┴──────────────┴─────────────┘

	filtering min_pkts=1000...
	saving: /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/mirage22/preprocessed/imc23/mirage22_filtered_minpkts1000.parquet
	samples count : filtered (min_pkts=1000)
	┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
	┃ app                              ┃ samples ┃
	┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
	│ com.discord                      │    2220 │
	│ us.zoom.videomeetings            │     425 │
	│ com.google.android.apps.meetings │     379 │
	│ com.microsoft.teams              │     321 │
	│ com.gotomeeting                  │     297 │
	│ com.facebook.orca                │     280 │
	│ com.cisco.webex.meetings         │     259 │
	│ com.Slack                        │     198 │
	│ com.skype.raider                 │     190 │
	├──────────────────────────────────┼─────────┤
	│ __total__                        │    4569 │
	└──────────────────────────────────┴─────────┘
	stats : number packets per-flow (min_pkts=1000)
	┏━━━━━━━┳━━━━━━━━━━━┓
	┃ stat  ┃     value ┃
	┡━━━━━━━╇━━━━━━━━━━━┩
	│ count │    4569.0 │
	│ mean  │  38321.32 │
	│ std   │   83282.0 │
	│ min   │    1001.0 │
	│ 25%   │    2863.0 │
	│ 50%   │    6303.0 │
	│ 75%   │   35392.0 │
	│ max   │ 1665842.0 │
	└───────┴───────────┘
	saving: /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/mirage22/preprocessed/imc23/mirage22_filtered_minpkts1000_splits.parquet
	┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
	┃ app                              ┃ train_samples ┃ val_samples ┃ test_samples ┃ all_samples ┃
	┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
	│ com.discord                      │          1798 │         200 │          222 │        2220 │
	│ us.zoom.videomeetings            │           344 │          39 │           42 │         425 │
	│ com.google.android.apps.meetings │           307 │          34 │           38 │         379 │
	│ com.microsoft.teams              │           260 │          29 │           32 │         321 │
	│ com.gotomeeting                  │           240 │          27 │           30 │         297 │
	│ com.facebook.orca                │           227 │          25 │           28 │         280 │
	│ com.cisco.webex.meetings         │           210 │          23 │           26 │         259 │
	│ com.Slack                        │           160 │          18 │           20 │         198 │
	│ com.skype.raider                 │           154 │          17 │           19 │         190 │
	├──────────────────────────────────┼───────────────┼─────────────┼──────────────┼─────────────┤
	│ __total__                        │          3700 │         412 │          457 │        4569 │
	└──────────────────────────────────┴───────────────┴─────────────┴──────────────┴─────────────┘
	```

!!! note "Focusing on the reports..."

	Notice the following:

	* The unfiltered dataset shows a lot of small flows, but
	this bias reduces when applying the filtering.

## `utmobilenet21`

The dataset is a collection of per-packet CSV files 
related to 17 Android apps across 
4 different *partitions* each corresponding
to a different interaction with each app.

The dataset cannot be downloaded directly
(because it is stored into a Box cloud
storage) so you need to pre-download it
and save it into a folder.

For instance
```
downloads/
└── UTMobileNet2021.zip
```

#### Original structure

Once unpacked, the datasets is organized as follows
```
downloads/
└── csvs
    ├── Action-Specific Wild Test Data
    ├── Deterministic Automated Data
    ├── Randomized Automated Data
    └── Wild Test Data
```

Each subfolders of `csvs/` corresponds to a different
partition (the folder name is informing you
about the semantic of the original measurement campaign).

Within each subfolder there is a collection of CSVs
generated running `tshark`, i.e., they are per-packet logs.

Differently from `mirage19` and `mirage22`, the only label available
is provided by CSV file names.

#### Curation

* Some of the original CSVs files have rows which
break the parsing via common utilities such as `pandas.read_csv()`.
Moreover, some columns have missing values, while others
have missing types between files (e.g., ports can be
either int of floats). So extra care is taken to properly
ingest the CSVs.

* We __Filter out packets__ which are not TCP or UDP.

* Then, packets are organized into flows using the traditional
network 5-tuple.

The final monolithic parquet files has the following columns

|Field|Description|
|:----|:----------|
|`row_id`|A unique flow id|
|`src_ip`|The source ip of the flow|
|`src_port`|The source port of the flow|
|`dst_ip`|The destination ip of the flow|
|`dst_port`|The destination port of the flow|
|`ip_proto`|The protocol of the flow (TCP or UDP)|
|`first`|Timestamp of the first packet|
|`last`|Timestamp of the last packet|
|`duration`|Duration of the flow|
|`packets`|Number of packets in the flow|
|`bytes`|Number of bytes in the flow|
|`partition`|From which folder the flow was originally stored|
|`location`|A label originally provided by the dataset (see the related paper for details)|
|`fname`|The original filename where the packets of the flow come from |
|`app`|The final label of the flow, encoded as pandas `category`|
|`pkts_size`|The numpy array for the packet size time series|
|`pkts_dir`|The numpy array for the packet diretion time series|
|`timetofirst`|The numpy array for the delta between the each packet timestamp the first packet of the flow|

#### Splits

Once preprocessed, the monolithic dataset is further processed to:

* Remove flows with < 10 samples
* Remove apps with < 100 samples

From the remaining traffic we define 5 train/val/test splits with the following logic

1. Shuffle the rows
2. Perform a 90/10 split where the 10-part is used for testing
3. From the 90-part, do a second 90/10 to define train and validation

The splits are NOT materialized, i.e., 
splits are a collection of row indexes
that needs to be applied on the filtered monolithic
parquet in order to obtain the data for modeling

The structure of the split table is

|Field|Description|
|:----|:----------|
|`train_indexes`|A numpy array with the `row_id` related to the train split|
|`val_indexes`| ... validation split|
|`test_indexes`| ... test split|
|`split_index`| The index of the split (0..4)|

#### Install

To install the dataset run
(assuming data was pre-downloaded under `/downloads`)

```
tcbench datasets install \
	--name utmobilenet21 \
	--input-folder downloads/
```

!!! info "Output"
	```
	╭──────╮
	│unpack│
	╰──────╯
	opening: downloads/UTMobileNet2021.zip

	╭──────────╮
	│preprocess│
	╰──────────╯
	processing: /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/libtcdatasets/datasets/utmobilenet21/raw/Action-Specific Wild Test Data
	found 43 files
	Converting CSVs... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 43/43 0:01:15
	stage1 completed
	stage2 completed
	stage3 completed
	stage4 completed
	saving: /tmp/processing-utmobilenet21/action-specific_wild_test_data.parquet

	processing: /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/libtcdatasets/datasets/utmobilenet21/raw/Wild Test Data
	found 14 files
	Converting CSVs... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 14/14 0:03:12
	stage1 completed
	stage2 completed
	stage3 completed
	stage4 completed
	saving: /tmp/processing-utmobilenet21/wild_test_data.parquet

	processing: /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/libtcdatasets/datasets/utmobilenet21/raw/Randomized Automated Data
	found 288 files
	Converting CSVs... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 288/288 0:01:35
	stage1 completed
	stage2 completed
	stage3 completed
	stage4 completed
	saving: /tmp/processing-utmobilenet21/randomized_automated_data.parquet

	processing: /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/libtcdatasets/datasets/utmobilenet21/raw/Deterministic Automated Data
	found 3438 files
	Converting CSVs... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3438/3438 0:08:26
	stage1 completed
	stage2 completed
	stage3 completed
	stage4 completed
	saving: /tmp/processing-utmobilenet21/deterministic_automated_data.parquet
	merging all partitions
	saving: /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/libtcdatasets/datasets/utmobilenet21/preprocessed/utmobilenet21.parquet

	╭────────────────────────╮
	│filter & generate splits│
	╰────────────────────────╯
	loading: /opt/anaconda/anaconda3/envs/tcbench-johndoe/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/utmobilenet21/preprocessed/utmobilenet21.parquet
	samples count : unfiltered
	┏━━━━━━━━━━━━━━┳━━━━━━━━━┓
	┃ app          ┃ samples ┃
	┡━━━━━━━━━━━━━━╇━━━━━━━━━┩
	│ youtube      │    4716 │
	│ reddit       │    3622 │
	│ google-maps  │    3475 │
	│ netflix      │    1804 │
	│ pinterest    │    1702 │
	│ dropbox      │    1609 │
	│ instagram    │    1426 │
	│ gmail        │     848 │
	│ google-drive │     709 │
	│ messenger    │     690 │
	│ hangout      │     483 │
	│ facebook     │     364 │
	│ twitter      │     308 │
	│ hulu         │     294 │
	│ spotify      │     252 │
	│ pandora      │      70 │
	│ skype        │      57 │
	├──────────────┼─────────┤
	│ __total__    │   22429 │
	└──────────────┴─────────┘
	stats : number packets per-flow (unfiltered)
	┏━━━━━━━┳━━━━━━━━━━━┓
	┃ stat  ┃     value ┃
	┡━━━━━━━╇━━━━━━━━━━━┩
	│ count │   22429.0 │
	│ mean  │    716.33 │
	│ std   │  22271.93 │
	│ min   │       1.0 │
	│ 25%   │       2.0 │
	│ 50%   │       2.0 │
	│ 75%   │      15.0 │
	│ max   │ 1973657.0 │
	└───────┴───────────┘

	saving: /opt/anaconda/anaconda3/envs/tcbench-johndoe/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/utmobilenet21/preprocessed/imc23/utmobilenet21_filtered_minpkts10.parquet
	samples count : filtered (min_pkts=10)
	┏━━━━━━━━━━━━━━┳━━━━━━━━━┓
	┃ app          ┃ samples ┃
	┡━━━━━━━━━━━━━━╇━━━━━━━━━┩
	│ youtube      │    2153 │
	│ google-maps  │    1391 │
	│ reddit       │     654 │
	│ netflix      │     317 │
	│ pinterest    │     312 │
	│ dropbox      │     211 │
	│ instagram    │     205 │
	│ hangout      │     176 │
	│ hulu         │     162 │
	│ google-drive │     104 │
	├──────────────┼─────────┤
	│ __total__    │    5685 │
	└──────────────┴─────────┘
	stats : number packets per-flow (min_pkts=10)
	┏━━━━━━━┳━━━━━━━━━━━┓
	┃ stat  ┃     value ┃
	┡━━━━━━━╇━━━━━━━━━━━┩
	│ count │    5685.0 │
	│ mean  │   2740.55 │
	│ std   │  44152.43 │
	│ min   │      11.0 │
	│ 25%   │      25.0 │
	│ 50%   │      44.0 │
	│ 75%   │     156.0 │
	│ max   │ 1973657.0 │
	└───────┴───────────┘
	saving: /opt/anaconda/anaconda3/envs/tcbench-johndoe/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/utmobilenet21/preprocessed/imc23/utmobilenet21_filtered_minpkts10_splits.parquet
	┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
	┃ app          ┃ train_samples ┃ val_samples ┃ test_samples ┃ all_samples ┃
	┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
	│ youtube      │          1743 │         194 │          216 │        2153 │
	│ google-maps  │          1127 │         125 │          139 │        1391 │
	│ reddit       │           530 │          59 │           65 │         654 │
	│ netflix      │           256 │          29 │           32 │         317 │
	│ pinterest    │           253 │          28 │           31 │         312 │
	│ dropbox      │           171 │          19 │           21 │         211 │
	│ instagram    │           166 │          18 │           21 │         205 │
	│ hangout      │           142 │          16 │           18 │         176 │
	│ hulu         │           131 │          15 │           16 │         162 │
	│ google-drive │            85 │           9 │           10 │         104 │
	├──────────────┼───────────────┼─────────────┼──────────────┼─────────────┤
	│ __total__    │          4604 │         512 │          569 │        5685 │
	└──────────────┴───────────────┴─────────────┴──────────────┴─────────────┘
	```

!!! note "Focusing on the reports..."

	Notice the following:

	* From the packet stats of the original (unfiltered)
	dataset we can see there are a lot of small flows.
	Those are removed when considering a minimum
	flow length of 10.

	* On top of this filtering we also remove
    apps with less than 100 flows, i.e., __only
    10 of the original 17 apps can be used for modeling__.
