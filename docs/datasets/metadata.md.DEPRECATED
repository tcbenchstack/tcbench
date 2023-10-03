The `tcbench` CLI can show 4 types of meta-data:

* [__:material-information-box: Static information__](#static-information): A collection of 
URL links with datasets documentation
and folders path related the installation.

* [__:material-file-tree-outline: List of curated files__](#list-of-curated-files): An organized
view of the files generated during installation.

* [__:material-table: Schemas__](#schemas): A formatted view of
the schemas of installed files.

* [__:octicons-number-24: Samples count report__](#samples-count-report): A per-app breakdown
of the number of samples.

### :material-information-box: Static information

The static information corresponds
to the information displayed in the 
[datasets properties](/datasets/install/#table-datasets-properties)
shown in the installation page.

To show it in the console run

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

The example above corresponds to the case when
no dataset is installed.

After a dataset is installed, the remaining
properties are filled. Specifically,
as suggested by the icon, the last
3 properties of each dataset 
correspond to folders generated
via the curation:

* `"installed"` is the path
where the raw data of the dataset is
unpacked.

* `"preprocessed"` is the
path where the preprocessed data
is stored, i.e., the monolithic
per-flow parquet files (with no
filtering applied).

* `"data splits"` is the folder
where filtered data and data splits
are stored, i.e., the data used
for modeling.

The `datasets info` sub-command supports
the option `--name` to filter out
information for just one dataset.

For instance, after installing
`ucdavis-icdm19`, its
information are:

```
tcbench datasets info --name ucdavis-icdm19
```

!!! into "Output"
	```
	Datasets
	└── ucdavis-icdm19
		└──  🚩 classes:       5
			 🔗 paper_url:     https://arxiv.org/pdf/1812.09761.pdf
			 🔗 website:       https://github.com/shrezaei/Semi-supervised-Learning-QUIC-
			 🔗 data:          https://drive.google.com/drive/folders/1Pvev0hJ82usPh6dWDlz7Lv8L6h3JpWhE
			 📁 installed:     /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19/raw
			 📁 preprocessed:  /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19/preprocessed
			 📁 data splits:   /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19/preprocessed/imc23
	```

### :material-file-tree-outline: List of curated files

As reported by `datasets info`, 
both datasets raw data and curated parquet files 
are stored into a subfolder of the python environment.

Specifically, the folder is structure is as follows:
```
/datasets
  └── <dataset-name> 
  	    └── /raw
  	    └── /preprocessed
  	         └── /imc23
```

where

* `/raw` contains the raw data of the datasets.
* `/preprocessed` contains the monolithic parquet files.
* `/imc23` contains the filtererd monolithic parquet files
and the splits generated for the submission.

One can better inspect this structure via the `datasets lsparquet` sub-command

```
tcbench datasets lsparquet
```

!!! info "Output"
	```
	Datasets
	├── ucdavis-icdm19
	│   └── 📁 preprocessed/
	│       ├── ucdavis-icdm19.parquet
	│       └── 📁 imc23/
	│           ├── test_split_human.parquet
	│           ├── test_split_script.parquet
	│           ├── train_split_0.parquet
	│           ├── train_split_1.parquet
	│           ├── train_split_2.parquet
	│           ├── train_split_3.parquet
	│           └── train_split_4.parquet
	├── mirage19
	│   └── 📁 preprocessed/
	│       ├── mirage19.parquet
	│       └── 📁 imc23/
	│           ├── mirage19_filtered_minpkts10.parquet
	│           └── mirage19_filtered_minpkts10_splits.parquet
	├── mirage22
	│   └── 📁 preprocessed/
	│       ├── mirage22.parquet
	│       └── 📁 imc23/
	│           ├── mirage22_filtered_minpkts10.parquet
	│           ├── mirage22_filtered_minpkts1000.parquet
	│           ├── mirage22_filtered_minpkts1000_splits.parquet
	│           └── mirage22_filtered_minpkts10_splits.parquet
	└── utmobilenet21
		└── 📁 preprocessed/
			├── utmobilenet21.parquet
			└── 📁 imc23/
				├── utmobilenet21_filtered_minpkts10.parquet
				└── utmobilenet21_filtered_minpkts10_splits.parquet
	```

While all datasets have a file `<dataset-name>.parquet` 
which corresponds to the monolithic version of the
raw data, the content of the `/imc23` folder
is slightly different between datasets

* For `ucdavis-icdm19` split are "materialized".
This means that the files `train_split_[0-4].parquet`
contains the data to use for training (the actual
train/val split is operated at runtime) while
`test_split_human.parquet` and `text_split_script.parquet`
are predefined test split already available
in the raw dataset.

* For all other datasets, the files `xyz_minpkts<N>.parquet`
contains a filtered version of the monolithic data
(see [install page](/datasets/install) for more details on the filtering)
and the related `xyz_minpkts<N>_split.parquet` contains
the index of the rows to use for train/val/test splits.

The [tutorial about load and explore data]() offers more
details about how to handle those differences.

### :material-table: Schemas

Via the `datasets schema` sub-command is possible
to visualize the schema of the curated parquet files.

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

While for `ucdavis-icdm19` the three schema types
are the same, for the other datasets there are differences.

Below we report all schemas for all datasets.
The section expanded suggest the datasets to be used,
while ==highlighted rows== suggest which fields
are more useful for modeling.

##### ucdavis-icdm19

```
tcbench datasets schema --name ucdavis-icdm19 --type unfiltered
```

!!! note "Output"
	```hl_lines="5 12 13 14"
	┏━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
	┃ Field       ┃ Dtype    ┃ Description                                         ┃
	┡━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
	│ row_id      │ int      │ Unique row id                                       │
	│ app         │ category │ Label of the flow                                   │
	│ flow_id     │ str      │ Original filename                                   │
	│ partition   │ str      │ Partition related to the flow                       │
	│ num_pkts    │ int      │ Number of packets in the flow                       │
	│ duration    │ float    │ Duration of the flow                                │
	│ bytes       │ int      │ Number of bytes of the flow                         │
	│ unixtime    │ str      │ Absolute time of each packet                        │
	│ timetofirst │ np.array │ Delta between a packet the first packet of the flow │
	│ pkts_size   │ np.array │ Packet size time series                             │
	│ pkts_dir    │ np.array │ Packet direction time series                        │
	│ pkts_iat    │ np.array │ Packet inter-arrival time series                    │
	└─────────────┴──────────┴─────────────────────────────────────────────────────┘
	```

??? note "tcbench datasets schema --name ucdavis-icdm19 --type filtered"
	```hl_lines="5 12 13 14"
	┏━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
	┃ Field       ┃ Dtype    ┃ Description                                         ┃
	┡━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
	│ row_id      │ int      │ Unique row id                                       │
	│ app         │ category │ Label of the flow                                   │
	│ flow_id     │ str      │ Original filename                                   │
	│ partition   │ str      │ Partition related to the flow                       │
	│ num_pkts    │ int      │ Number of packets in the flow                       │
	│ duration    │ float    │ Duration of the flow                                │
	│ bytes       │ int      │ Number of bytes of the flow                         │
	│ unixtime    │ str      │ Absolute time of each packet                        │
	│ timetofirst │ np.array │ Delta between a packet the first packet of the flow │
	│ pkts_size   │ np.array │ Packet size time series                             │
	│ pkts_dir    │ np.array │ Packet direction time series                        │
	│ pkts_iat    │ np.array │ Packet inter-arrival time series                    │
	└─────────────┴──────────┴─────────────────────────────────────────────────────┘
	```

??? note "tcbench datasets schema --name ucdavis-icdm19 --type splits"
	```hl_lines="5 12 13 14"
	┏━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
	┃ Field       ┃ Dtype    ┃ Description                                         ┃
	┡━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
	│ row_id      │ int      │ Unique row id                                       │
	│ app         │ category │ Label of the flow                                   │
	│ flow_id     │ str      │ Original filename                                   │
	│ partition   │ str      │ Partition related to the flow                       │
	│ num_pkts    │ int      │ Number of packets in the flow                       │
	│ duration    │ float    │ Duration of the flow                                │
	│ bytes       │ int      │ Number of bytes of the flow                         │
	│ unixtime    │ str      │ Absolute time of each packet                        │
	│ timetofirst │ np.array │ Delta between a packet the first packet of the flow │
	│ pkts_size   │ np.array │ Packet size time series                             │
	│ pkts_dir    │ np.array │ Packet direction time series                        │
	│ pkts_iat    │ np.array │ Packet inter-arrival time series                    │
	└─────────────┴──────────┴─────────────────────────────────────────────────────┘
	```

##### `mirage19`

??? note "tcbench datasets schema --name mirage19"
	```
	┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
	┃ Field                                                     ┃ Dtype    ┃ Description                                                ┃
	┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
	│ row_id                                                    │ int      │ Unique flow id                                             │
	│ conn_id                                                   │ str      │ Flow 5-tuple                                               │
	│ packet_data_src_port                                      │ np.array │ Time series of the source ports                            │
	│ packet_data_dst_port                                      │ np.array │ Time series of the destination ports                       │
	│ packet_data_packet_dir                                    │ np.array │ Time series of pkts direction (0 or 1)                     │
	│ packet_data_l4_payload_bytes                              │ np.array │ Time series of payload pkts size                           │
	│ packet_data_iat                                           │ np.array │ Time series of pkts inter arrival times                    │
	│ packet_data_tcp_win_size                                  │ np.array │ Time series of TCP window size                             │
	│ packet_data_l4_raw_payload                                │ np.array │ List of list with each packet payload                      │
	│ flow_features_packet_length_biflow_min                    │ float    │ Bidirectional min frame (i.e., pkt with headers) size      │
	│ flow_features_packet_length_biflow_max                    │ float    │ Bidirectional max frame size                               │
	│ flow_features_packet_length_biflow_mean                   │ float    │ Bidirectional mean frame size                              │
	│ flow_features_packet_length_biflow_std                    │ float    │ Bidirectional std frame size                               │
	│ flow_features_packet_length_biflow_var                    │ float    │ Bidirectional variance frame size                          │
	│ flow_features_packet_length_biflow_mad                    │ float    │ Bidirectional median absolute deviation frame size         │
	│ flow_features_packet_length_biflow_skew                   │ float    │ Bidirection skew frame size                                │
	│ flow_features_packet_length_biflow_kurtosis               │ float    │ Bidirectional kurtosi frame size                           │
	│ flow_features_packet_length_biflow_10_percentile          │ float    │ Bidirection 10%-ile of frame size                          │
	│ flow_features_packet_length_biflow_20_percentile          │ float    │ Bidirection 20%-ile of frame size                          │
	│ flow_features_packet_length_biflow_30_percentile          │ float    │ Bidirection 30%-ile of frame size                          │
	│ flow_features_packet_length_biflow_40_percentile          │ float    │ Bidirection 40%-ile of frame size                          │
	│ flow_features_packet_length_biflow_50_percentile          │ float    │ Bidirection 50%-ile of frame size                          │
	│ flow_features_packet_length_biflow_60_percentile          │ float    │ Bidirection 60%-le of frame size                           │
	│ flow_features_packet_length_biflow_70_percentile          │ float    │ Bidirection 70%-ile of frame size                          │
	│ flow_features_packet_length_biflow_80_percentile          │ float    │ Bidirection 80%-ile of frame size                          │
	│ flow_features_packet_length_biflow_90_percentile          │ float    │ Bidirection 90%-ile of frame size                          │
	│ flow_features_packet_length_upstream_flow_min             │ float    │ Upstream min frame (i.e., pkt with headers) size           │
	│ flow_features_packet_length_upstream_flow_max             │ float    │ Upstream max frame size                                    │
	│ flow_features_packet_length_upstream_flow_mean            │ float    │ Upstream mean frame size                                   │
	│ flow_features_packet_length_upstream_flow_std             │ float    │ Upstream std frame size                                    │
	│ flow_features_packet_length_upstream_flow_var             │ float    │ Upstream variance frame size                               │
	│ flow_features_packet_length_upstream_flow_mad             │ float    │ Upstream median absolute deviation frame size              │
	│ flow_features_packet_length_upstream_flow_skew            │ float    │ Upstream skew frame size                                   │
	│ flow_features_packet_length_upstream_flow_kurtosis        │ float    │ Upstream kurtosi frame size                                │
	│ flow_features_packet_length_upstream_flow_10_percentile   │ float    │ Upstream 10%-ile frame size                                │
	│ flow_features_packet_length_upstream_flow_20_percentile   │ float    │ Upstream 20%-ile frame size                                │
	│ flow_features_packet_length_upstream_flow_30_percentile   │ float    │ Upstream 30%-ile frame size                                │
	│ flow_features_packet_length_upstream_flow_40_percentile   │ float    │ Upstream 40%-ile frame size                                │
	│ flow_features_packet_length_upstream_flow_50_percentile   │ float    │ Upstream 50%-ile frame size                                │
	│ flow_features_packet_length_upstream_flow_60_percentile   │ float    │ Upstream 60%-ile frame size                                │
	│ flow_features_packet_length_upstream_flow_70_percentile   │ float    │ Upstream 70%-ile frame size                                │
	│ flow_features_packet_length_upstream_flow_80_percentile   │ float    │ Upstream 80%-ile frame size                                │
	│ flow_features_packet_length_upstream_flow_90_percentile   │ float    │ Upstream 90%-ile frame size                                │
	│ flow_features_packet_length_downstream_flow_min           │ float    │ Downstream min frame (i.e., pkt with headers) size         │
	│ flow_features_packet_length_downstream_flow_max           │ float    │ Downstream max frame size                                  │
	│ flow_features_packet_length_downstream_flow_mean          │ float    │ Downstream mean frame size                                 │
	│ flow_features_packet_length_downstream_flow_std           │ float    │ Downstream std frame size                                  │
	│ flow_features_packet_length_downstream_flow_var           │ float    │ Downstream variance frame size                             │
	│ flow_features_packet_length_downstream_flow_mad           │ float    │ Downstream max frame size                                  │
	│ flow_features_packet_length_downstream_flow_skew          │ float    │ Downstream skew frame size                                 │
	│ flow_features_packet_length_downstream_flow_kurtosis      │ float    │ Downstream kurtosi frame size                              │
	│ flow_features_packet_length_downstream_flow_10_percentile │ float    │ Downstream 10%-ile frame size                              │
	│ flow_features_packet_length_downstream_flow_20_percentile │ float    │ Downstream 20%-ile frame size                              │
	│ flow_features_packet_length_downstream_flow_30_percentile │ float    │ Downstream 30%-ile frame size                              │
	│ flow_features_packet_length_downstream_flow_40_percentile │ float    │ Downstream 40%-ile frame size                              │
	│ flow_features_packet_length_downstream_flow_50_percentile │ float    │ Downstream 50%-ile frame size                              │
	│ flow_features_packet_length_downstream_flow_60_percentile │ float    │ Downstream 60%-ile frame size                              │
	│ flow_features_packet_length_downstream_flow_70_percentile │ float    │ Downstream 70%-ile frame size                              │
	│ flow_features_packet_length_downstream_flow_80_percentile │ float    │ Downstream 80%-ile frame size                              │
	│ flow_features_packet_length_downstream_flow_90_percentile │ float    │ Downstream 90%-ile frame size                              │
	│ flow_features_iat_biflow_min                              │ float    │ Bidirectional min inter arrival time                       │
	│ flow_features_iat_biflow_max                              │ float    │ Bidirectional max inter arrival time                       │
	│ flow_features_iat_biflow_mean                             │ float    │ Bidirectional mean inter arrival time                      │
	│ flow_features_iat_biflow_std                              │ float    │ Bidirectional std inter arrival time                       │
	│ flow_features_iat_biflow_var                              │ float    │ Bidirectional variance inter arrival time                  │
	│ flow_features_iat_biflow_mad                              │ float    │ Bidirectional median absolute deviation inter arrival time │
	│ flow_features_iat_biflow_skew                             │ float    │ Bidirectional skew inter arrival time                      │
	│ flow_features_iat_biflow_kurtosis                         │ float    │ Bidirectional kurtosi inter arrival time                   │
	│ flow_features_iat_biflow_10_percentile                    │ float    │ Bidirectional 10%-tile inter arrival time                  │
	│ flow_features_iat_biflow_20_percentile                    │ float    │ Bidirectional 20%-tile inter arrival time                  │
	│ flow_features_iat_biflow_30_percentile                    │ float    │ Bidirectional 30%-tile inter arrival time                  │
	│ flow_features_iat_biflow_40_percentile                    │ float    │ Bidirectional 40%-tile inter arrival time                  │
	│ flow_features_iat_biflow_50_percentile                    │ float    │ Bidirectional 50%-tile inter arrival time                  │
	│ flow_features_iat_biflow_60_percentile                    │ float    │ Bidirectional 60%-tile inter arrival time                  │
	│ flow_features_iat_biflow_70_percentile                    │ float    │ Bidirectional 70%-tile inter arrival time                  │
	│ flow_features_iat_biflow_80_percentile                    │ float    │ Bidirectional 80%-tile inter arrival time                  │
	│ flow_features_iat_biflow_90_percentile                    │ float    │ Bidirectional 90%-tile inter arrival time                  │
	│ flow_features_iat_upstream_flow_min                       │ float    │ Upstream min inter arrival time                            │
	│ flow_features_iat_upstream_flow_max                       │ float    │ Upstream max inter arrival time                            │
	│ flow_features_iat_upstream_flow_mean                      │ float    │ Upstream avg inter arrival time                            │
	│ flow_features_iat_upstream_flow_std                       │ float    │ Upstream std inter arrival time                            │
	│ flow_features_iat_upstream_flow_var                       │ float    │ Upstream variance inter arrival time                       │
	│ flow_features_iat_upstream_flow_mad                       │ float    │ Upstream median absolute deviation inter arrival time      │
	│ flow_features_iat_upstream_flow_skew                      │ float    │ Upstream skew inter arrival time                           │
	│ flow_features_iat_upstream_flow_kurtosis                  │ float    │ Upstream kurtosi inter arrival time                        │
	│ flow_features_iat_upstream_flow_10_percentile             │ float    │ Upstream 10%-ile inter arrival time                        │
	│ flow_features_iat_upstream_flow_20_percentile             │ float    │ Upstream 20%-ile inter arrival time                        │
	│ flow_features_iat_upstream_flow_30_percentile             │ float    │ Upstream 30%-ile inter arrival time                        │
	│ flow_features_iat_upstream_flow_40_percentile             │ float    │ Upstream 40%-ile inter arrival time                        │
	│ flow_features_iat_upstream_flow_50_percentile             │ float    │ Upstream 50%-ile inter arrival time                        │
	│ flow_features_iat_upstream_flow_60_percentile             │ float    │ Upstream 60%-ile inter arrival time                        │
	│ flow_features_iat_upstream_flow_70_percentile             │ float    │ Upstream 70%-ile inter arrival time                        │
	│ flow_features_iat_upstream_flow_80_percentile             │ float    │ Upstream 80%-ile inter arrival time                        │
	│ flow_features_iat_upstream_flow_90_percentile             │ float    │ Upstream 90%-ile inter arrival time                        │
	│ flow_features_iat_downstream_flow_min                     │ float    │ Downstream min inter arrival time                          │
	│ flow_features_iat_downstream_flow_max                     │ float    │ Downstream max inter arrival time                          │
	│ flow_features_iat_downstream_flow_mean                    │ float    │ Downstream mean inter arrival time                         │
	│ flow_features_iat_downstream_flow_std                     │ float    │ Downstream std inter arrival time                          │
	│ flow_features_iat_downstream_flow_var                     │ float    │ Downstream variance inter arrival time                     │
	│ flow_features_iat_downstream_flow_mad                     │ float    │ Downstream median absolute deviation inter arrival time    │
	│ flow_features_iat_downstream_flow_skew                    │ float    │ Downstream skew inter arrival time                         │
	│ flow_features_iat_downstream_flow_kurtosis                │ float    │ Downstream kurtosi inter arrival time                      │
	│ flow_features_iat_downstream_flow_10_percentile           │ float    │ Downstream 10%-ile inter arrival time                      │
	│ flow_features_iat_downstream_flow_20_percentile           │ float    │ Downstream 20%-ile inter arrival time                      │
	│ flow_features_iat_downstream_flow_30_percentile           │ float    │ Downstream 30%-ile inter arrival time                      │
	│ flow_features_iat_downstream_flow_40_percentile           │ float    │ Downstream 40%-ile inter arrival time                      │
	│ flow_features_iat_downstream_flow_50_percentile           │ float    │ Downstream 50%-ile inter arrival time                      │
	│ flow_features_iat_downstream_flow_60_percentile           │ float    │ Downstream 60%-ile inter arrival time                      │
	│ flow_features_iat_downstream_flow_70_percentile           │ float    │ Downstream 70%-ile inter arrival time                      │
	│ flow_features_iat_downstream_flow_80_percentile           │ float    │ Downstream 80%-ile inter arrival time                      │
	│ flow_features_iat_downstream_flow_90_percentile           │ float    │ Downstream 90%-ile inter arrival time                      │
	│ flow_metadata_bf_label                                    │ str      │ original mirage label                                      │
	│ flow_metadata_bf_labeling_type                            │ str      │ exact=via netstat; most-common=via experiment              │
	│ flow_metadata_bf_num_packets                              │ float    │ Bidirectional number of pkts                               │
	│ flow_metadata_bf_ip_packet_bytes                          │ float    │ Bidirectional bytes (including headers)                    │
	│ flow_metadata_bf_l4_payload_bytes                         │ float    │ Bidirectional payload bytes                                │
	│ flow_metadata_bf_duration                                 │ float    │ Bidirectional duration                                     │
	│ flow_metadata_uf_num_packets                              │ float    │ Upload number of pkts                                      │
	│ flow_metadata_uf_ip_packet_bytes                          │ float    │ Upload bytes (including headers)                           │
	│ flow_metadata_uf_l4_payload_bytes                         │ float    │ Upload payload bytes                                       │
	│ flow_metadata_uf_duration                                 │ float    │ Upload duration                                            │
	│ flow_metadata_df_num_packets                              │ float    │ Download number of packets                                 │
	│ flow_metadata_df_ip_packet_bytes                          │ float    │ Download bytes (including headers)                         │
	│ flow_metadata_df_l4_payload_bytes                         │ float    │ Download payload bytes                                     │
	│ flow_metadata_df_duration                                 │ float    │ Download duration                                          │
	│ strings                                                   │ list     │ ASCII string extracted from payload                        │
	│ android_name                                              │ str      │ app name (based on filename)                               │
	│ device_name                                               │ str      │ device name (based on filename)                            │
	│ app                                                       │ category │ label (background|android app)                             │
	│ src_ip                                                    │ str      │ Source IP                                                  │
	│ src_port                                                  │ str      │ Source port                                                │
	│ dst_ip                                                    │ str      │ Destination IP                                             │
	│ dst_port                                                  │ str      │ Destination port                                           │
	│ proto                                                     │ str      │ L4 protocol                                                │
	│ packets                                                   │ int      │ Number of (bidirectional) packets                          │
	└───────────────────────────────────────────────────────────┴──────────┴────────────────────────────────────────────────────────────┘
	```

!!! note "tcbench datasets schema --name mirage19 --type filtered"
	```hl_lines="14 21 22 23"
	┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
	┃ Field                             ┃ Dtype    ┃ Description                                                          ┃
	┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
	│ row_id                            │ int      │ Unique flow id                                                       │
	│ conn_id                           │ str      │ Flow 5-tuple                                                         │
	│ packet_data_l4_raw_payload        │ np.array │ List of list with each packet payload                                │
	│ flow_metadata_bf_label            │ str      │ original mirage label                                                │
	│ flow_metadata_bf_labeling_type    │ str      │ exact=via netstat; most-common=via experiment                        │
	│ flow_metadata_bf_l4_payload_bytes │ float    │ Bidirectional payload bytes                                          │
	│ flow_metadata_bf_duration         │ float    │ Bidirectional duration                                               │
	│ strings                           │ list     │ ASCII string extracted from payload                                  │
	│ android_name                      │ str      │ app name (based on filename)                                         │
	│ device_name                       │ str      │ device name (based on filename)                                      │
	│ app                               │ category │ label (background|android app)                                       │
	│ src_ip                            │ str      │ Source IP                                                            │
	│ src_port                          │ str      │ Source port                                                          │
	│ dst_ip                            │ str      │ Destination IP                                                       │
	│ dst_port                          │ str      │ Destination port                                                     │
	│ proto                             │ str      │ L4 protocol                                                          │
	│ packets                           │ int      │ Number of (bidirectional) packets                                    │
	│ pkts_size                         │ str      │ Packet size time series                                              │
	│ pkts_dir                          │ str      │ Packet diretion time series                                          │
	│ timetofirst                       │ str      │ Delta between the each packet timestamp the first packet of the flow │
	└───────────────────────────────────┴──────────┴──────────────────────────────────────────────────────────────────────┘
	```

!!! note "tcbench datasets schema --name mirage19 --type splits"
	```
	┏━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
	┃ Field         ┃ Dtype    ┃ Description                  ┃
	┡━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
	│ train_indexes │ np.array │ row_id of training samples   │
	│ val_indexes   │ np.array │ row_id of validation samples │
	│ test_indexes  │ np.array │ row_id of test samples       │
	│ split_index   │ int      │ Split id                     │
	└───────────────┴──────────┴──────────────────────────────┘
	```

##### `mirage22`

??? note "tcbench datasets schema --name mirage22 --type unfiltered"
	```
	┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
	┃ Field                                                     ┃ Dtype    ┃ Description                                                ┃
	┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
	│ row_id                                                    │ int      │ Unique flow id                                             │
	│ conn_id                                                   │ str      │ Flow 5-tuple                                               │
	│ packet_data_timestamp                                     │ np.array │ Time series of packet unixtime                             │
	│ packet_data_src_port                                      │ np.array │ Time series of the source ports                            │
	│ packet_data_dst_port                                      │ np.array │ Time series of the destination ports                       │
	│ packet_data_packet_dir                                    │ np.array │ Time series of pkts direction (0 or 1)                     │
	│ packet_data_ip_packet_bytes                               │ np.array │ Time series pkts bytes (as from IP len field)              │
	│ packet_data_ip_header_bytes                               │ np.array │ Time series of IP header bytes                             │
	│ packet_data_l4_payload_bytes                              │ np.array │ Time series of payload pkts size                           │
	│ packet_data_l4_header_bytes                               │ np.array │ Time series of L4 header bytes                             │
	│ packet_data_iat                                           │ np.array │ Time series of pkts inter arrival times                    │
	│ packet_data_tcp_win_size                                  │ np.array │ Time series of TCP window size                             │
	│ packet_data_tcp_flags                                     │ np.array │ Time series of TCP flags                                   │
	│ packet_data_l4_raw_payload                                │ np.array │ List of list with each packet payload                      │
	│ packet_data_is_clear                                      │ np.array │ n.a.                                                       │
	│ packet_data_heuristic                                     │ str      │ n.a.                                                       │
	│ packet_data_annotations                                   │ str      │ n.a.                                                       │
	│ flow_features_packet_length_biflow_min                    │ float    │ Bidirectional min frame (i.e., pkt with headers) size      │
	│ flow_features_packet_length_biflow_max                    │ float    │ Bidirectional max frame size                               │
	│ flow_features_packet_length_biflow_mean                   │ float    │ Bidirectional mean frame size                              │
	│ flow_features_packet_length_biflow_std                    │ float    │ Bidirectional std frame size                               │
	│ flow_features_packet_length_biflow_var                    │ float    │ Bidirectional variance frame size                          │
	│ flow_features_packet_length_biflow_mad                    │ float    │ Bidirectional median absolute deviation frame size         │
	│ flow_features_packet_length_biflow_skew                   │ float    │ Bidirection skew frame size                                │
	│ flow_features_packet_length_biflow_kurtosis               │ float    │ Bidirectional kurtosi frame size                           │
	│ flow_features_packet_length_biflow_10_percentile          │ float    │ Bidirection 10%-ile of frame size                          │
	│ flow_features_packet_length_biflow_20_percentile          │ float    │ Bidirection 20%-ile of frame size                          │
	│ flow_features_packet_length_biflow_30_percentile          │ float    │ Bidirection 30%-ile of frame size                          │
	│ flow_features_packet_length_biflow_40_percentile          │ float    │ Bidirection 40%-ile of frame size                          │
	│ flow_features_packet_length_biflow_50_percentile          │ float    │ Bidirection 50%-ile of frame size                          │
	│ flow_features_packet_length_biflow_60_percentile          │ float    │ Bidirection 60%-le of frame size                           │
	│ flow_features_packet_length_biflow_70_percentile          │ float    │ Bidirection 70%-ile of frame size                          │
	│ flow_features_packet_length_biflow_80_percentile          │ float    │ Bidirection 80%-ile of frame size                          │
	│ flow_features_packet_length_biflow_90_percentile          │ float    │ Bidirection 90%-ile of frame size                          │
	│ flow_features_packet_length_upstream_flow_min             │ float    │ Upstream min frame (i.e., pkt with headers) size           │
	│ flow_features_packet_length_upstream_flow_max             │ float    │ Upstream max frame size                                    │
	│ flow_features_packet_length_upstream_flow_mean            │ float    │ Upstream mean frame size                                   │
	│ flow_features_packet_length_upstream_flow_std             │ float    │ Upstream std frame size                                    │
	│ flow_features_packet_length_upstream_flow_var             │ float    │ Upstream variance frame size                               │
	│ flow_features_packet_length_upstream_flow_mad             │ float    │ Upstream median absolute deviation frame size              │
	│ flow_features_packet_length_upstream_flow_skew            │ float    │ Upstream skew frame size                                   │
	│ flow_features_packet_length_upstream_flow_kurtosis        │ float    │ Upstream kurtosi frame size                                │
	│ flow_features_packet_length_upstream_flow_10_percentile   │ float    │ Upstream 10%-ile frame size                                │
	│ flow_features_packet_length_upstream_flow_20_percentile   │ float    │ Upstream 20%-ile frame size                                │
	│ flow_features_packet_length_upstream_flow_30_percentile   │ float    │ Upstream 30%-ile frame size                                │
	│ flow_features_packet_length_upstream_flow_40_percentile   │ float    │ Upstream 40%-ile frame size                                │
	│ flow_features_packet_length_upstream_flow_50_percentile   │ float    │ Upstream 50%-ile frame size                                │
	│ flow_features_packet_length_upstream_flow_60_percentile   │ float    │ Upstream 60%-ile frame size                                │
	│ flow_features_packet_length_upstream_flow_70_percentile   │ float    │ Upstream 70%-ile frame size                                │
	│ flow_features_packet_length_upstream_flow_80_percentile   │ float    │ Upstream 80%-ile frame size                                │
	│ flow_features_packet_length_upstream_flow_90_percentile   │ float    │ Upstream 90%-ile frame size                                │
	│ flow_features_packet_length_downstream_flow_min           │ float    │ Downstream min frame (i.e., pkt with headers) size         │
	│ flow_features_packet_length_downstream_flow_max           │ float    │ Downstream max frame size                                  │
	│ flow_features_packet_length_downstream_flow_mean          │ float    │ Downstream mean frame size                                 │
	│ flow_features_packet_length_downstream_flow_std           │ float    │ Downstream std frame size                                  │
	│ flow_features_packet_length_downstream_flow_var           │ float    │ Downstream variance frame size                             │
	│ flow_features_packet_length_downstream_flow_mad           │ float    │ Downstream max frame size                                  │
	│ flow_features_packet_length_downstream_flow_skew          │ float    │ Downstream skew frame size                                 │
	│ flow_features_packet_length_downstream_flow_kurtosis      │ float    │ Downstream kurtosi frame size                              │
	│ flow_features_packet_length_downstream_flow_10_percentile │ float    │ Downstream 10%-ile frame size                              │
	│ flow_features_packet_length_downstream_flow_20_percentile │ float    │ Downstream 20%-ile frame size                              │
	│ flow_features_packet_length_downstream_flow_30_percentile │ float    │ Downstream 30%-ile frame size                              │
	│ flow_features_packet_length_downstream_flow_40_percentile │ float    │ Downstream 40%-ile frame size                              │
	│ flow_features_packet_length_downstream_flow_50_percentile │ float    │ Downstream 50%-ile frame size                              │
	│ flow_features_packet_length_downstream_flow_60_percentile │ float    │ Downstream 60%-ile frame size                              │
	│ flow_features_packet_length_downstream_flow_70_percentile │ float    │ Downstream 70%-ile frame size                              │
	│ flow_features_packet_length_downstream_flow_80_percentile │ float    │ Downstream 80%-ile frame size                              │
	│ flow_features_packet_length_downstream_flow_90_percentile │ float    │ Downstream 90%-ile frame size                              │
	│ flow_features_iat_biflow_min                              │ float    │ Bidirectional min inter arrival time                       │
	│ flow_features_iat_biflow_max                              │ float    │ Bidirectional max inter arrival time                       │
	│ flow_features_iat_biflow_mean                             │ float    │ Bidirectional mean inter arrival time                      │
	│ flow_features_iat_biflow_std                              │ float    │ Bidirectional std inter arrival time                       │
	│ flow_features_iat_biflow_var                              │ float    │ Bidirectional variance inter arrival time                  │
	│ flow_features_iat_biflow_mad                              │ float    │ Bidirectional median absolute deviation inter arrival time │
	│ flow_features_iat_biflow_skew                             │ float    │ Bidirectional skew inter arrival time                      │
	│ flow_features_iat_biflow_kurtosis                         │ float    │ Bidirectional kurtosi inter arrival time                   │
	│ flow_features_iat_biflow_10_percentile                    │ float    │ Bidirectional 10%-tile inter arrival time                  │
	│ flow_features_iat_biflow_20_percentile                    │ float    │ Bidirectional 20%-tile inter arrival time                  │
	│ flow_features_iat_biflow_30_percentile                    │ float    │ Bidirectional 30%-tile inter arrival time                  │
	│ flow_features_iat_biflow_40_percentile                    │ float    │ Bidirectional 40%-tile inter arrival time                  │
	│ flow_features_iat_biflow_50_percentile                    │ float    │ Bidirectional 50%-tile inter arrival time                  │
	│ flow_features_iat_biflow_60_percentile                    │ float    │ Bidirectional 60%-tile inter arrival time                  │
	│ flow_features_iat_biflow_70_percentile                    │ float    │ Bidirectional 70%-tile inter arrival time                  │
	│ flow_features_iat_biflow_80_percentile                    │ float    │ Bidirectional 80%-tile inter arrival time                  │
	│ flow_features_iat_biflow_90_percentile                    │ float    │ Bidirectional 90%-tile inter arrival time                  │
	│ flow_features_iat_upstream_flow_min                       │ float    │ Upstream min inter arrival time                            │
	│ flow_features_iat_upstream_flow_max                       │ float    │ Upstream max inter arrival time                            │
	│ flow_features_iat_upstream_flow_mean                      │ float    │ Upstream avg inter arrival time                            │
	│ flow_features_iat_upstream_flow_std                       │ float    │ Upstream std inter arrival time                            │
	│ flow_features_iat_upstream_flow_var                       │ float    │ Upstream variance inter arrival time                       │
	│ flow_features_iat_upstream_flow_mad                       │ float    │ Upstream median absolute deviation inter arrival time      │
	│ flow_features_iat_upstream_flow_skew                      │ float    │ Upstream skew inter arrival time                           │
	│ flow_features_iat_upstream_flow_kurtosis                  │ float    │ Upstream kurtosi inter arrival time                        │
	│ flow_features_iat_upstream_flow_10_percentile             │ float    │ Upstream 10%-ile inter arrival time                        │
	│ flow_features_iat_upstream_flow_20_percentile             │ float    │ Upstream 20%-ile inter arrival time                        │
	│ flow_features_iat_upstream_flow_30_percentile             │ float    │ Upstream 30%-ile inter arrival time                        │
	│ flow_features_iat_upstream_flow_40_percentile             │ float    │ Upstream 40%-ile inter arrival time                        │
	│ flow_features_iat_upstream_flow_50_percentile             │ float    │ Upstream 50%-ile inter arrival time                        │
	│ flow_features_iat_upstream_flow_60_percentile             │ float    │ Upstream 60%-ile inter arrival time                        │
	│ flow_features_iat_upstream_flow_70_percentile             │ float    │ Upstream 70%-ile inter arrival time                        │
	│ flow_features_iat_upstream_flow_80_percentile             │ float    │ Upstream 80%-ile inter arrival time                        │
	│ flow_features_iat_upstream_flow_90_percentile             │ float    │ Upstream 90%-ile inter arrival time                        │
	│ flow_features_iat_downstream_flow_min                     │ float    │ Downstream min inter arrival time                          │
	│ flow_features_iat_downstream_flow_max                     │ float    │ Downstream max inter arrival time                          │
	│ flow_features_iat_downstream_flow_mean                    │ float    │ Downstream mean inter arrival time                         │
	│ flow_features_iat_downstream_flow_std                     │ float    │ Downstream std inter arrival time                          │
	│ flow_features_iat_downstream_flow_var                     │ float    │ Downstream variance inter arrival time                     │
	│ flow_features_iat_downstream_flow_mad                     │ float    │ Downstream median absolute deviation inter arrival time    │
	│ flow_features_iat_downstream_flow_skew                    │ float    │ Downstream skew inter arrival time                         │
	│ flow_features_iat_downstream_flow_kurtosis                │ float    │ Downstream kurtosi inter arrival time                      │
	│ flow_features_iat_downstream_flow_10_percentile           │ float    │ Downstream 10%-ile inter arrival time                      │
	│ flow_features_iat_downstream_flow_20_percentile           │ float    │ Downstream 20%-ile inter arrival time                      │
	│ flow_features_iat_downstream_flow_30_percentile           │ float    │ Downstream 30%-ile inter arrival time                      │
	│ flow_features_iat_downstream_flow_40_percentile           │ float    │ Downstream 40%-ile inter arrival time                      │
	│ flow_features_iat_downstream_flow_50_percentile           │ float    │ Downstream 50%-ile inter arrival time                      │
	│ flow_features_iat_downstream_flow_60_percentile           │ float    │ Downstream 60%-ile inter arrival time                      │
	│ flow_features_iat_downstream_flow_70_percentile           │ float    │ Downstream 70%-ile inter arrival time                      │
	│ flow_features_iat_downstream_flow_80_percentile           │ float    │ Downstream 80%-ile inter arrival time                      │
	│ flow_features_iat_downstream_flow_90_percentile           │ float    │ Downstream 90%-ile inter arrival time                      │
	│ flow_metadata_bf_device                                   │ str      │ Ethernet address                                           │
	│ flow_metadata_bf_label_source                             │ str      │ Constant value 'netstate'                                  │
	│ flow_metadata_bf_label                                    │ str      │ original mirage label                                      │
	│ flow_metadata_bf_sublabel                                 │ str      │ n.a.                                                       │
	│ flow_metadata_bf_label_version_code                       │ str      │ n.a.                                                       │
	│ flow_metadata_bf_label_version_name                       │ str      │ n.a.                                                       │
	│ flow_metadata_bf_labeling_type                            │ str      │ exact=via netstat; most-common=via experiment              │
	│ flow_metadata_bf_num_packets                              │ int      │ Bidirectional number of pkts                               │
	│ flow_metadata_bf_ip_packet_bytes                          │ int      │ Bidirectional bytes (including headers)                    │
	│ flow_metadata_bf_l4_payload_bytes                         │ int      │ Bidirectional payload bytes                                │
	│ flow_metadata_bf_duration                                 │ float    │ Bidirectional duration                                     │
	│ flow_metadata_uf_num_packets                              │ int      │ Upload number of pkts                                      │
	│ flow_metadata_uf_ip_packet_bytes                          │ int      │ Upload bytes (including headers)                           │
	│ flow_metadata_uf_l4_payload_bytes                         │ int      │ Upload payload bytes                                       │
	│ flow_metadata_uf_duration                                 │ float    │ Upload duration                                            │
	│ flow_metadata_uf_mss                                      │ float    │ Upload maximum segment size                                │
	│ flow_metadata_uf_ws                                       │ float    │ Upload window scaling                                      │
	│ flow_metadata_df_num_packets                              │ int      │ Download number of packets                                 │
	│ flow_metadata_df_ip_packet_bytes                          │ int      │ Download bytes (including headers)                         │
	│ flow_metadata_df_l4_payload_bytes                         │ int      │ Download payload bytes                                     │
	│ flow_metadata_df_duration                                 │ float    │ Download duration                                          │
	│ flow_metadata_df_mss                                      │ float    │ Download maximum segment size                              │
	│ flow_metadata_df_ws                                       │ float    │ Download window scaling                                    │
	│ flow_metadata_bf_activity                                 │ str      │ Experiment activity                                        │
	│ strings                                                   │ list     │ ASCII string extracted from payload                        │
	│ android_name                                              │ str      │ app name (based on filename)                               │
	│ device_name                                               │ str      │ device name (based on filename)                            │
	│ app                                                       │ category │ label (background|android app)                             │
	│ src_ip                                                    │ str      │ Source IP                                                  │
	│ src_port                                                  │ str      │ Source port                                                │
	│ dst_ip                                                    │ str      │ Destination IP                                             │
	│ dst_port                                                  │ str      │ Destination port                                           │
	│ proto                                                     │ str      │ L4 protol                                                  │
	│ packets                                                   │ int      │ Number of (bidirectional) packets                          │
	└───────────────────────────────────────────────────────────┴──────────┴────────────────────────────────────────────────────────────┘
	```

!!! note "tcbench datasets schema --name mirage22 --type filtered"
	```hl_lines="4 15 22 23 24"
	┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
	┃ Field                             ┃ Dtype    ┃ Description                                                          ┃
	┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
	│ row_id                            │ int      │ Unique flow id                                                       │
	│ conn_id                           │ str      │ Flow 5-tuple                                                         │
	│ packet_data_l4_raw_payload        │ np.array │ List of list with each packet payload                                │
	│ flow_metadata_bf_label            │ str      │ original mirage label                                                │
	│ flow_metadata_bf_activity         │ str      │ Experiment activity                                                  │
	│ flow_metadata_bf_labeling_type    │ str      │ exact=via netstat; most-common=via experiment                        │
	│ flow_metadata_bf_l4_payload_bytes │ int      │ Bidirectional payload bytes                                          │
	│ flow_metadata_bf_duration         │ float    │ Bidirectional duration                                               │
	│ strings                           │ list     │ ASCII string extracted from payload                                  │
	│ android_name                      │ str      │ app name (based on filename)                                         │
	│ device_name                       │ str      │ device name (based on filename)                                      │
	│ app                               │ category │ label (background|android app)                                       │
	│ src_ip                            │ str      │ Source IP                                                            │
	│ src_port                          │ str      │ Source port                                                          │
	│ dst_ip                            │ str      │ Destination IP                                                       │
	│ dst_port                          │ str      │ Destination port                                                     │
	│ proto                             │ str      │ L4 protocol                                                          │
	│ packets                           │ int      │ Number of (bidirectional) packets                                    │
	│ pkts_size                         │ str      │ Packet size time series                                              │
	│ pkts_dir                          │ str      │ Packet diretion time series                                          │
	│ timetofirst                       │ str      │ Delta between the each packet timestamp the first packet of the flow │
	└───────────────────────────────────┴──────────┴──────────────────────────────────────────────────────────────────────┘
	```

!!! note "tcbench datasets schema --name mirage22 --type splits"
	```
	┏━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
	┃ Field         ┃ Dtype    ┃ Description                  ┃
	┡━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
	│ train_indexes │ np.array │ row_id of training samples   │
	│ val_indexes   │ np.array │ row_id of validation samples │
	│ test_indexes  │ np.array │ row_id of test samples       │
	│ split_index   │ int      │ Split id                     │
	└───────────────┴──────────┴──────────────────────────────┘
	```

##### `utmobilenet21`

??? note "tcbench datasets schema --name utmobilenet21 --type unfiltered"
	```
	┏━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
	┃ Field       ┃ Dtype    ┃ Description                                                                  ┃
	┡━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
	│ row_id      │ int      │ Unique flow id                                                               │
	│ src_ip      │ str      │ Source ip of the flow                                                        │
	│ src_port    │ int      │ Source port of the flow                                                      │
	│ dst_ip      │ str      │ Destination ip of the flow                                                   │
	│ dst_port    │ int      │ Destination port of the flow                                                 │
	│ ip_proto    │ int      │ Protocol of the flow (TCP or UDP)                                            │
	│ first       │ float    │ Timestamp of the first packet                                                │
	│ last        │ float    │ Timestamp of the last packet                                                 │
	│ duration    │ float    │ Duration of the flow                                                         │
	│ packets     │ int      │ Number of packets in the flow                                                │
	│ bytes       │ int      │ Number of bytes in the flow                                                  │
	│ partition   │ str      │ From which folder the flow was originally stored                             │
	│ location    │ str      │ Label originally provided by the dataset (see the related paper for details) │
	│ fname       │ str      │ Original filename where the packets of the flow come from                    │
	│ app         │ category │ Final label of the flow, encoded as pandas category                          │
	│ pkts_size   │ np.array │ Packet size time series                                                      │
	│ pkts_dir    │ np.array │ Packet diretion time series                                                  │
	│ timetofirst │ np.array │ Delta between the each packet timestamp the first packet of the flow         │
	└─────────────┴──────────┴──────────────────────────────────────────────────────────────────────────────┘
	```

!!! note "tcbench datasets schema --name utmobilenet21 --type filtered"
	```hl_lines="4 18 19 20 21"
	┏━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
	┃ Field       ┃ Dtype    ┃ Description                                                                  ┃
	┡━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
	│ row_id      │ int      │ Unique flow id                                                               │
	│ src_ip      │ str      │ Source ip of the flow                                                        │
	│ src_port    │ int      │ Source port of the flow                                                      │
	│ dst_ip      │ str      │ Destination ip of the flow                                                   │
	│ dst_port    │ int      │ Destination port of the flow                                                 │
	│ ip_proto    │ int      │ Protocol of the flow (TCP or UDP)                                            │
	│ first       │ float    │ Timestamp of the first packet                                                │
	│ last        │ float    │ Timestamp of the last packet                                                 │
	│ duration    │ float    │ Duration of the flow                                                         │
	│ packets     │ int      │ Number of packets in the flow                                                │
	│ bytes       │ int      │ Number of bytes in the flow                                                  │
	│ partition   │ str      │ From which folder the flow was originally stored                             │
	│ location    │ str      │ Label originally provided by the dataset (see the related paper for details) │
	│ fname       │ str      │ Original filename where the packets of the flow come from                    │
	│ app         │ category │ Final label of the flow, encoded as pandas category                          │
	│ pkts_size   │ np.array │ Packet size time series                                                      │
	│ pkts_dir    │ np.array │ Packet diretion time series                                                  │
	│ timetofirst │ np.array │ Delta between the each packet timestamp the first packet of the flow         │
	└─────────────┴──────────┴──────────────────────────────────────────────────────────────────────────────┘
	```

!!! note "tcbench datasets schema --name utmobilenet21 --type splits"
	```
	┏━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
	┃ Field         ┃ Dtype    ┃ Description                  ┃
	┡━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
	│ train_indexes │ np.array │ row_id of training samples   │
	│ val_indexes   │ np.array │ row_id of validation samples │
	│ test_indexes  │ np.array │ row_id of test samples       │
	│ split_index   │ int      │ Split id                     │
	└───────────────┴──────────┴──────────────────────────────┘
	```

### :octicons-number-24: Samples count report

As the name suggests, a *samples count report* details
how many samples (i.e., flows) are available for each app.
These reports are shown during installation, but can 
be retrieved at any time using the subcommand `datasets samples-count`.

They can be generated for unfiltered, filtered or based on splits,
but the command requires familiarity with the parametrization
semantic.

#### ucdavis-icdm19

For instance, the following provides the unfitered view
for the `ucdavis-icdm19` dataset

```
tcbench datasets samples-count --name ucdavis-icdm19
```

!!! note "Output"
	```
	unfiltered
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
	```

While to obtain the breakdown of the first train split

```
tcbench datasets samples-count --name ucdavis-icdm19 --split 0
```

!!! note "Output"
	```
	filtered, split: 0
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
	```

...or the `human` test split

```
tcbench datasets samples-count --name ucdavis-icdm19 --split human
```

!!! note "Output"
	```
	filtered, split: human
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
	```

#### Examples for other datasets

Other datasets can be filtered based on the `--min_pkts` options.

For instance, the following is the overall
view for `mirage22`

```
tcbench datasets samples-count --name mirage22
```

!!! note "Output"
	```
	unfiltered
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
	```

This counts reduce when filtering by `--min-pkts 1000`

```
tcbench datasets samples-count --name mirage22 --min-pkts 1000
```

!!! note "Output"
	```
	min_pkts: 1000
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
	```

...and you can also obtain the breakdown from a specific split
```
tcbench datasets samples-count --name mirage22 --min-pkts 1000 --split 0
```

!!! note "Output"
	```
	min_pkts: 1000, split: 0
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
