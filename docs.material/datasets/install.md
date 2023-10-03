---
icon: material/arrow-down-bold-box
---

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


