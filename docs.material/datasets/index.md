# Datasets

TCBench supports the following *public* traffic classification datasets

##### Table : Datasets properties
| Name | Applications | Links | License | Our curation |
|:----:|:------------:|:-----:|:-------:|:------------:|
|[`ucdavis-icdm19`](/datasets/install/ucdavis-icdm19/)|5|[:fontawesome-regular-file-pdf:](https://arxiv.org/pdf/1812.09761.pdf)[:material-package-down:](https://drive.google.com/drive/folders/1Pvev0hJ82usPh6dWDlz7Lv8L6h3JpWhE)[:material-github:](https://github.com/shrezaei/Semi-supervised-Learning-QUIC-)| [:material-creative-commons:](https://creativecommons.org/licenses/by/4.0/) | [:simple-figshare:](https://figshare.com/articles/dataset/curated_datasets_ucdavisicdm19_tgz/23538141/1) |
|[`mirage19`](/datasets/install/mirage19/)|20|[:fontawesome-regular-file-pdf:](http://wpage.unina.it/antonio.montieri/pubs/MIRAGE_ICCCS_2019.pdf)[:material-package-down:](https://traffic.comics.unina.it/mirage/MIRAGE/MIRAGE-2019_traffic_dataset_downloadable_v2.tar.gz)[:material-web:](https://traffic.comics.unina.it/mirage/mirage-2019.html)| [:material-creative-commons: NC-ND](http://creativecommons.org/licenses/by-nc-nd/4.0/) | - |
|[`mirage22`](/datasets/install/mirage22/)|9|[:fontawesome-regular-file-pdf:](http://wpage.unina.it/antonio.montieri/pubs/_C__IEEE_CAMAD_2021___Traffic_Classification_Covid_app.pdf)[:material-package-down:](https://traffic.comics.unina.it/mirage/MIRAGE/MIRAGE-COVID-CCMA-2022.zip)[:material-web:](https://traffic.comics.unina.it/mirage/mirage-covid-ccma-2022.html)| [:material-creative-commons: NC-ND](http://creativecommons.org/licenses/by-nc-nd/4.0/) | - |
|[`utmobilenet21`](/datasets/install/utmobilenet21/)|17|[:fontawesome-regular-file-pdf:](https://ieeexplore.ieee.org/abstract/document/9490678/)[:material-package-down:](https://github.com/YuqiangHeng/UTMobileNetTraffic2021)[:material-github:](https://github.com/YuqiangHeng/UTMobileNetTraffic2021)| [:simple-gnu: GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) | [:simple-figshare:](https://figshare.com/articles/dataset/curated_datasets_utmobilenet21_tgz/23648703/1) |

At a glance, these datasets

* Are collections of either *CSV or JSON* files.

* Are reporting individual *packet level information or per-flow time series* and metrics.

* May have been organized in subfolders, namely *partitions*,
to reflect the related measurement campaign (see `ucdavis-icdm19`, `utmobilenet21`).

* May have file names carrying semantic.

* May require preprocessing to remove "background" noise, i.e.,
traffic unrelated to a target application (see `mirage19` and `mirage22`).

* Do not have reference train/validation/test splits.

In other words, these datasets need to be *curated* 
to be used.

!!! tip "Important"

    The integration of these datasets in tcbench does not break
    the original licensing of the data nor it breaks their ownership.
    Rather, the integration aims at easing the access to these dataset.
    We thus encourage researchers and practitioners interesting in
    using these datasets to cite the original publications 
    (see links in the table above).

## Terminology

When describing datasets and related processing we
use the following conventions:

* A __partition__ is a set of samples 
pre-defined by the authors of the dataset.
For instance, a partition can relate to a
specific set of samples to use for training/test 
(see [`ucdavis-icdm19`](/datasets/install/ucdavis-icdm19/)).

* A __split__ is a set of indexes of samples
that need to be used for train/validation/test.

* An __unfiltered__ dataset corresponds a
monolithic parquet files containing the
original raw data of a dataset (no filtering 
is applied).

* A __curated__ dataset is generated 
processing the unfiltered parquet 
to clean noise, remove small flows, etc.,
and each dataset have slightly different
curation rules.
