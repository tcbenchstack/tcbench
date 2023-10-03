---
title: Artifacts
icon: material/hexagon-outline
---

# IMC23 Paper Artifacts

The paper is associated to the following types of artifacts:

* __:octicons-stack-24: Data__: This includes 
    * The datasets curation and splits for [`ucdavis-icdm19`](/datasets/install/ucdavis-icdm19), 
    [`mirage19`](/datasets/install/mirage19), 
    [`mirage22`](/datasets/install/mirage22) and 
    [`utmobilenet21`](/datasets/install/utmobilenet21). Please refer to the 
    [datasets webpage](/datasets/) and related pages for more details.
    * All [:simple-docsdotrs: models and logs](/papers/imc23/ml_artifacts/) generated through our modeling campaigns.

* __:octicons-file-code-24: Code__: This includes 
    * A collection of [:simple-jupyter: Jupyter notebooks](/papers/imc23/notebooks) 
    used for the tables and figures of the paper.
    * A collection of data to support [:simple-pytest: pytest unittest](/papers/imc23/pytest) related to the 
    results collected for the paper.


## :simple-figshare: Figshare material

The artifacts are stored in a [:simple-figshare: figshare collection](https://figshare.com/collections/IMC23_artifacts_-_Replication_Contrastive_Learning_and_Data_Augmentation_in_Traffic_Classification_Using_a_Flowpic_Input_Representation/6849252)
with the following items:

* `curated_datasets_ucdavis-icdm19.tgz`: A curated version of the dataset presented by *Rezaei et al.* in ["How to Achieve High Classification Accuracy with Just a Few Labels: A Semi-supervised Approach Using Sampled Packets"](https://doi.org/10.48550/arXiv.1812.09761).

* `curated_datasets_utmobilenet21.tgz`: A curated version of the dataset presented by *Heng et al.* in ["UTMobileNetTraffic2021: A Labeled Public Network Traffic Dataset"](https://doi.org/10.1109/LNET.2021.3098455).

* `imc23_ml_artifacts.tgz`: Models and output logs generated via tcbench.

* `imc23_notebooks.tgz`: A collection of [jupyter notebooks](/papers/imc23/notebooks) for recreating tables and figures from the paper.

* `imc23_pytest_resources.tgz`: A collection of reference [resources for pytest](/papers/imc23/pytest) unit testing (to verify model training replicability).

* `ucdavis-icdm19-git-repo-forked.tgz`: A fork of the repository https://github.com/shrezaei/Semi-supervised-Learning-QUIC- to verify results of "How to Achieve High Classification Accuracy with Just a Few Labels: A Semi-supervised Approach Using Sampled Packets" https://doi.org/10.48550/arXiv.1812.09761


## :material-download: Downloading artifacts

Each artifact can be manually downloaded from the figshare collection. However, 
__make sure to refer to the latest version__ of an archive when downloading manually.

tcbench offers automated procedures to fetch the right content from figshare:

* For datasets please refer to [datasets page](/datasets) page, 
the specific page for each datasets and the [import command](/datasets/import).

* For the remaning, you can use the `fetch-artifacts` subcommand
with the following process

1.  First of all, prepare a python virtual environment, for example via :simple-anaconda: conda
    ```
    conda create -n tcbench python=3.10 pip
    conda activate tcbench
    ```

2. Clone the tcbench repo with the imc23 tag
    ```
    ```

3. Install tcbench
    ```
    python -m pip install .[dev]
    ```

4. Fetch the artifacts
    ```
    tcbench fetch-artifacts
    ```
