---
title: Artifacts
icon: material/hexagon-outline
---

# IMC23 Paper Artifacts

The paper is associated to the following types of artifacts:

* __:octicons-stack-24: Data__: This includes 
    * The datasets curation and splits for [`ucdavis-icdm19`](/tcbench/datasets/install/ucdavis-icdm19), 
    [`mirage19`](/tcbench/datasets/install/mirage19), 
    [`mirage22`](/tcbench/datasets/install/mirage22) and 
    [`utmobilenet21`](/tcbench/datasets/install/utmobilenet21). Please refer to the 
    [datasets webpage](/tcbench/datasets/) and related pages for more details.
    * All [:simple-docsdotrs: models and logs](/tcbench/papers/imc23/ml_artifacts/) generated through our modeling campaigns.

* __:octicons-file-code-24: Code__: This includes 
    * A collection of [:simple-jupyter: Jupyter notebooks](/tcbench/papers/imc23/notebooks) 
    used for the tables and figures of the paper.
    * A collection of data to support [:simple-pytest: pytest unittest](/tcbench/papers/imc23/pytest) related to the 
    results collected for the paper.


## :simple-figshare: Figshare material

The artifacts are stored in a [:simple-figshare: figshare collection](https://figshare.com/collections/IMC23_artifacts_-_Replication_Contrastive_Learning_and_Data_Augmentation_in_Traffic_Classification_Using_a_Flowpic_Input_Representation/6849252)
with the following items:

* `curated_datasets_ucdavis-icdm19.tgz`: A curated version of the dataset presented by *Rezaei et al.* in ["How to Achieve High Classification Accuracy with Just a Few Labels: A Semi-supervised Approach Using Sampled Packets"](https://doi.org/10.48550/arXiv.1812.09761).

* `curated_datasets_utmobilenet21.tgz`: A curated version of the dataset presented by *Heng et al.* in ["UTMobileNetTraffic2021: A Labeled Public Network Traffic Dataset"](https://doi.org/10.1109/LNET.2021.3098455).

* `imc23_ml_artifacts.tgz`: Models and output logs generated via tcbench.

* `imc23_notebooks.tgz`: A collection of [jupyter notebooks](/tcbench/papers/imc23/notebooks) for recreating tables and figures from the paper.

* `imc23_pytest_resources.tgz`: A collection of reference [resources for pytest](/tcbench/papers/imc23/pytest) unit testing (to verify model training replicability).

* `ucdavis-icdm19-git-repo-forked.tgz`: A fork of the repository https://github.com/shrezaei/Semi-supervised-Learning-QUIC- to verify results of "How to Achieve High Classification Accuracy with Just a Few Labels: A Semi-supervised Approach Using Sampled Packets" https://doi.org/10.48550/arXiv.1812.09761


## :material-download: Downloading artifacts

Each artifact can be manually downloaded from the figshare collection. However, 
__make sure to refer to the latest version__ of an archive when downloading manually.

tcbench offers automated procedures to fetch the right content from figshare:

* For datasets please refer to [datasets page](/tcbench/datasets) page, 
the specific page for each datasets and the [import command](/tcbench/datasets/import).

* For the remaning, you can use the `fetch-artifacts` subcommand
with the following process

1.  First of all, prepare a python virtual environment, for example via :simple-anaconda: conda
    ```
    conda create -n tcbench python=3.10 pip
    conda activate tcbench
    ```

2. Clone the tcbench repo and use the `imc23` branch
    ```
    git clone https://github.com/tcbenchstack/tcbench.git tcbench.git
    cd tcbench.git
    git checkout imc23
    ```

3. Install tcbench
    ```
    python -m pip install .[dev]
    ```

4. Fetch the artifacts
    ```
    tcbench fetch-artifacts
    ```

This will install locally

* The [notebooks](/tcbench/papers/imc23/notebooks/) for replicating tables and figures of the paper under `/notebooks/imc23`.
The cloned repository already contains the notebooks but since the code might
change, the version fetched from figshare is identical to what used for the submission.

* The [ml-artifacts](/tcbench/papers/imc23/ml_artifacts/) under `/notebooks/imc23/campaigns`.

* The [pytest resources](/tcbench/papers/imc23/pytest/) for enabling unit tests.


!!! warning "Packages depencency version and `/imc23` branch"

    When installing tcbench via pypi of from the main branch of the repository,
    only a few sensible packages have a pinned version.

    If you are trying to replicate the results of the paper, please
    refer to the `/imc23` branch which also contains a
    `requirements-imc23.txt` generated via `pip freeze` from 
    the environment used for collecting results. 

    Based on our experience, the most probable cause of results inconsistency
    is due to package version. 
