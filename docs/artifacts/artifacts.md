The submission is associated to three types of artifacts

* __:spider_web: Website__: This website serves as a primary source
of documentation. It collects
    * Documentation about [datasets :simple-artifacthub:](../datasets/install).
    * Documentation about our modeling framework called :material-link-off:[`tcbench`]().
    * Guides on how to [run experiments :fontawesome-solid-flask:](/modeling/campaigns/) via `tcbench`.

* __:octicons-file-code-24: Code__: This includes 
    * All source code related to :material-link-off:[`tcbench` :material-language-python:]().
    * A collection of [:simple-jupyter: Jupyter notebooks](../paper_tables_and_figures/reference) 
    used for the tables and figures of the submission.

* __:octicons-stack-24: Data__: This includes 
    * The [datasets install, curation and split generation :material-rhombus-split-outline:](../datasets/install) used in our modeling
    * All [models and logs :material-file-multiple-outline:](/modeling/exploring_artifacts/) generated through our modeling campaigns.

## :simple-figshare: Figshare material

A key objective of our submission is to made available all artifacts
to the research community. 
For instance, all code will be pushed to a :material-github: github repository,
this website will be published on github pages or similar solutions,
and data artifacts will be on a public cloud storage solution.

Yet, due to double-blind policy, we temporarily uploaded our artifacts to a
:simple-figshare: [figshare repository](https://figshare.com/s/cab23f730cfbc5172f78).

More specifically, on figshare you find the following tarball.

* `website_documentation.tgz`: Well...if you are reading this page
you already know the tarball contains this website :octicons-smiley-24:.

* `code_artifacts_paper132.tgz`: All code developed. See 
    * [Quick tour](../quick_tour) for `tcbench`.
    * [Table and figures](../paper_tables_and_figures/reference/) for jupyter notebooks.

* `curated_datasets.tgz`: The preprocessed version of the datasets. 
Please see the datasets pages in this website.

* `ml_artifacts.tgz`: All output data generated via modeling campaigns.
For fine grained view, those can be explored via [AIM web UI](/modeling/exploring_artifacts/#aim-web-ui) 
while results are generated via [:simple-jupyter: Jupyter notebooks](../paper_tables_and_figures/reference/).

## :material-package-variant: Unpack artifacts

In the figshare folder we also provide a `unpack_scripts.tgz` 
tarball containing the following scripts

```
unpack_all.sh
_unpack_code_artifacts_paper132.sh
_unpack_curated_datasets.sh
_unpack_ml_artifacts.sh
```

These are simple bash scripts to simplify the 
extraction and installation of all material.

Use the following process

1.  First of all, prepare a python virtual environment, for example via :simple-anaconda: conda
    ```
    conda create -n tcbench python=3.10 pip
    conda activate tcbench
    ```

2. Download all figshare tarballs in the same folder and run
    ```
    tar -xzvf unpack_script.tgz
    bash ./unpack_all.sh
    ```
