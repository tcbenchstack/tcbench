# The tcbench framework

tcbench is a ML/DL framework specific for __Traffic Classification (TC)__
created as research project by the AI4NET team of the Huawei Technologies
research center in Paris, France.

!!! info "What is Traffic Classification?"
    
    Nodes within a computer network operate by exchanging 
    information, namely *packets*, which is regulated according
    to standardized protocols (e.g., HTTP for the web). So to understand 
    the network health it is required to constantly monitor
    this information flow and react accordingly. For instance, one
    might want to prioritize certain traffic (e.g., video meeting)
    or block it (e.g., social media in working environment).

    Traffic classification is the the act of labeling an exchange of packets 
    based on the Internet application which generated it.


The academic literature is ripe with methods and proposals for TC.
Yet, it is scarce of code artifacts and public datasets 
do not offer common conventions of use.

We designed tcbench with the following goals in mind:

| Goal | State of the art | tcbench |
|:-----|:-----------------|:--------|
|__:octicons-stack-24: Data curation__ | There are a few public datasets for TC, yet no common format/schema, cleaning process, or standard train/val/test folds. | An (opinionated) curation of datasets to create easy to use parquet files with associated train/val/test fold.|
|__:octicons-file-code-24: Code__ | TC literature has no reference code base for ML/DL modeling | tcbench is [:material-github: open source](https://github.com/tcbenchstack/tcbench) with an easy to use CLI based on [:fontawesome-solid-arrow-pointer: click](https://click.palletsprojects.com/en/8.1.x/)|
|__:material-monitor-dashboard: Model tracking__ | Most of ML framework requires integration with cloud environments and subscription services | tcbench uses [aimstack](https://aimstack.io/) to save on local servers metrics during training which can be later explored via its web UI or aggregated in report summaries using tcbench |

## Features and roadmap

tcbench is still under development, but (as suggested by its name) ultimately aims
to be a reference framework for benchmarking multiple ML/DL solutions 
related to TC.

At the current stage, tcbench offers

* Integration with 4 datasets, namely `ucdavis-icdm19`, `mirage19`, `mirage22` and `utmobilenet21`.
You can use these datasets and their curated version independently from tcbench.
Check out the [dataset install](/tcbench/datasets/install) process and [dataset loading tutorial](/tcbench/datasets/guides/tutorial_load_datasets).

* Good support for flowpic input representation and minimal support
for 1d time series (based on network packets properties) input representation.

* Data augmentation functionality for flowpic input representation.

* Modeling via XGBoost, vanilla DL supervision and contrastive learning (via SimCLR or SupCon).

Most of the above functionalities described relate to our __:material-file-document-outline: [IMC23 paper](/tcbench/papers/imc23/)__.

More exiting features including more datasets and algorithms will come in the next months. 

Stay tuned :wink:!

