<div align="center">
  <img src="https://tcbenchstack.github.io/tcbench/tcbench_logo.svg" width="400px"/>
  <h3>An ML/DL framework for Traffic Classification TC</h3>
  <a href="https://tcbenchstack.github.io/tcbench">
  <img width="24" height="24" src="https://img.icons8.com/fluency/48/domain.png" alt="domain"/>
  <b>Documentation</b>
  </a>
</div>

<br>

tcbench design is cored in the following objectives:

* Easing ML/DL models training/testing results replicability.
* Tight integration with public TC datasets with ease data installation and curation,
* Model tracking via [AIM](https://github.com/aimhubio/aim). 
* Rich command line for executing modeling campaings and collecting performance reports.


## ...wait, what is Traffic Classification?
    
A computer network is formed by hosts that exchange
information, namely *packets*, according
to standardized protocols (e.g., [HTTP](https://en.wikipedia.org/wiki/HTTP) is the protocol used for the web). 
So to properly operate/manage networks one is required to monitor
this flow of information and react accordingly. For instance, 
in an office/enterprise environment, one might want to prioritize video meeting traffic
while limit social media traffic.

[__Traffic classification__](https://en.wikipedia.org/wiki/Traffic_classification) 
is the the act of labeling an exchange of packets 
between network hosts based on the application that generated it.
For instance, you want to identify traffic related to zoom/webx/skype/etc. calls 
or traffic related to twitter/instagram/facebook/mastodon
out of all traffic flowing throught the network.


## Motivations

The academic literature is ripe with methods and proposals for TC.
Yet, it is scarce of code artifacts and public datasets 
do not offer common conventions of use.

We designed tcbench with the following goals in mind:

| Goal | State of the art | tcbench |
|:-----|:-----------------|:--------|
| __Data curation__ | There are a few public datasets for TC, yet no common format/schema, cleaning process, or standard train/val/test folds. | An (opinionated) curation of datasets to create easy to use parquet files with associated train/val/test fold.|
|__Code__ | TC literature has no reference code base for ML/DL modeling | tcbench is [open source](https://github.com/tcbenchstack/tcbench) with an easy to use CLI based on [click](https://click.palletsprojects.com/en/8.1.x/)|
|__Model tracking__ | Most of ML framework requires integration with cloud environments and subscription services | tcbench uses [aimstack](https://aimstack.io/) to save on local servers metrics during training which can be later explored via its web UI or aggregated in report summaries using tcbench |

## Install

Create a conda environment

```
conda create -n tcbench python=3.10 pip
conda activate tcbench
python -m pip install tcbench
```

For the developer version
```
python -m pip install tcbench[dev]
```

## Features and roadmap

tcbench is still under development, but (as suggested by its name) ultimately aims
to be a reference framework for benchmarking multiple ML/DL solutions 
related to TC.

At the current stage, tcbench offers

* Integration with 4 datasets, namely `ucdavis-icdm19`, `mirage19`, `mirage22` and `utmobilenet21`.
You can use these datasets and their curated version independently from tcbench.
Check out the [dataset install](https://tcbenchstack.github.io/tcbench/datasets/install) process and [dataset loading tutorial](https://tcbenchstack.github.io/tcbench/datasets/guides/tutorial_load_datasets/).

* Good support for flowpic input representation.

* Initial support for for 1d packet time series (based on network packets properties) input representation.

* Data augmentation functionality for flowpic input representation.

* Modeling via XGBoost, vanilla DL supervision and contrastive learning (via SimCLR or SupCon).

More exiting features including more datasets and algorithms will come in the next months. 

Stay tuned ;)!

## Papers

* ["Replication: Contrastive Learning and Data Augmentation in Traffic Classification Using a Flowpic Input Representation"](https://arxiv.org/abs/2309.09733) __preprint__<br>
A. Finamore, C. Wang, J. Krolikowski, J. M. Navarro, F. Chen, D. Rossi<br>
ACM Internet Measurements Conference (IMC), 2023

