# Modeling introduction

When training ML/DL models, 
finding the right combination of data
preprocessing/splitting, algorithms and
hyper-parameters can be challenging.
Even more so when the modeling process 
aims to be [repeatable/replicable/reproducible](https://www.acm.org/publications/policies/artifact-review-badging).

To ease this process, tcbench is designed to ease the

* Collection of __:material-pencil-ruler: telemetry and metadata__.
This includes bind the generated output to 
the input parameters used to create models.

* Collection of __:simple-artifacthub: artifacts__ such as 
the models created as well performance
reports (e.g., loss evolution during training 
and testing confusion matrixes).

This is possible thanks to a tight integration
with AIM with some extra ad-hoc components.

## AIM tracking

[AIM stack](https://aimstack.io/) is an
open-source self-hosted model
tracking framework enabling logging of metrics 
related to model training. 

Such telemetry 
can later be explored via a [web interface](https://aimstack.readthedocs.io/en/latest/ui/overview.html)
or [programmatically extracted](https://aimstack.readthedocs.io/en/latest/using/query_runs.html) via AIM SKD.

!!! info "__Why AIM?__"

    There are [many solutions for model tracking](https://neptune.ai/blog/best-ml-experiment-tracking-tools).
    While frameworks such as __Weights & Biases__ or __Neptune.ai__
    are extremely rich with features, unfortunately they typically 
    are cloud-based solutions and not necessarily open-sourced.

    Alternative frameworks such as __Tensorboard__ and __MLFlow__
    have only primitive functionalities with respect to AIM.

    AIM is sitting in the middle of this spectrum:
    it is self-hosted (i.e., no need to push data to the cloud)
    and provides nice data exploration features.

AIM collects modeling metadata into __repositories__
fully under the control of the end-user:

* Repositories are not tied to specific projects and
it is up to the end-user define the semantic of the repository.
In other words, users can decide to track in a repository
metrics related to multiple experiments of the same model
(e.g., different hyper parametrization) but single repository
can be used to track completely different experiments 
(e.g., different metrics, hyper-parameters, properties
across experiments).

* There is no limit on the amount of repositories 
can be created. All repository are local to end-users
infrastructure so the only limitation is the end-users
storage.

AIM repositories are collection of "runs", each
representing a different modeling experiment and
identifies by a unique ID automatically generated
by the framework. 

## Runs and campaigns

`tcbench` tracks in an AIM repository two types of tasks,
namely *runs* and *campaigns*:

* A __:material-cube: run__ has a 1:1 matching with the run defined
by AIM, i.e., it corresponds to the training of an
individual ML/DL model and is "minimal experiment object" used by AIM,
i.e., any tracked metadata need to be associated to an AIM run.

A run is associated to both individual values
(e.g., best validation loss observed or the final accuracy score)
as well as series (e.g., loss value for each epoch).

Morever, tracked metrics are associated to a *context* 
expressing if they are generate using train, validation or test set.

* A __:simple-docsdotrs: campaign__ corresponds to a
collection of runs. 

!!! tip "Runs -vs- Collections"

    Runs are the fundamental building block for collecting
    modeling results. But they are also the fundamental
    unit when developing/debugging modeling tasks.

    Conversely, campaigns are intended for 
    grouping semantically similar runs 
    and store them into a single repository.
    Hence, different campaigns are stored into different
    repositories.
