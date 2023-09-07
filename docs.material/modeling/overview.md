# Modeling overview

When training ML/DL models, 
finding the right combination of data
preprocessing/splitting, algorithms and
hyper-parameters can be challenging.
Even more so when the modeling process 
aims to be [repeatable/replicable/reproducible](https://www.acm.org/publications/policies/artifact-review-badging).

To ease this process is key to

* Collect __telemetry and metadata__.
This includes both the parameters used to create models
as well as lower level metrics such as the evolution of the
training loss over time.

* Generate __artifacts__ such as 
reports about the overall performance
(e.g., confusion matrixes).

## AIM stack tracking

`tcbench` integrates
with [AIM stack](https://aimstack.io/), an
open-source and self-hosted model
tracking framework enabling logging of metrics 
related to model training. Such telemetry 
can later be explored via a [web interface](https://aimstack.readthedocs.io/en/latest/ui/overview.html)
or [programmatically extracted](https://aimstack.readthedocs.io/en/latest/using/query_runs.html) via AIM SKD.

!!! info "__Why not using more popular frameworks?__"

    There are [many solutions for model tracking](https://neptune.ai/blog/best-ml-experiment-tracking-tools).
    While frameworks such as __Weights & Biases__ or __Neptune.ai__
    are extremely rich with features, unfortunately they typically 
    are cloud-based solutions and not necessarily open-sourced.

    Alternative frameworks such as __Tensorboard__ and __MLFlow__
    have only primitive functionalities with respect to AIM stack.

    Aim stack is sitting in the middle of this spectrum:
    It is self-hosted (i.e., no need to push data to the cloud)
    and provides nice data exploration features.

## Runs and campaigns

AIM collects modeling metadata into __repositories__
which are fully controlled by end-users:

* Repositories are not tied to specific projects.
In other words, the end-user can store
in a repository models completely unrelated to each other.

* There is no limit on the amount of repositories 
can be created. 

`tcbench` tracks in an AIM repository two types of tasks,
namely *runs* and *campaigns*:

* A __run__ corresponds to the training of an
individual ML/DL model and is "minimal experiment object" used by AIM,
i.e., any tracked metadata need to be
associated to an AIM run.

* A __campaign__ corresponds to a
collection of runs. 

AIM assign a unique hash code to a run,
but a run object be further enriched with 
extra metadata using AIM SDK or web UI.

A run can be enriched with both individual values
(e.g., best validation loss observed or the final accuracy score)
as well as series (e.g., loss value for each epoch).
Morever, values can have a *context* to further
specify semantic (e.g., define if a registered metric
relates to trainining, validation or test).

While *run* is at term borrowed from AIM terminology,
`tcbench` introduces *campaign* to 
group runs which are semantically related
and need to be summarized together (e.g., results
collected across different train/val/test splits).

It follows that:

* Runs are the fundamental building block for collecting
modeling results. But they are also the fundamental
unit when developing/debugging modeling tasks.

* Campaigns bind multiple runs together. Hence,
are meant to be stored in separate AIM repositories
(although this is NOT a strict requirement for `tcbench`).
