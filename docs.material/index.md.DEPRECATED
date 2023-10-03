This website documents code and data artifacts related to the IMC23 submission #132 titled

!!! quote ""
    __Contrastive Learning and Data Augmentation in Traffic Classification via a Flowpic Representation__
    *Replicating and Reproducing “A Few Shots Traffic Classification with mini-FlowPic Augmentations”
    from IMC’22*

Our submission investigates the role of data
augmentation by using both supervised
and contrastive learning techniques
across [4 datasets](datasets/install).

It replicates and reproduces the following paper
from the IMC22 program


```
@inproceedings{10.1145/3517745.3561436,
author = {Horowicz, Eyal and Shapira, Tal and Shavitt, Yuval},
title = {A Few Shots Traffic Classification with Mini-FlowPic Augmentations},
year = {2022},
isbn = {9781450392594},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3517745.3561436},
doi = {10.1145/3517745.3561436},
booktitle = {Proceedings of the 22nd ACM Internet Measurement Conference},
pages = {647–654},
numpages = {8},
location = {Nice, France},
series = {IMC '22}
}
```

We adopt the same traffic representation used in :material-file-document-outline:`imc22-paper`,
namely a Flowpic -- a summarization of the packet size time series of a flow by means of 
frequency histograms extracted from consecutive time windows of the flow -- 
applied on the [`ucdavis-icdm19`](datasets/#ucdavis-icdm19).

In the first part of the submission we investigate how augmentations
affect classification performance -- the study considers 3 image transformations (*rotation, 
color jitter, horizontal flip*) and 3 time series transformations (*time shift, packet drop, change rtt*)
applied to packets timestamps -- when used either in a fully supervised setting or via
contrastive learning.

!!! info "Key takeaways from reproducibility"
    1. We can only partially reproduce the results from :material-file-document-outline:`imc22-paper` on [`ucdavis-icdm19`](datasets/#ucdavis-icdm19).
       Specifically, we uncover a data shift present in the dataset itself which justifies our results; 
       yet, we cannot comment on why this was not detected in :material-file-document-outline:`imc22-paper`.

    2. Simply based on the [`ucdavis-icdm19`](datasets/#ucdavis-icdm19) dataset, and differently
       from the argumentation presented in :material-file-document-outline:`imc22-paper`, 
       we do not find statistical significance differences across the different augmentations.

    3. Contrastive learning can help to "bootstrap" a model in an unsupervised fashion, yet
       relying on more samples is beneficial to boost performance.
       
Then, in the second part of the submission we replicate the 
analysis testing the same 6 augmentations across 3 other datasets.

!!! info "Key takeaways from replicability"
    Using multiple datasets allow to confirm the argument of the  :material-file-document-outline:`imc22-paper`, i.e.,
    *Change RTT* augmentation used in [`ucdavis-icdm19`](datasets/#ucdavis-icdm19)
    is superior to the alternative transformations presented in the paper.


## Website conventions

* :material-file-document-outline:`imc22-paper` is used to the reference the replicated/reproduced paper.

* WIP (Work in progress) and :construction: suggest documentation that is incomplete or not yet available.

* :material-link-off: suggests a link is expected to be added but is not yet available.
