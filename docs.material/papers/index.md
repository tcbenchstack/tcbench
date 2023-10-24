# Research articles featuring tcbench

[__Replication: Contrastive Learning and Data Augmentation in Traffic Classification__](/tcbench/papers/imc23)
<br>
*A. Finamore, C. Wang, J. Krolikowki, J. M. Navarro, F. Cheng, D. Rossi*, 
<br> ACM Internet Measurement Conference (IMC), 2023
<br> [:material-hexagon-outline: __Artifacts__](/tcbench/papers/imc23/artifacts) [:fontawesome-regular-file-pdf: __PDF__](https://arxiv.org/pdf/2309.09733)

=== "Bibtex"
    ```
    @misc{finamore2023contrastive,
      title={
        Contrastive Learning and Data Augmentation 
        in Traffic Classification Using a 
        Flowpic Input Representation
      }, 
      author={
        Alessandro Finamore and 
        Chao Wang and 
        Jonatan Krolikowski 
        and Jose M. Navarro 
        and Fuxing Chen and 
        Dario Rossi
      },
      year={2023},
      eprint={2309.09733},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
    }
    ```

=== "Abstract"
    Over the last years we witnessed a renewed interest towards
    Traffic Classification (TC) captivated by the rise of Deep
    Learning (DL). Yet, the vast majority of TC literature lacks
    code artifacts, performance assessments across datasets and
    reference comparisons against Machine Learning (ML) meth-
    ods. Among those works, a recent study from IMC'22 [17] is
    worth of attention since it adopts recent DL methodologies
    (namely, few-shot learning, self-sup ervision via contrastive
    learning and data augmentation) appealing for networking as
    they enable to learn from a few samples and transfer across
    datasets. The main result of [17] on the UCDAVIS19, ISCX-VPN
    and ISCX-Tor datasets is that, with such DL methodologies,
    100 input samples are enough to achieve very high accuracy
    using an input representation called "flowpic" (i.e., a per-flow
    2d histograms of the packets size evolution over time).
    In this paper (i) we rep roduce [17] on the same datasets
    and (ii) we rep licate its most salient aspect (the importance
    of data augmentation) on three additional public datasets,
    MIRAGE-19, MIRAGE-22 and UTMOBILENET21. While we con-
    firm most of the original results, we also found a 20% ac-
    curacy drop on some of the investigated scenarios due to
    a data shift of the original dataset that we uncovered. Ad-
    ditionally, our study validates that the data augmentation
    strategies studied in [17] perform well on other datasets too.
    In the spirit of reproducibility and replicability we make all
    artifacts (code and data) available at [10].
