# Contrastive Learning and Data Augmentation in Traffic Classification, IMC23

This work investigates the role of data
augmentation by using both supervised
and contrastive learning techniques
across [4 datasets](datasets/install), namely
[`ucdavis-icdm19`](/datasets/install/ucdavis-icdm19), 
[`mirage19`](/datasets/install/mirage19), 
[`mirage22`](/datasets/install/mirage22) and 
[`utmobilenet21`](/datasets/install/utmobilenet21).

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


## Scope of the study

This paper replicates and reproduces the following paper
from the IMC22 program

=== "Bibtex"
    ```
    @inproceedings{10.1145/3517745.3561436,
    author = {
        Horowicz, Eyal and 
        Shapira, Tal and 
        Shavitt, Yuval
    },
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

=== "Abstract"
    Internet traffic classification has been intensively studied over the past decade due to its importance for traffic engineering and cyber security. One of the best solutions to several traffic classification problems is the FlowPic approach, where histograms of packet sizes in consecutive time slices are transformed into a picture that is fed into a Convolution Neural Network (CNN) model for classification. However, CNNs (and the FlowPic approach included) require a relatively large labeled flow dataset, which is not always easy to obtain. In this paper, we show that we can overcome this obstacle by replacing the large labeled dataset with a few samples of each class and by using augmentations in order to inflate the number of training samples. We show that common picture augmentation techniques can help, but accuracy improves further when introducing augmentation techniques that mimic network behavior such as changes in the RTT. Finally, we show that we can replace the large FlowPics suggested in the past with much smaller mini-FlowPics and achieve two advantages: improved model performance and easier engineering. Interestingly, this even improves accuracy in some cases.


which

* Considers a traffic classification use-case modeled
with a CNN-based network using a flowpic input representation
(a 2d summary of traffic flow dynamics)

* Compares a supervised setting against a contrastive learning 
+ fine-tuning setting, both in a few shot scenario
(training with 100 input samples).

* Benchmarks 6 types of data augmentations (3 on time series
and 3 on images) against training with no data augmentation.

In our paper we replicate and reproduce the IMC22 paper results
and expand the previous analysis to more datasets.

## Takeaways

1. We were able only to partially reproduce the results from the IMC22 paper.
   Specifically, we found a 20% accuracy gap with respect to the IMC22
   paper which relates to a data shift in the [`ucdavis-icdm19`](/datasets/install/ucdavis-icdm19) dataset
   previously undetected in the reference paper.

2. Simply using the [`ucdavis-icdm19`](/datasets/install/ucdavis-icdm19), and differently
   from the IMC22 paper, we do not find statistical significance differences across 
   the 6 augmentations under analysis.

3. Contrastive learning can help to "bootstrap" a model in an unsupervised fashion. Yet,
   relying on more samples (than the 100 required as from the IMC22 paper modeling scenarios)
   is beneficial to boost performance, i.e., augmentations are not perfect replacement for 
   real input samples.
       
4. Using multiple datasets (namely [`mirage19`](/datasets/install/mirage19), 
   [`mirage22`](/datasets/install/mirage22) and 
   [`utmobilenet21`](/datasets/install/utmobilenet21) allowed to confirm 
   *Change RTT* and *Time Shift* augmentations (used in the IMC22 paper) 
   as superior with respect to alternatives on a flowpic input representation.
   Yet, the augmentations are not statistically different from each other.


