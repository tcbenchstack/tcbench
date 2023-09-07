# Tables and Figures from the submission

The tables and figures are created via a set of :simple-jupyter: Jupyter notebooks.
Specifically, these notebooks are located at `code_artifacts_paper132/notebooks/submission_tables_and_figures` 
in the code artifacts.

However, notice that they require from modeling artifacts and datasets installation.

* To install modeling artifacts, grab `ml_artifacts.tgz` and unpack it under the 
folder mentioned above. The tarball contains a `/campaigns` folder so the final
structure should be
```
tree notebooks/ -d -L 2
notebooks/
├── submission_tables_and_figures
│   └── campaigns
└── tutorials
```

* To install/import datasets refer to the [`install`](../../datasets/install) and [`import`](../../datasets/import) pages.

## Tables

- __Table 1__: Summary of Datasets Properties. 
<br>[:simple-jupyter: `table1_datasets_properties.ipynb`](../../paper_tables_and_figures/table1_datasets_properties/)

- __Table 2__:  (G0) Baseline ML performance without augmentation in a supervised setting.
<br>[:simple-jupyter: `table2_xgboost_baseline.ipynb`](../../paper_tables_and_figures/table2_xgboost_baseline/)

- __Table 3__: Comparing data augmentation functions applied in supervised training. 
<br>[:simple-jupyter: `table3_ucdavis-icdm19_comparing_data_augmentations_functions.ipynb`](../../paper_tables_and_figures/table3_ucdavis-icdm19_comparing_data_augmentations_functions/)

- __Table 4__:  Impact of dropout and SimCLR projection layer dimension on fine-tuning.
<br>[:simple-jupyter: `table4_simclr_dropout_and_projectionlayer.ipynb`](../../paper_tables_and_figures/table4_simclr_dropout_and_projectionlayer/)

- __Table 5__: Comparing the fine-tuning performance when using different pairs of augmentation for pretraining.
<br>[:simple-jupyter: `table5_simclr_other_augmentation_pairs.ipynb`](../../paper_tables_and_figures/table5_simclr_other_augmentation_pairs/)

- __Table 6__: Accuracy on 32x32 flowpic when enlarging training set (w/o Dropout).
<br>[:simple-jupyter: `table6_simclr_larger_trainset.ipynb`](../../paper_tables_and_figures/table6_simclr_larger_trainset/)

- __Table 7__: (G3) Data augmentation in supervised setting on other datasets. 
<br>[:simple-jupyter: `table7_augmentation-at-loading_on_other_datasets.ipynb`](../../paper_tables_and_figures/table7_augmentation-at-loading_on_other_datasets/)

- __Table 8__ - *appendix*: Macro-average Accuracy with different retraining dataset and different sampling methods.
<br>[:simple-jupyter: `table8_icdm_finetuning_per_class_metrics_on_human.ipynb`](../../paper_tables_and_figures/table8_icdm_finetuning_per_class_metrics_on_human/)

- __Table 9__ - *appendix*: Performance comparison across augmentations for different flowpic sizes.
<br>[:simple-jupyter: `table9_ucdavis-icdm19_tukey.ipynb`](../../paper_tables_and_figures/table9_ucdavis-icdm19_tukey/)


## Figures

- __Figure 1__: Average confusion matrixes for the 32x32 resolution across all experiments in Table 3. 
<br>[:simple-jupyter: `figure1_confusion_matrix_supervised_setting.ipynb`](../../paper_tables_and_figures/figure1_confusion_matrix_supervised_setting/)

- __Figure 2__: Average 32x32 flowpic for each class across multiple data splits. 
<br>[:simple-jupyter: `figure2_ucdavis_per_class_average_flowpic`](../../paper_tables_and_figures/figure2_ucdavis_per_class_average_flowpic/)

- __Figure 3__: Critical distance plot of the accuracy obtained with each augmentation for the 32x32 and 64x64 cases.
<br>[:simple-jupyter: `figure3_ucdavis_augmentations_comparison`](../../paper_tables_and_figures/figure3_ucdavis_augmentations_comparison/)

- __Figure 4__: Critical distance plot of the accuracy obtained with each augmentation across the four tested datasets.
<br>[:simple-jupyter: `figure4_augmentations_comparison_across_datasets_critical_distance`](../../paper_tables_and_figures/figure4_augmentations_comparison_across_datasets_critical_distance/)

- __Figure 5__: Average rank obtained per augmentation and dataset. Ranks closer to 1 indicate a better performance.
<br>[:simple-jupyter: `figure5_augmentations_comparison_across_datasets_average_rank`](../../paper_tables_and_figures/figure5_augmentations_comparison_across_datasets_average_rank/)

- __Figure 6__ - *appendix*: Investigating root cause of G1 discrepancies: Kernel density estimation of the per-class packet size distributions.
<br>[:simple-jupyter: `figure6_ucdavis_per_class_average_flowpic`](../../paper_tables_and_figures/figure6_ucdavis_kde_on_pkts_size/)

- __Figure 7__ - *appendix*: Accuracy on script with different sampling methods (borrowed from [Rezaei at al. ICM19](https://arxiv.org/abs/1812.09761) and added extra annotations).

- __Figure 8(b)__ - *appendix*: Classwise evaluation on `human`.
<br> [:simple-jupyter: `figure8b_icdm_finetuning_per_class_metrics_on_human`](../../paper_tables_and_figures/figure8b_icdm_finetuning_per_class_metrics_on_human/)

- __Figure 9__ - *appendix*: Accuracy difference w/ and w/o Dropout in supervised learning.
<br> [:simple-jupyter: `figure9_dropout_impact_supervised_setting.ipynb`](../../paper_tables_and_figures/figure9_dropout_impact_supervised_setting/)

## :material-bug: Errata corrige

When compiling this documentation we noticed
the following mistake/typos with respect to the results reported by the submission.

1. In Table 1, the row `mirage-22` @ 10pkts report wrong values. Please refer
to [the related notebook](../../paper_tables_and_figures/table1_datasets_properties/).

2. In Table 4, the CI have (minor) differences with respect to the notebook.

3. In Table 8, the CI were wrongly reported (the original computation was not right).
