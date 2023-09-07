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

- __Table 2__: Summary of Datasets Properties. 
<br>[:simple-jupyter: `table2_datasets_properties.ipynb`](../../paper_tables_and_figures/table2_datasets_properties/)

- __Table 3__:  (G0) Baseline ML performance without augmentation in a supervised setting.
<br>[:simple-jupyter: `table3_xgboost_baseline.ipynb`](../../paper_tables_and_figures/table3_xgboost_baseline/)

- __Table 4__: Comparing data augmentation functions applied in supervised training. 
<br>[:simple-jupyter: `table4_ucdavis-icdm19_comparing_data_augmentations_functions.ipynb`](../../paper_tables_and_figures/table4_ucdavis-icdm19_comparing_data_augmentations_functions/)

- __Table 5__:  Impact of dropout and SimCLR projection layer dimension on fine-tuning.
<br>[:simple-jupyter: `table5_simclr_dropout_and_projectionlayer.ipynb`](../../paper_tables_and_figures/table5_simclr_dropout_and_projectionlayer/)

- __Table 6__: Comparing the fine-tuning performance when using different pairs of augmentation for pretraining.
<br>[:simple-jupyter: `table6_simclr_other_augmentation_pairs.ipynb`](../../paper_tables_and_figures/table6_simclr_other_augmentation_pairs/)

- __Table 7__: Accuracy on 32x32 flowpic when enlarging training set (w/o Dropout).
<br>[:simple-jupyter: `table7_larger_trainset.ipynb`](../../paper_tables_and_figures/table7_larger_trainset/)

- __Table 8__: (G3) Data augmentation in supervised setting on other datasets. 
<br>[:simple-jupyter: `table8_augmentation-at-loading_on_other_datasets.ipynb`](../../paper_tables_and_figures/table8_augmentation-at-loading_on_other_datasets/)

- __Table 9__ - *appendix*: Macro-average Accuracy with different retraining dataset and different sampling methods for [Rezaei at al. ICM19](https://arxiv.org/abs/1812.09761).
<br>[:simple-jupyter: `table9_icdm_finetuning_per_class_metrics_on_human.ipynb`](../../paper_tables_and_figures/table9_icdm_finetuning_per_class_metrics_on_human/)

- __Table 10__ - *appendix*: Performance comparison across augmentations for different flowpic sizes.
<br>[:simple-jupyter: `table10_ucdavis-icdm19_tukey.ipynb`](../../paper_tables_and_figures/table10_ucdavis-icdm19_tukey/)


## Figures

- __Figure 3__: Average confusion matrixes for the 32x32 resolution across all experiments in Table 4. 
<br>[:simple-jupyter: `figure3_confusion_matrix_supervised_setting.ipynb`](../../paper_tables_and_figures/figure3_confusion_matrix_supervised_setting/)

- __Figure 4__: Average 32x32 flowpic for each class across multiple data splits. 
<br>[:simple-jupyter: `figure4_ucdavis_per_class_average_flowpic`](../../paper_tables_and_figures/figure4_ucdavis_per_class_average_flowpic/)

- __Figure 5__: Critical distance plot of the accuracy obtained with each augmentation for the 32x32 and 64x64 cases.
<br>[:simple-jupyter: `figure5_ucdavis_augmentations_comparison`](../../paper_tables_and_figures/figure5_ucdavis_augmentations_comparison/)

- __Figure 6__: Critical distance plot of the accuracy obtained with each augmentation across the four tested datasets.
<br>[:simple-jupyter: `figure6_augmentations_comparison_across_datasets_critical_distance`](../../paper_tables_and_figures/figure6_augmentations_comparison_across_datasets_critical_distance/)

- __Figure 7__: Average rank obtained per augmentation and dataset. Ranks closer to 1 indicate a better performance.
<br>[:simple-jupyter: `figure7_augmentations_comparison_across_datasets_average_rank`](../../paper_tables_and_figures/figure7_augmentations_comparison_across_datasets_average_rank/)

- __Figure 8__ - *appendix*: Investigating root cause of G1 discrepancies: Kernel density estimation of the per-class packet size distributions.
<br>[:simple-jupyter: `figure8_ucdavis_per_class_average_flowpic`](../../paper_tables_and_figures/figure8_ucdavis_kde_on_pkts_size/)

- __Figure 10(b)__ - *appendix*: Classwise evaluation on `human`.
<br> [:simple-jupyter: `figure10b_icdm_finetuning_per_class_metrics_on_human`](../../paper_tables_and_figures/figure10b_icdm_finetuning_per_class_metrics_on_human/)

- __Figure 11__ - *appendix*: Accuracy difference w/ and w/o Dropout in supervised learning.
<br> [:simple-jupyter: `figure11_dropout_impact_supervised_setting.ipynb`](../../paper_tables_and_figures/figure11_dropout_impact_supervised_setting/)

## Others

- Miscellaneous stats across the paper.
<br>[:simple-jupyter: `miscellaneous_stats.ipynb`](../../paper_tables_and_figures/miscellaneous_stats/)
