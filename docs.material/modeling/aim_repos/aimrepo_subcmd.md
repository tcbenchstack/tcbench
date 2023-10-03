---
title: aimrepo subcmd
icon: octicons/terminal-16
---

# Repository reports

`tcbench` offers the `aimrepo` subcommand to 
interact with the content of a repository.

```
tcbench aimrepo --help
```

!!! note "Output"
    ```
     Usage: tcbench aimrepo [OPTIONS] COMMAND [ARGS]...

     Investigate AIM repository content.

     ╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
     │ --help      Show this message and exit.                                                          │
     ╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
     ╭─ Commands ───────────────────────────────────────────────────────────────────────────────────────╮
     │ ls                List a subset of properties of each run.                                       │
     │ merge             Coalesce different AIM repos into a single new repo.                           │
     │ properties        List properties across all runs.                                               │
     │ report            Summarize runs performance metrics.                                            │
     ╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
    ```

In the following we illustrate each
sub-command using the 
[`ucdavis-icdm19/augmentation-at-loading-with-dropout`](/modeling/campaigns/#ucdavis-icdm19augmentation-at-loading-with-dropout)
repository from our [IMC23 paper](/papers/imc23).


## Listing runs hash

The `ls` sub-command simply list 
the hash, creation time and end time of each run.

```
 tcbench aimrepo ls \
	--aim-repo notebooks/imc23/campaigns/ucdavis-icdm19/augmentation-at-loading-with-dropout \
    | head -5
```

!!! note "Output"
	```
	hash                      creation_time      end_time
	9cf03ef2a61848e8975ea90c  1693512039.103155  1693513226.327002
	3a5fd371e8064c4a89b1df6a  1693510803.708324  1693512038.63982
	a845811424ee4e13a454ab79  1693509166.65158   1693510144.254626
	c1ecfb414b8f4f809c4eb142  1693508952.200609  1693510803.343346
	```

## Listing properties 

The `properties` sub-command shows
aggregate information and the list of
unique values of the properties of an AIM repo
(i.e., run hyper params and other meta data).


```
tcbench aimrepo properties \
	--aim-repo notebooks/imc23/campaigns/ucdavis-icdm19/augmentation-at-loading-with-dropout \
```

!!! note "Output"
	```
	╭───────────────────────────┬────────────┬─────────────────────────────────────────────────────────╮
	│ Name                      │ No. unique │ Value                                                   │
	├───────────────────────────┼────────────┼─────────────────────────────────────────────────────────┤
	│ runs                      │          - │ 315                                                     │
	│ run duration (mean ± std) │          - │ 9m6s ± 21m3s                                            │
	│ metrics                   │          7 │ ['acc', 'best_epoch', 'best_loss', 'f1', 'loss',        │
	│                           │            │ 'precision', 'recall']                                  │
	│ contexts                  │          5 │ ['test-human', 'test-script',                           │
	│                           │            │ 'test-train-val-leftover', 'train', 'val']              │
	├───────────────────────────┼────────────┼─────────────────────────────────────────────────────────┤
	│ experiment                │          1 │ ['augmentation-at-loading']                             │
	│ aug_name                  │          7 │ ['changertt', 'colorjitter', 'horizontalflip', 'noaug', │
	│                           │            │ 'packetloss', 'rotate', 'timeshift']                    │
	│ campaign_exp_idx          │        105 │ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, │
	│                           │            │ 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, │
	│                           │            │ 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, │
	│                           │            │ 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, │
	│                           │            │ 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, │
	│                           │            │ 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, │
	│                           │            │ 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,     │
	│                           │            │ 100, 101, 102, 103, 104, 105]                           │
	│ campaign_id               │          1 │ ['augment-at-loading-with-dropout']                     │
	│ dataset                   │          1 │ ['ucdavis-icdm19']                                      │
	│ dataset_minpkts           │          1 │ [-1]                                                    │
	│ flowpic_block_duration    │          1 │ [15]                                                    │
	│ flowpic_dim               │          3 │ [32, 64, 1500]                                          │
	│ max_samples_per_class     │          1 │ [-1]                                                    │
	│ patience_steps            │          1 │ [5]                                                     │
	│ seed                      │          3 │ [42, 666, 12345]                                        │
	│ split_index               │          5 │ [0, 1, 2, 3, 4]                                         │
	│ suppress_val_augmentation │          1 │ [False]                                                 │
	│ with_dropout              │          1 │ [True]                                                  │
	╰───────────────────────────┴────────────┴─────────────────────────────────────────────────────────╯
	```

The table is split into two parts to separate general properties (top)
from hyper parameters (bottom). General properties are common across repositories
while hyper parameters vary depending of the campaign.

Considering the general properties

* *runs* indicates the total number of runs in the repository (i.e., 315 = 3 seeds * 3 resolutions * 5 splits * 7 augmentations).

* *run duration* indicates the average duration of each run.

* *metrics* lists which metrics are tracked for each run.

* *context* is term borrored from AIM terminology and refers to ability to 
bind a metric to multiple semantic. For instance, in this repository
the metrics are bounded to the data used to measure them.


## Summary reports 

The `report` sub-command provides a summary
across metrics grouped by properties.

```
tcbench aimrepo report \
    --aim-repo notebooks/imc23/campaigns/ucdavis-icdm19/augmentation-at-loading-with-dropout/
```


!!! note "Output"
    ```
	campaign_id: augment-at-loading-with-dropout
	runs: 315

										hparams                           acc                 run_duration
					   split        aug_name   flowpic_dim  runs   mean    std   ci95     mean       std      ci95
	 ───────────────────────  ────────────────────────────  ────  ───────────────────  ───────────────────────────
				  test-human       changertt            32    15  70.76    3.6   1.99    71.32      8.52      4.72
														64    15  71.49   2.87   1.59    76.95      7.93      4.39
													  1500    15  71.97   1.96   1.08  2308.22   1921.22   1063.94
							  ────────────────────────────  ────  ───────────────────  ───────────────────────────
								 colorjitter            32    15  68.43    5.1   2.82    59.65     11.83      6.55
														64    15   70.2    3.6   1.99    72.37     11.55       6.4
													  1500    15  69.08   3.11   1.72  2017.51   2215.73   1227.03
							  ────────────────────────────  ────  ───────────────────  ───────────────────────────
							  horizontalflip            32    15   69.4   2.94   1.63    42.27      3.93      2.18
														64    15  70.52   3.67   2.03    57.49      3.44       1.9
													  1500    15   73.9   1.91   1.06   501.25    105.56     58.46
							  ────────────────────────────  ────  ───────────────────  ───────────────────────────
									   noaug            32    15  68.84   2.61   1.45    25.48      2.37      1.31
														64    15  69.08   2.44   1.35    34.87      1.57      0.87
													  1500    15  69.32   2.95   1.63   353.54    106.06     58.73
							  ────────────────────────────  ────  ───────────────────  ───────────────────────────
								  packetloss            32    15  70.68   2.44   1.35    60.18      8.05      4.46
														64    15  71.33   2.62   1.45    72.91      7.44      4.12
													  1500    15  71.08   2.04   1.13  1072.35    273.49    151.45
							  ────────────────────────────  ────  ───────────────────  ───────────────────────────
									  rotate            32    15  71.65   3.58   1.98    65.27     13.04      7.22
														64    15  71.08   2.73   1.51   104.31      7.95       4.4
													  1500    15  68.19   1.75   0.97   1375.8    617.11    341.74
							  ────────────────────────────  ────  ───────────────────  ───────────────────────────
								   timeshift            32    15  70.36   2.94   1.63    62.17     10.46      5.79
														64    15  71.89   2.87   1.59    75.28      8.35      4.62
													  1500    15  71.08   2.41   1.33  2961.36   3200.62   1772.44
	 ───────────────────────  ────────────────────────────  ────  ───────────────────  ───────────────────────────
				 test-script       changertt            32    15  97.29   0.64   0.35    71.32      8.52      4.72
														64    15  97.02   0.83   0.46    76.95      7.93      4.39
													  1500    15  96.93   0.55   0.31  2308.22   1921.22   1063.94
							  ────────────────────────────  ────  ───────────────────  ───────────────────────────
								 colorjitter            32    15  97.56    1.0   0.55    59.65     11.83      6.55
														64    15  97.16   1.11   0.62    72.37     11.55       6.4
													  1500    15  94.93   1.23   0.68  2017.51   2215.73   1227.03
							  ────────────────────────────  ────  ───────────────────  ───────────────────────────
							  horizontalflip            32    15  95.47    0.8   0.45    42.27      3.93      2.18
														64    15   96.0   1.07   0.59    57.49      3.44       1.9
													  1500    15  94.89   1.42   0.79   501.25    105.56     58.46
							  ────────────────────────────  ────  ───────────────────  ───────────────────────────
									   noaug            32    15  95.64   0.66   0.37    25.48      2.37      1.31
														64    15  95.87   0.52   0.29    34.87      1.57      0.87
													  1500    15  94.93    1.3   0.72   353.54    106.06     58.73
							  ────────────────────────────  ────  ───────────────────  ───────────────────────────
								  packetloss            32    15  96.89   0.93   0.52    60.18      8.05      4.46
														64    15  96.84   1.14   0.63    72.91      7.44      4.12
													  1500    15  95.96   0.92   0.51  1072.35    273.49    151.45
							  ────────────────────────────  ────  ───────────────────  ───────────────────────────
									  rotate            32    15  96.31   0.79   0.44    65.27     13.04      7.22
														64    15  96.93   0.83   0.46   104.31      7.95       4.4
													  1500    15  95.69   0.71   0.39   1375.8    617.11    341.74
							  ────────────────────────────  ────  ───────────────────  ───────────────────────────
								   timeshift            32    15  96.71   1.08    0.6    62.17     10.46      5.79
														64    15  97.16   0.89   0.49    75.28      8.35      4.62
													  1500    15  96.89   0.48   0.27  2961.36   3200.62   1772.44
	 test-train-val-leftover       changertt            32    15  98.38   0.32   0.18    71.32      8.52      4.72
														64    15  97.97   0.71   0.39    76.95      7.93      4.39
													  1500    15  98.19   0.39   0.22  2308.22   1921.22   1063.94
							  ────────────────────────────  ────  ───────────────────  ───────────────────────────
								 colorjitter            32    15  96.93   1.01   0.56    59.65     11.83      6.55
														64    15  96.46   0.82   0.46    72.37     11.55       6.4
													  1500    15  95.47   0.88   0.49  2017.51   2215.73   1227.03
							  ────────────────────────────  ────  ───────────────────  ───────────────────────────
							  horizontalflip            32    15  95.68   0.72    0.4    42.27      3.93      2.18
														64    15  96.32   1.06   0.59    57.49      3.44       1.9
													  1500    15  95.97   1.45    0.8   501.25    105.56     58.46
							  ────────────────────────────  ────  ───────────────────  ───────────────────────────
									   noaug            32    15  95.78   0.53   0.29    25.48      2.37      1.31
														64    15  96.09   0.68   0.38    34.87      1.57      0.87
													  1500    15  95.79   0.92   0.51   353.54    106.06     58.73
							  ────────────────────────────  ────  ───────────────────  ───────────────────────────
								  packetloss            32    15  96.99    0.7   0.39    60.18      8.05      4.46
														64    15  97.25    0.7   0.39    72.91      7.44      4.12
													  1500    15  96.84   0.89   0.49  1072.35    273.49    151.45
							  ────────────────────────────  ────  ───────────────────  ───────────────────────────
									  rotate            32    15  96.74   0.63   0.35    65.27     13.04      7.22
														64    15   97.0   0.69   0.38   104.31      7.95       4.4
													  1500    15  95.79   0.56   0.31   1375.8    617.11    341.74
							  ────────────────────────────  ────  ───────────────────  ───────────────────────────
								   timeshift            32    15  97.02    0.9    0.5    62.17     10.46      5.79
														64    15  97.51   0.83   0.46    75.28      8.35      4.62
													  1500    15  97.67   0.53   0.29  2961.36   3200.62   1772.44

	saving: notebooks/imc23/campaigns/ucdavis-icdm19/augmentation-at-loading-with-dropout/campaign_summary/augment-at-loading-with-dropout/runsinfo_flowpic_dim_1500.parquet
	saving: notebooks/imc23/campaigns/ucdavis-icdm19/augmentation-at-loading-with-dropout/campaign_summary/augment-at-loading-with-dropout/summary_flowpic_dim_1500.csv
	saving: notebooks/imc23/campaigns/ucdavis-icdm19/augmentation-at-loading-with-dropout/campaign_summary/augment-at-loading-with-dropout/runsinfo_flowpic_dim_64.parquet
	saving: notebooks/imc23/campaigns/ucdavis-icdm19/augmentation-at-loading-with-dropout/campaign_summary/augment-at-loading-with-dropout/summary_flowpic_dim_64.csv
	saving: notebooks/imc23/campaigns/ucdavis-icdm19/augmentation-at-loading-with-dropout/campaign_summary/augment-at-loading-with-dropout/runsinfo_flowpic_dim_32.parquet
	saving: notebooks/imc23/campaigns/ucdavis-icdm19/augmentation-at-loading-with-dropout/campaign_summary/augment-at-loading-with-dropout/summary_flowpic_dim_32.csv
    ```

Starting from the top, the console output
is informing about the `campaign_id` 
and the number of runs in the repository.

Then it follows a table grouped by the 
`test-<XYZ>` contexts found.
By default all hyper parameters with more than one value
are also added (see the previous description about properties).

In the example, the `acc` metric is measured with mean, standard
deviation and 95th %tile confidence intervals.
Next to the metric is reported also `run_duration` which 
corresponds to the overall time for train/validation/test 
in the run execution (hence values looks duplicated
across different partitions of the table).

The example above show the default configuration of th
report. You can use 

* The `--groupby` option to change the order of the grouping
(e.g., swap augmentation and resolution).

* The `--contexts` option to add/remove context (e.g., 
add training and validation).

* The `--metrics` option allows to specify which metrics
to use (e.g., use f1 rather than accuracy).

```
tcbench aimrepo report \
    --aim-repo notebooks/imc23/campaigns/ucdavis-icdm19/augmentation-at-loading-with-dropout/ \
    --groupby flowpic_dim,aug_name \
    --contexts test-human
	--metrics acc,f1
```

!!! note "Output"
    ```
    campaign_id: 1684447037
    runs: 315

                           hparams                           acc                  duration
          split  flowpic_dim         aug_name  runs   mean    std   ci95     mean      std     ci95
     ──────────  ────────────────────────────  ────  ───────────────────  ─────────────────────────
     test-human           32        changertt    15  70.04   4.41   2.44    83.93      7.8     4.32
                                  colorjitter    15  68.84   4.69   2.59    78.01     10.9     6.03
                               horizontalflip    15   69.8   2.51   1.39     56.9     1.64     0.91
                                        noaug    15  69.48   2.12   1.17    37.97     1.46     0.81
                                   packetloss    15   71.0   1.85   1.02    80.83    11.11     6.15
                                       rotate    15  71.57   3.52   1.95    78.52     11.1     6.15
                                    timeshift    15  70.36   2.98   1.65    80.08    12.72     7.04
                 ────────────────────────────  ────  ───────────────────  ─────────────────────────
                          64        changertt    15  72.05    2.1   1.16    61.57      5.2     2.88
                                  colorjitter    15  71.33   3.35   1.86    67.15    22.49    12.45
                               horizontalflip    15  70.92   3.31   1.83    63.86    28.97    16.04
                                        noaug    15  69.88   2.28   1.26    38.06    20.21    11.19
                                   packetloss    15  73.17   1.61   0.89    57.08     4.75     2.63
                                       rotate    15   71.0   2.43   1.35    88.49    33.41     18.5
                                    timeshift    15  72.53   1.83   1.02    64.23    21.91    12.13
                 ────────────────────────────  ────  ───────────────────  ─────────────────────────
                        1500        changertt    15  72.69   2.68   1.48  1303.72   336.55   186.37
                                  colorjitter    15  68.59   3.17   1.76   765.43   152.53    84.47
                               horizontalflip    15  73.82   1.47   0.82    471.8     58.6    32.45
                                        noaug    15  68.67   1.93   1.07   374.88    94.41    52.28
                                   packetloss    15  72.13   1.87   1.04  1164.89   245.69   136.06
                                       rotate    15  67.87   1.56   0.86  1313.58    301.7   167.08
                                    timeshift    15  70.84   2.42   1.34   907.03   165.48    91.64

    saving: campaing_summary/1684447037/runsinfo_flowpic_dim_1500.parquet
    saving: campaing_summary/1684447037/summary_flowpic_dim_1500.csv
    saving: campaing_summary/1684447037/runsinfo_flowpic_dim_32.parquet
    saving: campaing_summary/1684447037/summary_flowpic_dim_32.csv
    saving: campaing_summary/1684447037/runsinfo_flowpic_dim_64.parquet
    saving: campaing_summary/1684447037/summary_flowpic_dim_64.csv
    ```

The `report` sub-command also creates output artifacts.

* An output folder is created based on the `campaign_id` value.

* A set of `runinfo_<XYZ>.parquet` files collect runs
    hyper param and metrics. 

* A set of `summary_<XYZ>.csv` files collect the
    aggregate table reported on the console.


## Merge repositories

Currently tcbench does not have a scheduler able to 
automatically distribute runs of a campaign across
servers or GPUs.

You can however split your workload across multiple
repositories and then use the `aimrepo merge` subcommand
to consolidate all repositories into a single one.

```
tcbench aimrepo merge --help
```

!!! note "Output"
```
	 Usage: tcbench aimrepo merge [OPTIONS]

	 Coalesce different AIM repos into a single new repo.

	╭─ Options ──────────────────────────────────────────────────────────────────╮
	│ *  --src     PATH  AIM repository to merge. [required]                     │
	│    --dst     PATH  New AIM repository to create. [default: aim-repo]       │
	│    --help          Show this message and exit.                             │
	╰────────────────────────────────────────────────────────────────────────────╯
```

If the destionation repository does not exists, it will be created.
When using merge, the recommendation is to create a new destination
(so that the primary source is always available in case something goes wrong).

