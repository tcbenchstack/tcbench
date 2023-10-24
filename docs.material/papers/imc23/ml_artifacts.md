---
title: ML artifacts
icon: simple/docsdotrs
---

The ML artifacts correspond to
a collection of 13 modeling campaigns.

```
<root>
├── mirage19
│   └── augmentation-at-loading-no-dropout
│       └── minpkts10
│           ├── .aim
│           ├── artifacts
│           └── campaign_summary
├── mirage22
│   └── augmentation-at-loading-no-dropout
│       ├── minpkts10
│       │   ├── .aim
│       │   ├── artifacts
│       │   └── campaign_summary
│       └── minpkts1000
│           ├── .aim
│           ├── artifacts
│           └── campaign_summary
├── ucdavis-icdm19
│   ├── augmentation-at-loading-dropout-impact
│   │   ├── .aim
│   │   ├── artifacts
│   │   └── campaign_summary
│   ├── augmentation-at-loading-with-dropout
│   │   ├── .aim
│   │   ├── artifacts
│   │   └── campaign_summary
│   ├── larger-trainset
│   │   ├── augmentation-at-loading
│   │   │   ├── .aim
│   │   │   ├── artifacts
│   │   │   └── campaign_summary
│   │   └── simclr
│   │       ├── .aim
│   │       ├── artifacts
│   │       └── campaign_summary
│   ├── simclr-dropout-and-projection
│   │   ├── .aim
│   │   ├── artifacts
│   │   └── campaign_summary
│   ├── simclr-other-augmentation-pairs
│   │   ├── .aim
│   │   ├── artifacts
│   │   └── campaign_summary
│   └── xgboost
│       ├── noaugmentation-flowpic
│       │   ├── .aim
│       │   ├── artifacts
│       │   └── campaign_summary
│       └── noaugmentation-timeseries
│           ├── .aim
│           ├── artifacts
│           └── campaign_summary
├── ucdavis-icdm19-git-repo-forked
│   └── artifacts
│       ├── FixedStepSampling_Retraining(human-triggered)_10
│       ├── FixedStepSampling_Retraining(script-triggered)_10
│       ├── IncrementalSampling_Retraining(human-triggered)_10
│       ├── IncrementalSampling_Retraining(human-triggered)_20
│       ├── IncrementalSampling_Retraining(script-triggered)_10
│       ├── RandomSampling_Retraining(human-triggered)_10
│       └── RandomSampling_Retraining(script-triggered)_10
└── utmobilenet21
    └── augmentation-at-loading-no-dropout
        ├── minpkts10
        │   ├── .aim
        │   ├── artifacts
        │   └── campaign_summary
        └── minpkts10.STILL-WITH-BUG
            ├── .aim
            ├── artifacts
            └── campaign_summary
```


Each subfolder relates to a different campaign
with some semantic encoded in the folder names themselves.

* Subfolders containing an `.aim/` folder are [AIM repositories](/tcbench/modeling/aim_repos/).

* Subfolders named `artifacts/` collect each [run artifacts](/tcbench/modeling/aim_repos/).

* Subfolders named `campaign_summary/` contains [reports summarizing a campaign](/tcbench/modeling/aim_repos/aimrepo_subcmd/#summary-reports).

The following reference table details how
the different campaigns map to the results in the paper.

##### Mapping campaigns folder to submission results

| Subfolder | Results | CLI trigger |
|:----------|:-------:|:-----------:|
|`ucdavis-icdm19/xgboost/noaugmentation-flowpic`| Table 3 | [:octicons-terminal-24:](#ucdavis-icdm19xgboostnoaugmentation-flowpic) |
|`ucdavis-icdm19/xgboost/noaugmentation-timeseries`| Table 3 | [:octicons-terminal-24:](#ucdavis-icdm19xgboostnoaugmentation-timeseries) |
|`ucdavis-icdm19/augmentation-at-loading-with-dropout`| Table 4<br>Figure 3,5 | [:octicons-terminal-24:](#ucdavis-icdm19augmentation-at-loading-with-dropout) |
|`ucdavis-icdm19/simclr-dropout-and-projection`| Table 5 | [:octicons-terminal-24:](#ucdavis-icdm19simclr-dropout-and-projection)|
|`ucdavis-icdm19/simclr-other-augmentation-pairs`| Table 6 | [:octicons-terminal-24:](#ucdavis-icdm19simclr-other-augmentation-pairs)|
|`ucdavis-icdm19/larger-trainset/augmentation-at-loading`| Table 7 | [:octicons-terminal-24:](#ucdavis-icdm19larger-trainsetaugmentation-at-loading)|
|`ucdavis-icdm19/larger-trainset/simclr`| Table 7 | [:octicons-terminal-24:](#ucdavis-icdm19larger-trainsetsimclr)|
|`mirage19/augmentation-at-loading-no-dropout/minpkts10`| Table 8<br>Figure 6,7 | [:octicons-terminal-24:](#mirage19augmentation-at-loading-no-dropoutminpkts10)|
|`mirage22/augmentation-at-loading-no-dropout/minpkts10`| Table 8<br>Figure 6,7 | [:octicons-terminal-24:](#mirage22augmentation-at-loading-no-dropoutminpkts10)|
|`mirage22/augmentation-at-loading-no-dropout/minpkts1000`| Table 8<br>Figure 6,7 | [:octicons-terminal-24:](#mirage22augmentation-at-loading-no-dropoutminpkts1000)|
|`utmobilenet21/augmentation-at-loading-no-dropout/minpkts10`| Table 8<br>Figure 6,7 | [:octicons-terminal-24:](#utmobilenet21augmentation-at-loading-no-dropoutminpkts10)|
|`ucdavis-icdm19-git-repo-forked`| Table 9<br>Figure 10 |  [:octicons-terminal-24:](/tcbench/modeling/campaigns/#ucdavis-icdm19-git-repo-forked) |
|`ucdavis-icdm19/augmentation-at-loading-dropout-impact`| Figure 11 | [:octicons-terminal-24:](#ucdavis-icdm19augmentation-at-loading-dropout-impact)|


##### ucdavis-icdm19/xgboost/noaugmentation-flowpic [:material-arrow-up:](#mapping-campaigns-folder-to-submission-results)

```bash
REPO=campaigns/ucdavis-icdm19/xgboost/noaugmentation-flowpic

rm -rf $REPO

tcbench campaign augment-at-loading \
    --aim-repo $REPO \
    --artifacts-folder $REPO/artifacts \
    --seeds 12345,42,666 \
    --campaign-id 'noaugmentation-flowpic' \
    --aim-experiment-name 'xgboost-flowpic' \
    --split-indexes 0,1,2,3,4 \
    --dataset ucdavis-icdm19 \
    --method xgboost \
    --input-repr flowpic \
    --flowpic-dims 32 \
    --augmentations noaug

tcbench aimrepo report \
    --aim-repo $REPO
```

!!! note "Campaign dry run"
	```
	##########
	# campaign_id: noaugmentation-flowpic | run 1/15 - time to completion 0:00:00
	##########

	split_indexes (5): [0, 1, 2, 3, 4]
	augmentations (1): ['noaug']
	seeds         (3): [12345, 42, 666]
	flowpic_dims  (1): [32]
	```

##### ucdavis-icdm19/xgboost/noaugmentation-timeseries  [:material-arrow-up:](#mapping-campaigns-folder-to-submission-results)

```bash
REPO=campaigns/ucdavis-icdm19/xgboost/noaugmentation-timeseries

rm -rf $REPO

tcbench campaign augment-at-loading \
    --aim-repo $REPO \
    --artifacts-folder $REPO/artifacts \
    --seeds 12345,42,666 \
    --campaign-id 'noaugmentation-timeseries' \
    --aim-experiment-name 'xgboost-pktseries-10pkts' \
    --split-indexes 0,1,2,3,4 \
    --dataset ucdavis-icdm19 \
    --method xgboost \
    --input-repr pktseries \
    --pktseries-len 10 \
    --augmentations noaug

tcbench aimrepo report \
    --aim-repo $REPO
```

!!! info "Campaign dry run"
	```
	##########
	# campaign_id: noaugmentation-timeseries | run 1/15 - time to completion 0:00:00
	##########

	split_indexes (5): [0, 1, 2, 3, 4]
	augmentations (1): ['noaug']
	seeds         (3): [12345, 42, 666]
	max_n_pkts    (1): [10]
	```

##### ucdavis-icdm19/augmentation-at-loading-with-dropout [:material-arrow-up:](#mapping-campaigns-folder-to-submission-results)

```bash
REPO=campaigns/ucdavis-icdm19/augmentation-at-loading-with-dropout

rm -rf $REPO

tcbench campaign augment-at-loading \
    --aim-repo $REPO \
    --artifacts-folder $REPO/artifacts \
    --seeds 12345,42,666 \
    --campaign-id 'augment-at-loading-with-dropout' \
    --aim-experiment-name 'augment-at-loading' \
    --split-indexes 0,1,2,3,4 \
    --dataset ucdavis-icdm19 \
    --method monolithic \
    --input-repr flowpic \
    --flowpic-dims 32,64,1500

tcbench aimrepo report \
    --aim-repo $REPO
```

!!! note "Campaign dry run"

	```
	##########
	# campaign_id: augment-at-loading-with-dropout | run 1/315 - time to completion 0:00:00
	##########

	split_indexes (5): [0, 1, 2, 3, 4]
	augmentations (7): ['noaug', 'rotate', 'horizontalflip', 'colorjitter', 'packetloss', 'changertt', 'timeshift']
	flowpic_dims  (3): [32, 64, 1500]
	seeds         (3): [12345, 42, 666]
	```


##### ucdavis-icdm19/simclr-dropout-and-projection [:material-arrow-up:](#mapping-campaigns-folder-to-submission-results)

```bash
REPO=campaigns/ucdavis-icdm19/simclr-dropout-and-projection

rm -rf $REPO

tcbench campaign contralearn-and-finetune \
    --aim-repo $REPO \
    --artifacts-folder $REPO/artifacts \
    --campaign-id 'simclr-dropout-and-projection' \
    --flowpic-dims 32 \
    --split-indexes 0,1,2,3,4 \
    --augmentations changertt,timeshift \
    --cl-projection-layer-dims 30,84 \
    --cl-seeds 12345,1,2,3,4 \
    --ft-seeds 12345,1,2,3,4 \
    --dropout enabled,disabled

tcbench aimrepo report \
    --aim-repo $REPO \
	--groupby projection_layer_dim,with_dropout
```

!!! info "Campaign dru run"
	```
	##########
	# campaign_id: simclr-dropout-and-projection | run 1/500 - time to completion 0:00:00
	##########

	split_indexes              (5): [0, 1, 2, 3, 4]
	contrastive learning seeds (5): [12345, 1, 2, 3, 4]
	finetune seeds             (5): [12345, 1, 2, 3, 4]
	projection layer dims      (2): [30, 84]
	dropout                    (2): ['enabled', 'disabled']
	flowpic dims               (1): [32]
	```

##### ucdavis-icdm19/simclr-other-augmentation-pairs [:material-arrow-up:](#mapping-campaigns-folder-to-submission-results)

```bash
REPO=campaigns/ucdavis-icdm19/simclr-other-augmentation-pairs

rm -rf $REPO

for AUGMENTATIONS in \
    "packetloss,colorjitter"\
    "packetloss,rotate"\
    "colorjitter,rotate"\
    "changertt,timeshift"\
    "changertt,rotate"\
    "changertt,colorjitter"; do

	tcbench campaign contralearn-and-finetune \
			--dry-run \
			--aim-repo $REPO \
			--artifacts-folder $REPO/artifacts \
			--campaign-id 'simclr-other-augmentation-pairs' \
			--flowpic-dims 32 \
			--split-indexes 0,1,2,3,4 \
			--augmentations $AUGMENTATIONS \
			--cl-projection-layer-dims 30 \
			--cl-seeds 12345,1,2,3,4 \
			--ft-seeds 12345,1,2,3,4 \
			--dropout disabled

done

tcbench aimrepo report \
    --aim-repo $REPO \
    --groupby augmentations
```

!!! note "Campaign dry run"

	Each of the combination of augmentations has the following grid
	```
	##########
	# campaign_id: simclr-other-augmentation-pairs | run 1/125 - time to completion 0:00:00
	##########

	split_indexes              (5): [0, 1, 2, 3, 4]
	contrastive learning seeds (5): [12345, 1, 2, 3, 4]
	finetune seeds             (5): [12345, 1, 2, 3, 4]
	projection layer dims      (1): [30]
	dropout                    (1): ['disabled']
	flowpic dims               (1): [32]
	```


##### ucdavis-icdm19/larger-trainset/augmentation-at-loading  [:material-arrow-up:](#mapping-campaigns-folder-to-submission-results)

```
REPO=campaigns/ucdavis-icdm19/larger-trainset/augmentation-at-loading
rm -rf $REPO

tcbench campaign augment-at-loading \
    --aim-repo $REPO \
    --artifacts-folder $REPO/artifacts \
    --seeds 6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25 \
    --campaign-id 'augment-at-loading-larger-trainset' \
    --split-indexes -1 \
    --dataset ucdavis-icdm19 \
    --method monolithic \
    --input-repr flowpic \
    --flowpic-dims 32 \
    --augmentations noaug,rotate,horizontalflip,colorjitter,changertt,timeshift,packetloss \
    --no-dropout \
    --no-test-leftover \

tcbench aimrepo report --aim-repo $DST
```

!!! info "Campaign dry run"

	```
	##########
	# campaign_id: augment-at-loading-larger-trainset | run 1/140 - time to completion 0:00:00
	##########

	split_indexes (1): [-1]
	augmentations (7): ['noaug', 'rotate', 'horizontalflip', 'colorjitter', 'changertt', 'timeshift', 'packetloss']
	flowpic_dims  (1): [32]
	seeds         (20): [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
	```

##### ucdavis-icdm19/larger-trainset/simclr  [:material-arrow-up:](#mapping-campaigns-folder-to-submission-results)

```
REPO=campaigns/ucdavis-icdm19/larger-trainset/simclr

rm -rf $REPO

CONTRALEARN_SEEDS=(32 33 34 35 6 7 8 9 10 11 12 13 14 15 16 17 18 20 43 64)
FINETUNE_SEEDS=(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 22 43 64)


for idx in {0..19}; do
    tcbench campaign contralearn-and-finetune \
        --aim-repo $REPO \
        --artifacts-folder $REPO/artifacts \
        --campaign-id 'simclr-larger-trainset' \
        --batch-size 32 \
        --flowpic-dims 32 \
        --dropout disabled \
        --split-indexes -1 \
        --cl-projection-layer-dims 30 \
        --cl-seeds ${CONTRALEARN_SEEDS[$idx]} \
        --ft-seeds ${FINETUNE_SEEDS[$idx]}
done


tcbench aimrepo report \
	--aim-repo $REPO \
	--groupby campaign_id
```

!!! note "Campaign dry run"

	Each of the campaing has only single run with the following grid
	```
	##########
	# campaign_id: simclr-larger-trainset | run 1/1 - time to completion 0:00:00
	##########

	split_indexes              (1): [-1]
	contrastive learning seeds (1): [34]
	finetune seeds             (1): [4]
	projection layer dims      (1): [30]
	dropout                    (1): ['disabled']
	flowpic dims               (1): [32]
	```

##### mirage19/augmentation-at-loading-no-dropout/minpkts10 [:material-arrow-up:](#mapping-campaigns-folder-to-submission-results)

```bash
REPO=campaigns/mirage19/augmentation-at-loading-no-dropout/minpkts10

rm -rf $REPO

tcbench campaign augment-at-loading \
    --aim-repo $REPO \
    --campaign-id 'augment-at-loading' \
    --artifacts-folder $REPO/artifacts \
    --seeds 12345,42,666 \
    --dataset mirage19 \
    --method monolithic \
    --input-repr flowpic \
    --flowpic-dims 32 \
    --dataset-minpkts 10 \
    --no-dropout

tcbench aimrepo report \
	--aim-repo $REPO \
	--metrics acc,f1,precision,recall
```

!!! note "Campaign dry run"
	```
	##########
	# campaign_id: augment-at-loading | run 1/105 - time to completion 0:00:00
	##########

	split_indexes (5): [0, 1, 2, 3, 4]
	augmentations (7): ['noaug', 'rotate', 'horizontalflip', 'colorjitter', 'packetloss', 'changertt', 'timeshift']
	flowpic_dims  (1): [32]
	seeds         (3): [12345, 42, 666]
	```

##### mirage22/augmentation-at-loading-no-dropout/minpkts10 [:material-arrow-up:](#mapping-campaigns-folder-to-submission-results)

```bash
REPO=campaigns/mirage22/augmentation-at-loading-no-dropout/minpkts10

rm -rf $REPO

tcbench campaign augment-at-loading \
    --aim-repo $REPO \
    --campaign-id 'augment-at-loading' \
    --artifacts-folder $REPO/artifacts \
    --seeds 12345,42,666 \
    --dataset mirage22 \
    --method monolithic \
    --input-repr flowpic \
    --flowpic-dims 32 \
    --dataset-minpkts 10 \
    --no-dropout

tcbench aimrepo report \
	--aim-repo $REPO \
	--metrics acc,f1,precision,recall
```

!!! note "Campaign dry run"
	```
	##########
	# campaign_id: augment-at-loading | run 1/105 - time to completion 0:00:00
	##########

	split_indexes (5): [0, 1, 2, 3, 4]
	augmentations (7): ['noaug', 'rotate', 'horizontalflip', 'colorjitter', 'packetloss', 'changertt', 'timeshift']
	flowpic_dims  (1): [32]
	seeds         (3): [12345, 42, 666]
	```

##### mirage22/augmentation-at-loading-no-dropout/minpkts1000 [:material-arrow-up:](#mapping-campaigns-folder-to-submission-results)

```bash
REPO=campaigns/mirage22/augmentation-at-loading-no-dropout/minpkts1000

rm -rf $REPO

tcbench campaign augment-at-loading \
    --aim-repo $REPO \
    --campaign-id 'augment-at-loading' \
    --artifacts-folder $REPO/artifacts \
    --seeds 12345,42,666 \
    --dataset mirage22 \
    --method monolithic \
    --input-repr flowpic \
    --flowpic-dims 32 \
    --dataset-minpkts 1000 \
    --no-dropout

tcbench aimrepo report \
	--aim-repo $REPO \
	--metrics acc,f1,precision,recall
```

!!! note "Campaign dry run"
	```
	##########
	# campaign_id: augment-at-loading | run 1/105 - time to completion 0:00:00
	##########

	split_indexes (5): [0, 1, 2, 3, 4]
	augmentations (7): ['noaug', 'rotate', 'horizontalflip', 'colorjitter', 'packetloss', 'changertt', 'timeshift']
	flowpic_dims  (1): [32]
	seeds         (3): [12345, 42, 666]
	```

##### utmobilenet21/augmentation-at-loading-no-dropout/minpkts10 [:material-arrow-up:](#mapping-campaigns-folder-to-submission-results)

```bash
REPO=campaigns/utmobilenet21/augmentation-at-loading-no-dropout/minpkts10

rm -rf $REPO

tcbench campaign augment-at-loading \
    --aim-repo $REPO \
    --campaign-id 'augment-at-loading' \
    --artifacts-folder $REPO/artifacts \
    --seeds 12345,42,666 \
    --dataset utmobilenet21 \
    --method monolithic \
    --input-repr flowpic \
    --flowpic-dims 32 \
    --dataset-minpkts 10 \
    --no-dropout

tcbench aimrepo report \
	--aim-repo $REPO \
	--metrics acc,f1,precision,recall
```

!!! note "Campaign dry run"
	```
	##########
	# campaign_id: augment-at-loading | run 1/105 - time to completion 0:00:00
	##########

	split_indexes (5): [0, 1, 2, 3, 4]
	augmentations (7): ['noaug', 'rotate', 'horizontalflip', 'colorjitter', 'packetloss', 'changertt', 'timeshift']
	flowpic_dims  (1): [32]
	seeds         (3): [12345, 42, 666]
	```

##### ucdavis-icdm19/augmentation-at-loading-dropout-impact [:material-arrow-up:](#mapping-campaigns-folder-to-submission-results)

```bash
REPO=campaigns/ucdavis-icdm19/augmentation-at-loading-dropout-impact

rm -rf $REPO

tcbench campaign augment-at-loading \
    --aim-repo $REPO \
    --artifacts-folder $REPO/artifacts \
    --seeds 12345,42,666 \
    --campaign-id 'augment-at-loading-dropout-impact' \
    --aim-experiment-name 'augment-at-loading' \
    --split-indexes 0,1,2,3,4 \
    --dataset ucdavis-icdm19 \
    --method monolithic \
    --input-repr flowpic \
    --flowpic-dims 32,1500

tcbench campaign augment-at-loading \
    --aim-repo $REPO \
    --artifacts-folder $REPO/artifacts \
    --seeds 12345,42,666 \
    --campaign-id 'augment-at-loading-dropout-impact' \
    --aim-experiment-name 'augment-at-loading' \
    --split-indexes 0,1,2,3,4 \
    --dataset ucdavis-icdm19 \
    --method monolithic \
    --input-repr flowpic \
    --flowpic-dims 32,1500 \
	--no-dropout

tcbench aimrepo report \
    --aim-repo $REPO \
    --groupby flowpic_dim,with_dropout,aug_name
```

!!! note "Campaign dry run"
	Each of the two campaign has the following grid
	```
	##########
	# campaign_id: augment-at-loading-dropout-impact | run 1/210 - time to completion 0:00:00
	##########

	split_indexes (5): [0, 1, 2, 3, 4]
	augmentations (7): ['noaug', 'rotate', 'horizontalflip', 'colorjitter', 'packetloss', 'changertt', 'timeshift']
	flowpic_dims  (2): [32, 1500]
	seeds         (3): [12345, 42, 666]
	```

##### ucdavis-icdm19-git-repo-forked [:material-arrow-up:](#mapping-campaigns-folder-to-submission-results)

This campaign differ from all the others as it 
repeats the experiments of 
[*Rezaei et al.* ICDM19 paper](https://arxiv.org/abs/1812.09761).
using our version of the [`ucdavis-icdm19`](/tcbench/datasets/install/ucdavis-icdm19) datasets
to validate that our preprocessing does not alter the dataset itself.

To do so, we did minor modification (stored in this
[figshare archive](https://figshare.com/ndownloader/files/42539035)) 
of the [*Rezaei et al.* code base](https://github.com/shrezaei/Semi-supervised-Learning-QUIC-)
without changing how to use it.

In other words, to generate the result run
```
python dataProcessInMemoryQUIC.py 
python pre-training.py
python re-training.py
```

All results are collected in an output folder `/artifacts` as csv files.



