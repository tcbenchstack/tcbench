## Submission campaigns commands

We report below the command used to trigger the campaigns
collected in the [ML artifacts](/tcbench/modeling/exploring_artifacts/)

#### `ucdavis-icdm19/xgboost/noaugmentation-flowpic`

```
tcbench campaign augment-at-loading \
    --method xgboost \
    --augmentations noaug \
    --input-repr flowpic \
    --flowpic-dims 32,64,1500 \
    --seeds 12345,42,666 \
    --dataset ucdavis-icdm19 \
    --aim-repo ucdavis-icdm19/xgboost/noaugmentation-flowpic
    --artifacts-folder ucdavis-icdm19/xgboost/noaugmentation-flowpic/artifacts
```

!!! info "Runs grid"
    The campaigns has 45 runs

    ```
    split_indexes (5): [0, 1, 2, 3, 4]
    augmentations (1): ['noaug']
    seeds         (3): [12345, 42, 666]
    flowpic_dims  (3): [32, 64, 1500]
    ```

    In the submission we just reported results for flowpic with 32x32 resolution.

#### `ucdavis-icdm19/xgboost/noaugmentation-timeseries`

```
tcbench campaign augment-at-loading \
    --method xgboost \
    --augmentations noaug \
    --input-repr pktseries \
    --pktseries-len 10,30 \
    --seeds 12345,42,666 \
    --dataset ucdavis-icdm19 \
    --aim-repo ucdavis-icdm19/xgboost/noaugmentation-timeseries \
    --artifacts-folder ucdavis-icdm19/xgboost/noaugmentation-timeseries/artifacts
```

!!! info "Runs grid"
    The campaign has 30 runs

	```
	split_indexes (5): [0, 1, 2, 3, 4]
	augmentations (1): ['noaug']
	seeds         (3): [12345, 42, 666]
	max_n_pkts    (2): [10, 30]
	```

    In the submission we just reported results for time series with 10 packets

#### `ucdavis-icdm19/augmentation-at-loading-with-dropout`

```
tcbench campaign augment-at-loading \
    --method monolithic \
    --seeds 12345,42,666 \
    --dataset ucdavis-icdm19 \
    --aim-repo ucdavis-icdm19/augmentation-at-loading-with-dropout \
    --artifacts-folder ucdavis-icdm19/augmentation-at-loading-with-dropout/artifacts
```

!!! info "Runs grid"
    The campaign has 315 runs
	```
	split_indexes (5): [0, 1, 2, 3, 4]
	augmentations (7): ['noaug', 'rotate', 'horizontalflip', 'colorjitter', 'packetloss', 'changertt', 'timeshift']
	flowpic_dims  (3): [32, 64, 1500]
	seeds         (3): [12345, 42, 666]
	```

#### `mirage19/augmentation-at-loading-no-dropout/minpkts10`

```
tcbench campaign augment-at-loading \
    --method monolithic \
    --seeds 12345,42,666 \
    --dataset mirage19 \
    --dataset-minpkts 10 \
    --augmentations noaug \
    --no-dropout \
    --flowpic-dims 32 \
    --aim-repo mirage19/augmentation-at-loading-no-dropout/minpkts10 \
    --artifacts-folder mirage19/augmentation-at-loading-no-dropout/minpkts10/artifacts
```

!!! info "Runs grid"
    The campaign has 15 runs
	```
	split_indexes (5): [0, 1, 2, 3, 4]
	augmentations (1): ['noaug']
	flowpic_dims  (1): [32]
	seeds         (3): [12345, 42, 666]
	```

#### `mirage22/augmentation-at-loading-no-dropout/minpkts10`

```
tcbench campaign augment-at-loading \
    --method monolithic \
    --seeds 12345,42,666 \
    --dataset mirage22 \
    --dataset-minpkts 10 \
    --augmentations noaug \
    --no-dropout \
    --flowpic-dims 32 \
    --aim-repo mirage22/augmentation-at-loading-no-dropout/minpkts10 \
    --artifacts-folder mirage22/augmentation-at-loading-no-dropout/minpkts10/artifacts
```

!!! info "Runs grid"
	The campaign has 15 runs
	```
	split_indexes (5): [0, 1, 2, 3, 4]
	augmentations (1): ['noaug']
	flowpic_dims  (1): [32]
	seeds         (3): [12345, 42, 666]
	```

#### `mirage22/augmentation-at-loading-no-dropout/minpkts1000`

```
tcbench campaign augment-at-loading \
    --method monolithic \
    --seeds 12345,42,666 \
    --dataset mirage22 \
    --dataset-minpkts 10 \
    --augmentations noaug \
    --no-dropout \
    --flowpic-dims 32 \
    --aim-repo mirage22/augmentation-at-loading-no-dropout/minpkts10 \
    --artifacts-folder mirage22/augmentation-at-loading-no-dropout/minpkts10
```

!!! info "Runs grid"
	The campaign has 15 runs
	```
	split_indexes (5): [0, 1, 2, 3, 4]
	augmentations (1): ['noaug']
	flowpic_dims  (1): [32]
	seeds         (3): [12345, 42, 666]
	```

#### `utmobilenet21/augmentation-at-loading-no-dropout/minpkts10`

```
tcbench campaign augment-at-loading \
    --method monolithic \
    --seeds 12345,42,666 \
    --dataset utmobilenet21 \
    --dataset-minpkts 10 \
    --augmentations noaug \
    --no-dropout \
    --flowpic-dims 32 \
    --aim-repo utmobilenet21/augmentation-at-loading-no-dropout/minpkts10 \
    --artifacts-folder utmobilenet21/augmentation-at-loading-no-dropout/minpkts10/artifacts
```

!!! info "Runs grid"
	The campaign has 15 runs
	```
	split_indexes (5): [0, 1, 2, 3, 4]
	augmentations (1): ['noaug']
	flowpic_dims  (1): [32]
	seeds         (3): [12345, 42, 666]
	```

#### `ucdavis-icdm19/simclr-other-augmentation-pairs/colorjitter-changertt`
```
tcbench campaign contralearn-and-finetune \
    --augmentations colorjitter,changertt \
    --flowpic-dims 32 \
    --cl-seeds 12345,1,2,3,4 \
    --ft-seeds 12345,1,2,3,4 \
    --cl-projection-layer-dims 30 \
    --dropout disable \
    --aim-repo ucdavis-icdm19/simclr-other-augmentation-pairs/colorjitter-changertt \
    --artifacts-folder ucdavis-icdm19/simclr-other-augmentation-pairs/colorjitter-changertt/artifacts
```

!!! info "Runs grid"
	The campaign has 125 runs
	```
    split_indexes              (5): [0, 1, 2, 3, 4]
    contrastive learning seeds (5): [12345, 1, 2, 3, 4]
    finetune seeds             (5): [12345, 1, 2, 3, 4]
    projection layer dims      (1): [30]
    dropout                    (1): ['disable']
    flowpic dims               (1): [32]
	```
#### `ucdavis-icdm19/simclr-other-augmentation-pairs/colorjitter-packetloss`
```
tcbench campaign contralearn-and-finetune \
    --augmentations colorjitter,packetloss \
    --flowpic-dims 32 \
    --cl-seeds 12345,1,2,3,4 \
    --ft-seeds 12345,1,2,3,4 \
    --cl-projection-layer-dims 30 \
    --dropout disable \
    --aim-repo ucdavis-icdm19/simclr-other-augmentation-pairs/colorjitter-packetloss \
    --artifacts-folder ucdavis-icdm19/simclr-other-augmentation-pairs/colorjitter-packetloss/artifacts
```

!!! info "Runs grid"
	The campaign has 125 runs
	```
    split_indexes              (5): [0, 1, 2, 3, 4]
    contrastive learning seeds (5): [12345, 1, 2, 3, 4]
    finetune seeds             (5): [12345, 1, 2, 3, 4]
    projection layer dims      (1): [30]
    dropout                    (1): ['disable']
    flowpic dims               (1): [32]
	```

#### `ucdavis-icdm19/simclr-other-augmentation-pairs/colorjitter-rotate`
```
tcbench campaign contralearn-and-finetune \
    --augmentations colorjitter,rotate \
    --flowpic-dims 32 \
    --cl-seeds 12345,1,2,3,4 \
    --ft-seeds 12345,1,2,3,4 \
    --cl-projection-layer-dims 30 \
    --dropout disable \
    --aim-repo ucdavis-icdm19/simclr-other-augmentation-pairs/colorjitter-rotate \
    --artifacts-folder ucdavis-icdm19/simclr-other-augmentation-pairs/colorjitter-rotate/artifacts
```

!!! info "Runs grid"
	The campaign has 125 runs
	```
    split_indexes              (5): [0, 1, 2, 3, 4]
    contrastive learning seeds (5): [12345, 1, 2, 3, 4]
    finetune seeds             (5): [12345, 1, 2, 3, 4]
    projection layer dims      (1): [30]
    dropout                    (1): ['disable']
    flowpic dims               (1): [32]
	```

#### `ucdavis-icdm19/simclr-other-augmentation-pairs/rotate-changertt`
```
tcbench campaign contralearn-and-finetune \
    --augmentations rotate,changertt \
    --flowpic-dims 32 \
    --cl-seeds 12345,1,2,3,4 \
    --ft-seeds 12345,1,2,3,4 \
    --cl-projection-layer-dims 30 \
    --dropout disable \
    --aim-repo ucdavis-icdm19/simclr-other-augmentation-pairs/rotate-changertt \
    --artifacts-folder ucdavis-icdm19/simclr-other-augmentation-pairs/rotate-changertt/artifacts
```

!!! info "Runs grid"
	The campaign has 125 runs
	```
    split_indexes              (5): [0, 1, 2, 3, 4]
    contrastive learning seeds (5): [12345, 1, 2, 3, 4]
    finetune seeds             (5): [12345, 1, 2, 3, 4]
    projection layer dims      (1): [30]
    dropout                    (1): ['disable']
    flowpic dims               (1): [32]
	```

#### `ucdavis-icdm19/simclr-other-augmentation-pairs/rotate-packetloss`
```
tcbench campaign contralearn-and-finetune \
    --augmentations rotate,packetloss \
    --flowpic-dims 32 \
    --cl-seeds 12345,1,2,3,4 \
    --ft-seeds 12345,1,2,3,4 \
    --cl-projection-layer-dims 30 \
    --dropout disable \
    --aim-repo ucdavis-icdm19/simclr-other-augmentation-pairs/rotate-packetloss \
    --artifacts-folder ucdavis-icdm19/simclr-other-augmentation-pairs/rotate-packetloss/artifacts
```

!!! info "Runs grid"
	The campaign has 215 runs
	```
    split_indexes              (5): [0, 1, 2, 3, 4]
    contrastive learning seeds (5): [12345, 1, 2, 3, 4]
    finetune seeds             (5): [12345, 1, 2, 3, 4]
    projection layer dims      (1): [30]
    dropout                    (1): ['disable']
    flowpic dims               (1): [32]
	```

#### `ucdavis-icdm19/simclr-dropout-and-projection`

```
tcbench campaign contralearn-and-finetune \
    --augmentations changertt,timeshift \
    --flowpic-dims 32 \
    --cl-projection-layer-dims 30,84 \
    --cl-seeds 12345,1,2,3,4 \
    --ft-seeds 12345,1,2,3,4 \
    --dropout disable,enable \
    --aim-repo ucdavis-icdm19/simclr-dropout-and-projection \
    --artifacts-folder ucdavis-icdm19/simclr-dropout-and-projection/artifacts
```

!!! info "Run grid"
	The campaign has 500 runs
	```
	split_indexes              (5): [0, 1, 2, 3, 4]
	contrastive learning seeds (5): [12345, 1, 2, 3, 4]
	finetune seeds             (5): [12345, 1, 2, 3, 4]
	projection layer dims      (2): [30, 84]
	dropout                    (2): ['disable', 'enable']
	flowpic dims               (1): [32]
	```

#### `ucdavis-icdm19/augmentation-at-loading-suppress-dropout`
```
tcbench campaign augment-at-loading \
    --flowpic-dims 32,1500 \
    --seeds 12345,42,666 \
    --no-dropout \
    --aim-repo ucdavis-icdm19/simclr-dropout-and-projection \
    --artifacts-folder ucdavis-icdm19/simclr-dropout-and-projection/artifacts
```

!!! info "Run grid"
	The campaign has 210 runs
	```
	split_indexes (5): [0, 1, 2, 3, 4]
	augmentations (7): ['noaug', 'rotate', 'horizontalflip', 'colorjitter', 'packetloss', 'changertt', 'timeshift']
	flowpic_dims  (2): [32, 1500]
	seeds         (3): [12345, 42, 666]
	```


#### `ucdavis-icdm19/larger-trainset/augmentation-at-loading`

This campaign is a composition of two sub-campaigns 
stored in the same AIM repository.

The first is for augmentation at loading
```
tcbench campaign augment-at-loading \
    --seeds 6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25 \
    --split-indexes -1 \
    --method monolithic \
    --no-dropout \
    --flowpic-dims 32 \
    --aim-repo ucdavis-icdm19/larger-trainset/augmentation-at-loading \
    --artifacts-folder ucdavis-icdm19/larger-trainset/augmentation-at-loading/artifacts
```

!!! info "Run grid"
    The campaign has 140 runs
    ```
    split_indexes (1): [-1]
    augmentations (7): ['noaug', 'rotate', 'horizontalflip', 'colorjitter', 'packetloss', 'changertt', 'timeshift']
    flowpic_dims  (1): [32]
    seeds         (20): [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    ```

The second is for contrastive learning created with the following script to
manually compose 20 runs.

```bash
#!/bin/bash

CONTRALEARN_SEEDS=(32 33 34 35 6 7 8 9 10 11 12 13 14 15 16 17 18 20 43 64)
FINETUNE_SEEDS=(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 22 43 64)


for idx in {0..19}; do
    tcbench run contralearn-and-finetune \
        --dataset ucdavis-icdm19 \
        --batch-size 32 \
        --flowpic-dim 32 \
        --no-dropout \
        --split-index -1 \
        --cl-projection-layer-dim 30 \
        --cl-seed ${CONTRALEARN_SEEDS[$idx]} \
        --ft-seed ${FINETUNE_SEEDS[$idx]} \
        --aim-repo ucdavis-icdm19/larger-trainset/augmentation-at-loading \
        --artifacts-folder ucdavis-icdm19/larger-trainset/augmentation-at-loading/artifacts
done
```

#### `ucdavis-icdm19-git-repo-forked`

This campaign is different from all the others
because is just repeating the experiments
of the following paper ICDM19.

```
@misc{rezaei2020achieve,
title={How to Achieve High Classification Accuracy with Just a Few Labels: A Semi-supervised Approach Using Sampled Packets}, 
author={Shahbaz Rezaei and Xin Liu},
year={2020},
eprint={1812.09761},
archivePrefix={arXiv},
primaryClass={cs.NI}
}
```

The related code is available at this 
[:simple-github: repository](https://github.com/shrezaei/Semi-supervised-Learning-QUIC-)

We just did minor modifications (mostly
for changing output folders) without
affecting how to execute the code.

In other words, to generate the result run
```
python dataProcessInMemoryQUIC.py 
python pre-training.py
python re-training.py
```

All results are collected in an output `/artifacts` folder
(but now AIM repository is generated).

