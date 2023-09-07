# Overview

## :material-language-python: Install `tcbench`

First prepare a python virtual environment, for example via :simple-anaconda: conda
```
conda create -n tcbench python=3.10 pip
conda activate tcbench
```

Grab the latest `code_artifacts_paper132.tgz` 
from [:simple-figshare: figshare](https://figshare.com/s/cab23f730cfbc5172f78) and unpack it.
It contains a folder `/code_artifacts_paper132` from which you can trigger the installation.

```
cd code_artifacts_paper132
python -m pip install .
```

All dependecies are automatically installed.

```
tcbench --version
```

!!! note "Output"
    ```
    version: 0.0.16
    ```

## `tcbench` internals

`tcbench` is composed by 3 collections of modules:

* `cli` contains modules for composing
the command line interfaces using
[click](https://click.palletsprojects.com/en/8.1.x/) and 
[rich](https://github.com/Textualize/rich).

* `libtcdatasets` contains modules for 
datasets curation.

* `modeling` contains modules for DL/ML modeling.

### `cli`

The entry point of `tcbench` is the script
`tcbench/cli/main.py`: This is the root of the
[click](https://click.palletsprojects.com/en/8.1.x/) prompt.

The different sub-commands are organized in separate modules

* `cli.command_aimrepo` for the `aimrepo` sub-command.
* `cli.command_campaign` for the `campaign` sub-command.
* `cli.command_datasets` for the `datasets` sub-command.
* `cli.command_singlerun` for the `run` sub-command.

while `cli.clickutils` and `cli.richutils` collects
some utility functions used for formatting the CLI
and its output.

### `libtcdatasets`

A module `datasets_utils` contains general utility function
used across the other modules in this group.

The remaining are pairs of modules each associated to 
a different datasets.

* `XYZ_to_parquet` modules convert the original raw data into the curated
format 

* `XYZ_generate_splits` modules split the data into train/val/test splits.

For instance, for `ucdavis-icdm19` the two modules are
`ucdavis_icdm19_csv_to_parquet` and `ucdavis_icdm19_generate_splits`.
Please refer to [datasets install](/datasets/install) for more details
about this pre-processing steps.

These module pairs are designed to be also usable from the command line.
As a matter of fact, the curation triggered from the `tcbench` CLI
is just a wrapper around the lower level command line parser.

For instance
```
cd code_artifacts_paper132
python tcbench/libtcdatasets/ucdavis_icdm19_csv_to_parquet.py --help
```

!!! note "Output"
	```
	usage: ucdavis_icdm19_csv_to_parquet.py [-h] --input-folder INPUT_FOLDER
											[--output-folder OUTPUT_FOLDER]
											[--num-workers NUM_WORKERS]

	options:
	  -h, --help            show this help message and exit
	  --input-folder INPUT_FOLDER, -i INPUT_FOLDER
							Root folder of UCDavis dataset
	  --output-folder OUTPUT_FOLDER, -o OUTPUT_FOLDER
							Folder where to save output parquet files (default:
							datasets/ucdavis-icdm19)
	  --num-workers NUM_WORKERS, -w NUM_WORKERS
							Number of workers for parallel loading (default: 4)
	```

!!! tip "Important"

	We discourage the direct use of these lower level modules in favor
    of the global `tcbench` CLI which automatically handles
    configurations so to have a uniform installation across datasets.


### `modeling`

These modules in this group handle DL/ML model training.

* `modeling.aimutils` collects utility functions related to AIM repositories.
* `modeling.augmentation` collects classes and functions related to data augmentation.
* `modeling.backbone` collects classes and functions for DL architectures.
* `modeling.dataprep` collects classes and functions related to data loading and preparation.
* `modeling.losses` collectsion functions for SimCLR losses.
* `modeling.methods` collects classes handling training.
* `modeling.utils` collects complementary utility functions.

These modules are "glued" together into two sub-group utilities

* `run_<XYZ>` are modules triggering [runs](/modeling/runs)
* `run_campaign_<XYZ>` are modules tringgering [campaigns](/modeling/campaigns)

Both these module types work also as script and can be invoked on the command line.

For instance with the following we can
trigger individual run to investigate
the role of different augmentations.
This is equivalent to use `tcbench run augment-at-loading`.

```
python tcbench/modeling/run_augmentations_at_loading.py --help
```

!!! note "Output"
	```
	usage: run_augmentations_at_loading.py [-h] [--artifacts-folder ARTIFACTS_FOLDER]
										   [--workers WORKERS] [--gpu-index GPU_INDEX]
										   [--aim-repo AIM_REPO]
										   [--aim-experiment-name AIM_EXPERIMENT_NAME]
										   [--final] [--flowpic-dim {32,64,1500}]
										   [--flowpic-block-duration FLOWPIC_BLOCK_DURATION]
										   [--dataset {ucdavis-icdm19,utmobilenet21,mirage19,mirage22}]
										   [--dataset-minpkts {-1,10,100,1000}]
										   [--split-index SPLIT_INDEX]
										   [--max-samples-per-class MAX_SAMPLES_PER_CLASS]
										   [--train-val-split-ratio TRAIN_VAL_SPLIT_RATIO]
										   [--aug-name {noaug,rotate,horizontalflip,colorjitter,packetloss,timeshift,changertt}]
										   [--suppress-val-augmentation] [--seed SEED]
										   [--batch-size BATCH_SIZE]
										   [--patience-steps PATIENCE_STEPS]
										   [--learning-rate LEARNING_RATE] [--epochs EPOCHS]
										   [--suppress-test-train-val-leftover]
										   [--suppress-dropout]

	options:
	  -h, --help            show this help message and exit
	  --artifacts-folder ARTIFACTS_FOLDER
							Artifact folder (default: aim-repo/artifacts)
	  --workers WORKERS     Number of parallel worker for loading the data (default: 20)
	  --gpu-index GPU_INDEX
							The GPU id to use (default: 0)
	  --aim-repo AIM_REPO   Local aim folder or URL of AIM remote server (default: aim-repo)
	  --aim-experiment-name AIM_EXPERIMENT_NAME
							The name of the experiment for AIM tracking (default: augmentation-
							at-loading)
	  --final
	  --flowpic-dim {32,64,1500}
							Flowpic dimension (default: 32)
	  --flowpic-block-duration FLOWPIC_BLOCK_DURATION
							Flowpic block duration (in seconds) (default: 15)
	  --dataset {ucdavis-icdm19,utmobilenet21,mirage19,mirage22}
							Dataset to use for modeling (default: ucdavis-icdm19)
	  --dataset-minpkts {-1,10,100,1000}
							When used in combination with --dataset can refine the dataset and
							split to use for modeling (default: -1)
	  --split-index SPLIT_INDEX
							Datasplit index (default: 0)
	  --max-samples-per-class MAX_SAMPLES_PER_CLASS
							Activated when --split-index is -1 to define how many samples to
							select for train+val (with a 80/20 split between train and val
							(default: -1)
	  --train-val-split-ratio TRAIN_VAL_SPLIT_RATIO
							Training train/val split (default: 0.8)
	  --aug-name {noaug,rotate,horizontalflip,colorjitter,packetloss,timeshift,changertt}
							Augmentation policy (default: noaug)
	  --suppress-val-augmentation
							Do not augment validation set (default: False)
	  --seed SEED           Random seed (default: 12345)
	  --batch-size BATCH_SIZE
							Training batch size (default: 64)
	  --patience-steps PATIENCE_STEPS
	  --learning-rate LEARNING_RATE
							Traning learning rate (default: 0.001)
	  --epochs EPOCHS       Number of epochs for training (default: 50)
	  --suppress-test-train-val-leftover
							Skip test on leftover split (default: False)
	  --suppress-dropout    Mask dropout layers with Identity (default: False)
	```
