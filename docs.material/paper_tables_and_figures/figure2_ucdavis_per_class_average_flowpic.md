# Figure 2: Average 32x32 flowpic for each class across multiple data splits.

[:simple-jupyter: :material-download:](../../paper_tables_and_figures/figure2_ucdavis_per_class_average_flowpic/figure2_ucdavis_per_class_average_flowpic.ipynb)


```python
import itertools

import numpy as np
import pandas as pd
```


```python
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize

%matplotlib inline
%config InlineBackend.figure_format='retina'
```


```python
import tcbench as tcb
from tcbench import dataprep
```


```python
FLOWPIC_DIM = 32
FLOWPIC_BLOCK_DURATION = 15
```


```python
# load unfiltered dataset
dset = dataprep.FlowpicDataset(
    data=tcb.load_parquet(tcb.DATASETS.UCDAVISICDM19),
    timetofirst_colname="timetofirst",
    pkts_size_colname="pkts_size",
    pkts_dir_colname="pkts_dir",
    target_colname="app",
    flowpic_dim=FLOWPIC_DIM,
    flowpic_block_duration=FLOWPIC_BLOCK_DURATION,
)
```


```python
# load the first train split
dset_train_split = dataprep.FlowpicDataset(
    data=tcb.load_parquet(tcb.DATASETS.UCDAVISICDM19, split=0),
    timetofirst_colname="timetofirst",
    pkts_size_colname="pkts_size",
    pkts_dir_colname="pkts_dir",
    target_colname="app",
    flowpic_dim=FLOWPIC_DIM,
    flowpic_block_duration=FLOWPIC_BLOCK_DURATION,
)
```


```python
def compute_average_flowpic(df):
    # gather all (precomputed) flowpic
    # in a single tensor (n x dim x dim)
    # and do an avereage across the 1st dimension
    mtx = np.stack(df["flowpic"], axis=0).mean(
        axis=0, keepdims=True
    )  # .astype(np.uint8)
    return mtx
```


```python
TARGETS_LABEL = sorted(dset.df["app"].unique())
PARTITIONS_NAME = sorted(dset.df["partition"].unique())

TARGETS_LABEL, PARTITIONS_NAME
```




    (['google-doc', 'google-drive', 'google-music', 'google-search', 'youtube'],
     ['pretraining', 'retraining-human-triggered', 'retraining-script-triggered'])




```python
average_flowpic = dict()
for partition_name in [
    "pretraining",
    "train-split",
    "retraining-script-triggered",
    "retraining-human-triggered",
]:
    if partition_name != "train-split":
        df_partition = dset.df[dset.df["partition"] == partition_name]
    else:
        df_partition = dset_train_split.df

    average_flowpic[partition_name] = dict()

    for target in TARGETS_LABEL:
        df_app = df_partition[df_partition["app"] == target]
        mtx = compute_average_flowpic(df_app).squeeze()
        average_flowpic[partition_name][target] = mtx
```


```python
mtx_min = 100
mtx_max = -100
for partition_name, app in itertools.product(
    [
        "pretraining",
        "train-split",
        "retraining-script-triggered",
        "retraining-human-triggered",
    ],
    TARGETS_LABEL,
):
    mtx = average_flowpic[partition_name][app]
    nonzero = mtx.flatten()
    nonzero = nonzero[nonzero > 0]
    if nonzero.min() < mtx_min:
        mtx_min = nonzero.min()
    if mtx.max() > mtx_max:
        mtx_max = mtx.max()

mtx_min, mtx_max
```




    (0.0005221932114882506, 238.0)




```python
mpl.rcParams.update({"font.size": 10})

fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(8, 6), sharex=True, sharey=True)

cbar_ax = fig.add_axes([0.91, 0.2, 0.03, 0.6])  # (left, bottom, width, height)

for i, ax, (partition_name, app) in zip(
    range(len(axes.flatten())),
    axes.flatten(),
    itertools.product(
        [
            "pretraining",
            "train-split",
            "retraining-script-triggered",
            "retraining-human-triggered",
        ],
        TARGETS_LABEL,
    ),
):
    mtx = average_flowpic[partition_name][app]

    sns.heatmap(
        ax=ax,
        data=np.where(mtx == 0, np.nan, mtx),
        vmin=mtx_min,
        vmax=mtx_max,
        cbar=i == 0, 
        cmap=plt.get_cmap("viridis_r"),
        square=True,
        norm=LogNorm(mtx_min, mtx_max),
        cbar_ax=None if i else cbar_ax,  # <19
    )

    # ax.set_title(target)
    for pos in ("top", "bottom", "right", "left"):
        ax.spines[pos].set_color("lightgray")
        ax.spines[pos].set_visible(True)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_ticks([], None)
    ax.set_ylabel("")

for ax, app in zip(axes[0], TARGETS_LABEL):
    ax.set_title(app, fontsize=10)

for ax, partition_name in zip(
    axes[:, 0], ["pretraining", "pretraining\none split", "script", "human"]
):  
    ax.set_ylabel(partition_name, fontsize=10)

fig.tight_layout(rect=[0, 0, 0.9, 1])

plt.show()
```

    /tmp/ipykernel_30167/2059813758.py:54: UserWarning: This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.
      fig.tight_layout(rect=[0, 0,



    
![png](figure2_ucdavis_per_class_average_flowpic_files/figure2_ucdavis_per_class_average_flowpic_12_1.png)
    

