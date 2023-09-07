# Figure 8: Investigating root cause of G1 discrepancies: Kernel density estimation of the per-class packet size distributions.

[:simple-jupyter: :material-download:](../../paper_tables_and_figures/figure8_ucdavis_kde_on_pkts_size/figure8_ucdavis_kde_on_pkts_size.ipynb)


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
REPLACE = {
    "google-doc": "G. Doc",
    "google-drive": "G. Drive",
    "google-music": "G. Music",
    "google-search": "G. Search",
    "youtube": "YouTube",
    "retraining-human-triggered": "Human",
    "retraining-script-triggered": "Script",
}

dset.df = dset.df.assign(
    app = dset.df["app"].replace(REPLACE),
    partition = dset.df["partition"].replace(REPLACE)
)
```


```python
TARGETS_LABEL = sorted(dset.df["app"].unique())
PARTITIONS_NAME = sorted(dset.df["partition"].unique())
```


```python
all_pkts_size = dict()

for partition_name in PARTITIONS_NAME:
    all_pkts_size[partition_name] = dict()

    for app in TARGETS_LABEL:
        df_tmp = dset.df[
            (dset.df["partition"] == partition_name) & (dset.df["app"] == app)
        ]

        l = []
        for idx in df_tmp.index:
            ser = df_tmp.loc[idx]
            indexes = np.where(ser["timetofirst"] < FLOWPIC_BLOCK_DURATION)[0]
            pkts_size = ser["pkts_size"][indexes]
            l.append(pkts_size)
        all_pkts_size[partition_name][app] = np.concatenate(l)
```


```python
# WARNING: computing the KDE will take a few minutes

fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 5))

line_props = {
    "pretraining": dict(linestyle="-"),
    "Script": dict(
        linestyle=(0, (1, 1))
    ), 
    "Human": dict(linestyle=(0, (1, 1))),
}

for ax, app in zip(axes, TARGETS_LABEL):
    for partition_name in [
        "pretraining",
        "Script",
        "Human",
    ]:
        props = line_props[partition_name]
        sns.kdeplot(
            ax=ax,
            data=all_pkts_size[partition_name][app],
            linewidth=2,
            label=partition_name,
            **props,
            fill=True,
            alpha=0.1
        )
    ax.legend(bbox_to_anchor=(0.5, 1.5), loc="upper center")
    ax.set_title(app, fontsize=10)
    ax.set_xlim((-500, 1800))
    ax.set_xlabel("packet size")
    ax.set_ylabel("kde")

plt.tight_layout()
plt.savefig("ucdavid-icdm19_kde_pkts_size.png", dpi=300, bbox_inches='tight')
```


    
![png](figure8_ucdavis_kde_on_pkts_size_files/figure8_ucdavis_kde_on_pkts_size_10_0.png)
    

