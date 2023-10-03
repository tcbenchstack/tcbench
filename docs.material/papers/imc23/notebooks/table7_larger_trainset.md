
<style>
code.outputcode {
    background-color: white;
    border-left: solid 2px #4051b5;
    line-height:normal;
    font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace;
}
pre.outputcode {
    background-color: white;
    border-left: solid 2px #4051b5;
    line-height:normal;
    font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace;
    padding-left: 15px;
}
.ansi-red-fg {
  color: #e75c58;
}
.ansi-blue-fg {
  color: #208ffb;
}
</style>
# Table 7: Accuracy on 32x32 flowpic when enlargin training set (w/o Dropout)

[:simple-jupyter: :material-download:](../../paper_tables_and_figures/table7_simclr_larger_trainset/table7_simclr_larger_trainset.ipynb)


```python
import itertools
import pathlib

import pandas as pd
```

```python
RENAME = {
    "noaug": "No augmentation",
    "rotate": "Rotate",
    "horizontalflip": "Horizontal flip",
    "colorjitter": "Color jitter",
    "packetloss": "Packet loss",
    "timeshift": "Time shift",
    "changertt": "Change RTT",
}
```

```python
folder = pathlib.Path("campaigns/ucdavis-icdm19/larger-trainset/")
```

```python
df_sup = pd.read_csv(
    folder
    / "augmentation-at-loading/campaign_summary/augment-at-loading-larger-trainset/summary_flowpic_dim_32.csv",
    header=[0, 1],
    index_col=[0, 1],
)
df_sup = df_sup["acc"][["mean", "ci95"]]
df_sup.index.set_names(["test_split_name", "aug_name"], inplace=True)
df_sup = df_sup.reset_index().pivot(
    columns=["test_split_name"], index="aug_name", values=["mean", "ci95"]
)
df_sup.columns.set_names(["stat", "test_split_name"], inplace=True)
df_sup = df_sup.reorder_levels(["test_split_name", "stat"], axis=1)
df_sup = df_sup[
    list(itertools.product(["test-script", "test-human"], ["mean", "ci95"]))
]
df_sup = df_sup.rename(RENAME, axis=0).rename(RENAME, axis=1)
df_sup.index.set_names([""], inplace=True)
df_sup.columns.set_names(["", ""], inplace=True)
df_sup = df_sup.round(2)
df_sup = df_sup.loc[list(RENAME.values())]
df_sup.to_csv("table7_larger-trainset_augment-at-loading.csv")
df_sup
```



<div class="md-typeset__scrollwrap">
<div class="md-typeset__table">
<table>
<thead>
<tr>
<th></th>
<th colspan="2" halign="left">test-script</th>
<th colspan="2" halign="left">test-human</th>
</tr>
<tr>
<th></th>
<th>mean</th>
<th>ci95</th>
<th>mean</th>
<th>ci95</th>
</tr>
<tr>
<th></th>
<th></th>
<th></th>
<th></th>
<th></th>
</tr>
</thead>
<tbody>
<tr>
<th>No augmentation</th>
<td>98.37</td>
<td>0.19</td>
<td>72.95</td>
<td>0.96</td>
</tr>
<tr>
<th>Rotate</th>
<td>98.47</td>
<td>0.25</td>
<td>73.73</td>
<td>1.09</td>
</tr>
<tr>
<th>Horizontal flip</th>
<td>98.20</td>
<td>0.15</td>
<td>74.58</td>
<td>1.16</td>
</tr>
<tr>
<th>Color jitter</th>
<td>98.63</td>
<td>0.21</td>
<td>72.47</td>
<td>1.02</td>
</tr>
<tr>
<th>Packet loss</th>
<td>98.63</td>
<td>0.19</td>
<td>73.43</td>
<td>1.25</td>
</tr>
<tr>
<th>Time shift</th>
<td>98.60</td>
<td>0.22</td>
<td>73.25</td>
<td>1.17</td>
</tr>
<tr>
<th>Change RTT</th>
<td>98.33</td>
<td>0.16</td>
<td>72.47</td>
<td>1.04</td>
</tr>
</tbody>
</table>
</div>
</div>



```python
df_cl = pd.read_csv(
    folder
    / "simclr/campaign_summary/simclr-larger-trainset/summary_flowpic_dim_32.csv",
    header=[0, 1],
    index_col=[0, 1],
)
df_cl = df_cl["acc"][["mean", "ci95"]]
df_cl = df_cl.droplevel(1, axis=0).round(2)
df_cl = df_cl.loc[["test-script", "test-human"]]
df_cl.to_csv("table7_larger-trainset_simclr.csv")
df_cl
```



<div class="md-typeset__scrollwrap">
<div class="md-typeset__table">
<table>
<thead>
<tr style="text-align: right;">
<th></th>
<th>mean</th>
<th>ci95</th>
</tr>
</thead>
<tbody>
<tr>
<th>test-script</th>
<td>93.90</td>
<td>0.74</td>
</tr>
<tr>
<th>test-human</th>
<td>80.45</td>
<td>2.37</td>
</tr>
</tbody>
</table>
</div>
</div>

