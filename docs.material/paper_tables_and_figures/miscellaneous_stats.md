# Miscellaneous stats across the paper

[:simple-jupyter: :material-download:](../../paper_tables_and_figures/miscellaneous_stats/miscellaneous_stats.ipynb)


```python
import json
import pathlib

import numpy as np
import pandas as pd
import xgboost as xgb
from tcbench.modeling import backbone
```

# Section 3

## total number of campaigns


```python
def find_artifacts_folder(folder):
    if folder.name == "artifacts":
        return [folder]

    res = []
    for item in folder.iterdir():
        if item.is_dir():
            res += find_artifacts_folder(item)
    return res
            
# "campaigns/mirage19/augmentation-at-loading-no-dropout/minpkts10/
```


```python
folders = find_artifacts_folder(pathlib.Path("./campaigns/"))
len(folders)
```




    13



## total number of runs


```python
sum([len(list(path.iterdir())) for path in folders])
```




    2760



# Section 4

### Average depth of xgboost models


```python
class Node:
    def __init__(self, node_id, left=None, right=None):
        self.node_id = node_id
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.left is None and self.right is None
```


```python
def build_graph(tree_data):
    df_tmp = tree_data.fillna(-1)
    nodes = {node_id: Node(node_id) for node_id in df_tmp["ID"]}
    nodes[-1] = None

    for idx, node in zip(range(len(df_tmp)), nodes.values()):
        left_id, right_id = df_tmp.iloc[idx][["Yes", "No"]]
        node.left = nodes.get(left_id, None)
        node.right = nodes.get(right_id, None)

    return next(iter(nodes.values()))
```


```python
def _graph_max_depth(node, depth=0):
    if node.is_leaf():
        return depth
    return max(
        _graph_max_depth(node.left, depth + 1), _graph_max_depth(node.right, depth + 1)
    )
```


```python
def _tree_max_depth(df_tree):
    root = build_graph(df_tree)
    return _graph_max_depth(root)
```


```python
def trees_avg_depth(fname):
    xgb_model = backbone.xgboost_factory().xgb_model
    xgb_model.load_model(fname)
    booster_data = xgb_model.get_booster().trees_to_dataframe()
    return booster_data.groupby("Tree").apply(_tree_max_depth).mean()
```


```python
folder = pathlib.Path(
    "campaigns/ucdavis-icdm19/xgboost/noaugmentation-timeseries/artifacts/"
)

np.array([trees_avg_depth(fname) for fname in folder.glob("*/*.json")]).mean()
```




    1.6982666666666666




```python
folder = pathlib.Path(
    "campaigns/ucdavis-icdm19/xgboost/noaugmentation-flowpic/artifacts/"
)

np.array([trees_avg_depth(fname) for fname in folder.glob("*/*.json")]).mean()
```




    1.3896



# Section 4

# average experiment duration


```python
folder = pathlib.Path(
    "campaigns/ucdavis-icdm19/augmentation-at-loading-with-dropout/campaign_summary/augment-at-loading-with-dropout/"
)
```


```python
# all test splits are evaluated at the same time
# so it's enough to check one of them
runs_1500 = pd.read_parquet(folder / "runsinfo_flowpic_dim_1500.parquet")
runs_1500[runs_1500["test_split_name"] == "test-script"]["run_duration"].mean()
```




    1512.8632845379057




```python
runs_32 = pd.read_parquet(folder / "runsinfo_flowpic_dim_32.parquet")
runs_32[runs_32["test_split_name"] == "test-script"]["run_duration"].mean()
```




    55.191846643175396




```python
runs_64 = pd.read_parquet(folder / "runsinfo_flowpic_dim_64.parquet")
runs_64[runs_64["test_split_name"] == "test-script"]["run_duration"].mean()
```




    70.5957797731672



# number of samples when doing a 80/20 train/val split based on all samples available


```python
folder = pathlib.Path(
    "campaigns/ucdavis-icdm19/larger-trainset/augmentation-at-loading"
)
```


```python
# this is reported in the logs so we can simply check one run
# that does not have any augmentation

runs = pd.read_parquet(
    folder
    / "campaign_summary/augment-at-loading-larger-trainset/runsinfo_flowpic_dim_32.parquet"
)
```


```python
run_hash = runs[runs["aug_name"] == "noaug"]["hash"].values[0]
```


```python
fname_log = folder / "artifacts" / run_hash / "log.txt"
fname_log.read_text().splitlines()[:32]
```




    ['',
     'connecting to AIM repo at: /mnt/storage/finamore/imc23-submission/camera-ready/campaigns/ucdavis-icdm19/augment-at-loading_larger-trainset/__staging__/netml05_gpu0',
     'created aim run hash=d0af742e1b0846169452b04a',
     'artifacts folder at: /mnt/storage/finamore/imc23-submission/camera-ready/campaigns/ucdavis-icdm19/augment-at-loading_larger-trainset/__staging__/netml05_gpu0/artifacts/d0af742e1b0846169452b04a',
     'WARNING: the artifact folder is not a subfolder of the AIM repo',
     '--- run hparams ---',
     'flowpic_dim: 32',
     'flowpic_block_duration: 15',
     'split_index: -1',
     'max_samples_per_class: -1',
     'aug_name: noaug',
     'patience_steps: 5',
     'suppress_val_augmentation: False',
     'dataset: ucdavis-icdm19',
     'dataset_minpkts: -1',
     'seed: 25',
     'with_dropout: False',
     'campaign_id: augment-at-loading-larger-trainset',
     'campaign_exp_idx: 20',
     '-------------------',
     'loaded: /opt/anaconda/anaconda3/envs/super-tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19/preprocessed/ucdavis-icdm19.parquet',
     'no augmentation',
     'no augmentation',
     'dataset samples count',
     '               train  val',
     'app                      ',
     'google-search   1532  383',
     'google-drive    1307  327',
     'google-doc       976  245',
     'youtube          861  216',
     'google-music     473  119',
     '']



## comparing SimCLR results between 100 samples and larger training


```python
df_100_samples = pd.read_csv(
    "campaigns/ucdavis-icdm19/simclr-dropout-and-projection/campaign_summary/simclr-dropout-and-projection/summary_flowpic_dim_32.csv",
    header = [0, 1],
    index_col = [0, 1, 2]
)
```


```python
ser_100samples = df_100_samples["acc"].xs(30, level=1, axis=0).xs(False, level=1, axis=0)["mean"]
ser_100samples
```




    test-human     74.690909
    test-script    92.184000
    Name: mean, dtype: float64




```python
df_largerdataset = pd.read_csv(
    "campaigns/ucdavis-icdm19/larger-trainset/simclr/campaign_summary/simclr-larger-trainset/summary_flowpic_dim_32.csv",
    header = [0, 1],
    index_col = [0, 1]
)
```


```python
ser_largerdataset = df_largerdataset["acc"]["mean"].droplevel(1, axis=0)
```


```python
ser_largerdataset
```




    test-human     80.454545
    test-script    93.900000
    Name: mean, dtype: float64




```python
(ser_largerdataset - ser_100samples).round(2)
```




    test-human     5.76
    test-script    1.72
    Name: mean, dtype: float64



## min and max from Table 3


```python
df_script = pd.read_csv(
    "table3_ucdavis-icdm19_comparing_data_augmentations_functions_test_on_script.csv",
    header=[0, 1, 2],
    index_col=[0],
)

df_human = pd.read_csv(
    "table3_ucdavis-icdm19_comparing_data_augmentations_functions_test_on_human.csv",
    header=[0, 1, 2],
    index_col=[0],
)
```


```python
ser_script = df_script["ours"]["32"]["mean"].drop("mean_diff", axis=0)
ser_script.name = "script"

ser_human = df_human["ours"]["32"]["mean"].drop("mean_diff", axis=0)
ser_human.name = "human"

df_tmp = pd.concat((ser_script, ser_human), axis=1)
df_tmp.max() - df_tmp.min()
```




    script    2.09
    human     3.22
    dtype: float64



## min and max from Table 8


```python
df_others = pd.read_csv(
    "table8_augmentation-at-loading_on_other_datasets.csv", header=[0, 1], index_col=[0]
)
df_tmp = df_others.xs("mean", level=1, axis=1)
df_tmp.max() - df_tmp.min()
```




    mirage22 - minpkts10          5.50
    mirage22 - minpkts1000       10.08
    utmobilenet21 - minpkts10     4.15
    mirage19 - minpkts10         13.93
    dtype: float64


