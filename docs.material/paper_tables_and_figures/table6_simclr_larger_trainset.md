# Table 6: Accuracy on 32x32 flowpic when enlargin training set (w/o Dropout)

[:simple-jupyter: :material-download:](../../paper_tables_and_figures/table6_simclr_larger_trainset/table6_simclr_larger_trainset.ipynb)


```python
import os
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.stats.api as sms
import yaml
from aim import Repo
```


```python
def compute_confidence_intervals(array, alpha=0.05):
    array = np.array(array)
    low, high = sms.DescrStatsW(array).tconfint_mean(alpha)
    mean = array.mean()
    ci = high - mean
    return ci
```


```python
campaign_folder = Path(
    "./campaigns/ucdavis-icdm19/larger-trainset/augmentation-at-loading"
)

repo = Repo.from_path(str(campaign_folder))

ct_df = pd.DataFrame(
    columns=[
        "hash_id",
        "contrastive_learning_seed",
        "finetune_seed",
        "suppress_dropout",
        "test_set",
        "test_acc",
    ]
)
sup_df = pd.DataFrame(
    columns=["hash_id", "aug_name", "seed", "suppress_dropout", "test_set", "test_acc"]
)


ct_nodropout_hashes = []
sup_nodropout_hashes = []
for hash_id in list(repo._all_run_hashes()):
    path = campaign_folder / "artifacts" / hash_id / "params.yml"
    if os.path.isfile(path):
        conf = yaml.safe_load(path.read_text())

        suppress_dropout = conf["suppress_dropout"]

        human_acc = (
            pd.read_csv(
                campaign_folder / "artifacts" / hash_id / "test-human_class_rep.csv"
            )
            .iloc[5]
            .values[1]
        )

        script_acc = (
            pd.read_csv(
                campaign_folder / "artifacts" / hash_id / "test-script_class_rep.csv"
            )
            .iloc[5]
            .values[1]
        )

        if conf["aim_experiment_name"] == "contrastive-learning-and-finetune":
            if suppress_dropout:
                ct_nodropout_hashes.append(hash_id)

                contrastive_learning_seed = conf["contrastive_learning_seed"]
                finetune_seed = conf["finetune_seed"]

                new_row_human = {
                    "hash_id": hash_id,
                    "contrastive_learning_seed": contrastive_learning_seed,
                    "finetune_seed": finetune_seed,
                    "suppress_dropout": suppress_dropout,
                    "test_set": "human",
                    "test_acc": human_acc * 100,
                }
                new_row_script = {
                    "hash_id": hash_id,
                    "contrastive_learning_seed": contrastive_learning_seed,
                    "finetune_seed": finetune_seed,
                    "suppress_dropout": suppress_dropout,
                    "test_set": "script",
                    "test_acc": script_acc * 100,
                }

                ct_df.loc[len(ct_df)] = new_row_human
                ct_df.loc[len(ct_df)] = new_row_script

        elif conf["aim_experiment_name"] == "augmentation-at-loading":
            if suppress_dropout:
                sup_nodropout_hashes.append(hash_id)

                seed = conf["seed"]

                aug_name = conf["aug_name"]

                new_row_human = {
                    "hash_id": hash_id,
                    "aug_name": aug_name,
                    "seed": seed,
                    "suppress_dropout": suppress_dropout,
                    "test_set": "human",
                    "test_acc": human_acc * 100,
                }
                new_row_script = {
                    "hash_id": hash_id,
                    "aug_name": aug_name,
                    "seed": seed,
                    "suppress_dropout": suppress_dropout,
                    "test_set": "script",
                    "test_acc": script_acc * 100,
                }

                sup_df.loc[len(sup_df)] = new_row_human
                sup_df.loc[len(sup_df)] = new_row_script

        else:  # aim runs rm
            print(1, hash_id)

    else:  # aim runs rm
        print(2, hash_id)
```


```python
sup_df_merged = sup_df.groupby(
    [
        "suppress_dropout",
        "aug_name",
        "test_set",
    ]
).agg({"test_acc": ["mean", compute_confidence_intervals]})
sup_df_merged = sup_df_merged.rename(columns={"compute_confidence_intervals": "ci95"})
sup_df_merged = sup_df_merged.droplevel(0, axis=1)

sup_df_merged = sup_df_merged.droplevel(0).reset_index()
```


```python
sup_df_merged = sup_df_merged.pivot(
    index="aug_name", columns="test_set", values=["mean", "ci95"]
)
sup_df_merged = sup_df_merged[
    [("mean", "script"), ("ci95", "script"), ("mean", "human"), ("ci95", "human")]
]
sup_df_merged.columns = pd.MultiIndex.from_tuples(
    tpl[::-1] for tpl in sup_df_merged.columns
)
```


```python
AUGMENTATIONS_ORDER = [
    "noaug",
    "rotate",
    "horizontalflip",
    "colorjitter",
    "packetloss",
    "timeshift",
    "changertt",
]

RENAME = {
    "noaug": "No augmentation",
    "changertt": "Change rtt",
    "horizontalflip": "Horizontal flip",
    "colorjitter": "Color jitter",
    "packetloss": "Packet loss",
    "rotate": "Rotate",
    "timeshift": "Time shift",
}

print("=== supervised augmentation (at loading) ===")
sup_df_merged.loc[AUGMENTATIONS_ORDER].rename(RENAME).round(2)
```

    === supervised augmentation (at loading) ===





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">script</th>
      <th colspan="2" halign="left">human</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>ci95</th>
      <th>mean</th>
      <th>ci95</th>
    </tr>
    <tr>
      <th>aug_name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No augmentation</th>
      <td>98.40</td>
      <td>0.21</td>
      <td>73.31</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>Rotate</th>
      <td>98.53</td>
      <td>0.13</td>
      <td>74.16</td>
      <td>1.22</td>
    </tr>
    <tr>
      <th>Horizontal flip</th>
      <td>98.20</td>
      <td>0.18</td>
      <td>74.52</td>
      <td>1.15</td>
    </tr>
    <tr>
      <th>Color jitter</th>
      <td>98.43</td>
      <td>0.32</td>
      <td>72.71</td>
      <td>0.64</td>
    </tr>
    <tr>
      <th>Packet loss</th>
      <td>98.70</td>
      <td>0.19</td>
      <td>72.89</td>
      <td>0.59</td>
    </tr>
    <tr>
      <th>Time shift</th>
      <td>98.70</td>
      <td>0.26</td>
      <td>72.71</td>
      <td>0.97</td>
    </tr>
    <tr>
      <th>Change rtt</th>
      <td>98.53</td>
      <td>0.13</td>
      <td>71.45</td>
      <td>1.04</td>
    </tr>
  </tbody>
</table>
</div>




```python
ct_df_merged = ct_df.groupby(["suppress_dropout", "test_set"]).agg(
    {"test_acc": ["mean", compute_confidence_intervals]}
)
ct_df_merged = ct_df_merged.rename(columns={"compute_confidence_intervals": "ci95"})
ct_df_merged = ct_df_merged.droplevel(0, axis=1).droplevel(0)

print("=== using simclr and finetuning ===")
ct_df_merged.loc[["script", "human"]].round(2)
```

    === using simclr and finetuning ===





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>ci95</th>
    </tr>
    <tr>
      <th>test_set</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>script</th>
      <td>94.10</td>
      <td>0.48</td>
    </tr>
    <tr>
      <th>human</th>
      <td>80.61</td>
      <td>2.96</td>
    </tr>
  </tbody>
</table>
</div>


