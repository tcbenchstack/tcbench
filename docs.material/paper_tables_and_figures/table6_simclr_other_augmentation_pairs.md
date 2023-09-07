# Table 6: Comparing the fine-tuning performance when using different pairs of augmentation for pretraining.

[:simple-jupyter: :material-download:](../../paper_tables_and_figures/table6_simclr_other_augmentation_pairs/table6_simclr_other_augmentation_pairs.ipynb)


```python
import itertools

import pandas as pd
```


```python
RENAME = {
    "colorjitter": "Color jitter",
    "timeshift": "Time shift",
    "changertt": "Change RTT",
    "rotate": "Rotate",
    "packetloss": "Packet loss",
}
```


```python
df = pd.read_csv(
    "./campaigns/ucdavis-icdm19/simclr-other-augmentation-pairs/campaign_summary/simclr-other-augmentation-pairs/summary_flowpic_dim_32.csv",
    header=[0, 1],
    index_col=[0, 1],
)

df = df["acc"][["mean", "ci95"]].round(2)
df = df.reset_index()
df = df.assign(
    aug1=df["level_1"].apply(eval).str[0],
    aug2=df["level_1"].apply(eval).str[1],
)
df = df.drop("level_1", axis=1)
df = df.rename({"level_0": "test_split_name"}, axis=1)
df = df.replace(RENAME)
df = df.pivot(index="test_split_name", columns=["aug1", "aug2"])
df.columns.set_names(["stat", "aug1", "aug2"], inplace=True)
df = df.reorder_levels(["aug1", "aug2", "stat"], axis=1)
df.columns.set_names(["", "", ""], inplace=True)
df.index.name = None

df = df[
    list(itertools.product(["Change RTT"], ["Time shift"], ["mean", "ci95"]))
    + list(
        itertools.product(["Packet loss"], ["Color jitter", "Rotate"], ["mean", "ci95"])
    )
    + list(
        itertools.product(["Change RTT"], ["Color jitter", "Rotate"], ["mean", "ci95"])
    )
    + list(itertools.product(["Color jitter"], ["Rotate"], ["mean", "ci95"]))
]
df = df.loc[["test-script", "test-human"]]

df.to_csv("table5_simclr_other_augmentation_pairs.csv")
df
```




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
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">Change RTT</th>
      <th colspan="4" halign="left">Packet loss</th>
      <th colspan="4" halign="left">Change RTT</th>
      <th colspan="2" halign="left">Color jitter</th>
    </tr>
    <tr>
      <th></th>
      <th colspan="2" halign="left">Time shift</th>
      <th colspan="2" halign="left">Color jitter</th>
      <th colspan="2" halign="left">Rotate</th>
      <th colspan="2" halign="left">Color jitter</th>
      <th colspan="2" halign="left">Rotate</th>
      <th colspan="2" halign="left">Rotate</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>ci95</th>
      <th>mean</th>
      <th>ci95</th>
      <th>mean</th>
      <th>ci95</th>
      <th>mean</th>
      <th>ci95</th>
      <th>mean</th>
      <th>ci95</th>
      <th>mean</th>
      <th>ci95</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>test-script</th>
      <td>92.18</td>
      <td>0.31</td>
      <td>90.17</td>
      <td>0.41</td>
      <td>91.94</td>
      <td>0.3</td>
      <td>91.72</td>
      <td>0.36</td>
      <td>92.38</td>
      <td>0.32</td>
      <td>91.79</td>
      <td>0.34</td>
    </tr>
    <tr>
      <th>test-human</th>
      <td>74.69</td>
      <td>1.13</td>
      <td>73.67</td>
      <td>1.24</td>
      <td>71.22</td>
      <td>1.2</td>
      <td>75.56</td>
      <td>1.23</td>
      <td>74.33</td>
      <td>1.26</td>
      <td>71.64</td>
      <td>1.23</td>
    </tr>
  </tbody>
</table>
</div>


