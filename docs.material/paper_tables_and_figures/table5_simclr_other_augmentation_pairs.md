# Table 5: Comparing the fine-tuning performance when using different pairs of augmentation for pretraining.

[:simple-jupyter: :material-download:](../../paper_tables_and_figures/table5_simclr_other_augmentation_pairs/table5_simclr_other_augmentation_pairs.ipynb)


```python
import pandas as pd
```


```python
def load_summary_csv(fname, level1, level2):
    df = (
        pd.read_csv(fname)
        .set_index("test_split_name", drop=True)
        .drop(["with_dropout", "projection_layer_dim", "finetune_augmentation"], axis=1)
    )
    df.columns = df.iloc[0].values
    df = df.iloc[1:]
    df = df[["mean", "ci95"]].astype(float).round(2)
    df.columns = pd.MultiIndex.from_arrays(
        [[level1, level1], [level2, level2], df.columns.tolist()]
    )
    df = df.loc[["test-script", "test-human"]]
    df.index.name = None
    return df
```


```python
pd.concat(
    (
        load_summary_csv(
            "campaigns/ucdavis-icdm19/simclr-other-augmentation-pairs/colorjitter-packetloss/campaign_summary/1684886215/summary_flowpic_dim_32.csv",
            level1="Packet loss",
            level2="Color jitter",
        ),
        load_summary_csv(
            "campaigns/ucdavis-icdm19/simclr-other-augmentation-pairs/rotate-packetloss/campaign_summary/1684886215/summary_flowpic_dim_32.csv",
            level1="Packet loss",
            level2="Rotate",
        ),
        load_summary_csv(
            "campaigns/ucdavis-icdm19/simclr-other-augmentation-pairs/colorjitter-changertt/campaign_summary/1684886215/summary_flowpic_dim_32.csv",
            level1="Change rtt",
            level2="Color jitter",
        ),
        load_summary_csv(
            "campaigns/ucdavis-icdm19/simclr-other-augmentation-pairs/rotate-changertt/campaign_summary/1684886215/summary_flowpic_dim_32.csv",
            level1="Change rtt",
            level2="Rotate",
        ),
        load_summary_csv(
            "campaigns/ucdavis-icdm19/simclr-other-augmentation-pairs/colorjitter-rotate/campaign_summary/1684886215/summary_flowpic_dim_32.csv",
            level1="Color jitter",
            level2="Rotate",
        ),
    ),
    axis=1,
)
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
      <th colspan="4" halign="left">Packet loss</th>
      <th colspan="4" halign="left">Change rtt</th>
      <th colspan="2" halign="left">Color jitter</th>
    </tr>
    <tr>
      <th></th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>test-script</th>
      <td>89.87</td>
      <td>0.37</td>
      <td>91.79</td>
      <td>0.28</td>
      <td>91.42</td>
      <td>0.36</td>
      <td>92.56</td>
      <td>0.32</td>
      <td>91.79</td>
      <td>0.33</td>
    </tr>
    <tr>
      <th>test-human</th>
      <td>73.84</td>
      <td>1.25</td>
      <td>71.56</td>
      <td>1.21</td>
      <td>74.59</td>
      <td>1.10</td>
      <td>73.43</td>
      <td>1.35</td>
      <td>70.93</td>
      <td>1.18</td>
    </tr>
  </tbody>
</table>
</div>



The pair ("Time shift", "Change rtt") missing from the table is coming from Table4 results (see [:simple-jupyter: `table4_simclr_dropout_and_projectionlayer.ipynb`](../../paper_tables_and_figures/table4_simclr_dropout_and_projectionlayer/))
