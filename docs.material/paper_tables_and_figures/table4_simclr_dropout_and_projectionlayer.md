# Table 4: Impact of dropout and SimCLR projection layer dimension on fine-tuning.

[:simple-jupyter: :material-download:](../../paper_tables_and_figures/table4_simclr_dropout_and_projectionlayer/table4_simclr_dropout_and_projectionlayer.ipynb)


```python
import pandas as pd
```


```python
import itertools
```


```python
df = pd.read_csv(
    "campaigns/ucdavis-icdm19/simclr-dropout-and-projection/campaign_summary/1684821411/summary_flowpic_dim_32.csv"
)

# reformat the raw report
RENAME = {
    "acc": "count",
    "acc.1": "mean",
    "acc.2": "std",
    "acc.3": "ci95"
}
df = df.set_index(['test_split_name', 'with_dropout', 'projection_layer_dim'], drop=True)
df = df.rename(RENAME, axis=1)
df = df.drop('finetune_augmentation', axis=1)
df = df.iloc[1:].astype(float).round(2)
df = df[['mean', 'ci95']].T
df = df[list(itertools.product(['test-script', 'test-human'], [True, False], [30.0, 84.0]))]
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
      <th>test_split_name</th>
      <th colspan="4" halign="left">test-script</th>
      <th colspan="4" halign="left">test-human</th>
    </tr>
    <tr>
      <th>with_dropout</th>
      <th colspan="2" halign="left">True</th>
      <th colspan="2" halign="left">False</th>
      <th colspan="2" halign="left">True</th>
      <th colspan="2" halign="left">False</th>
    </tr>
    <tr>
      <th>projection_layer_dim</th>
      <th>30.0</th>
      <th>84.0</th>
      <th>30.0</th>
      <th>84.0</th>
      <th>30.0</th>
      <th>84.0</th>
      <th>30.0</th>
      <th>84.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mean</th>
      <td>91.80</td>
      <td>92.19</td>
      <td>92.26</td>
      <td>92.49</td>
      <td>71.88</td>
      <td>73.55</td>
      <td>75.22</td>
      <td>74.04</td>
    </tr>
    <tr>
      <th>ci95</th>
      <td>0.38</td>
      <td>0.37</td>
      <td>0.32</td>
      <td>0.31</td>
      <td>1.33</td>
      <td>1.10</td>
      <td>1.22</td>
      <td>1.38</td>
    </tr>
  </tbody>
</table>
</div>


