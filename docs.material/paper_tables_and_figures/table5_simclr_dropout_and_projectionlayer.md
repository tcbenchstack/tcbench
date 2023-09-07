# Table 5: Impact of dropout and SimCLR projection layer dimension on fine-tuning.

[:simple-jupyter: :material-download:](../../paper_tables_and_figures/table5_simclr_dropout_and_projectionlayer/table5_simclr_dropout_and_projectionlayer.ipynb)


```python
import itertools

import pandas as pd
```


```python
df = pd.read_csv(
    "campaigns/ucdavis-icdm19/simclr-dropout-and-projection/campaign_summary/simclr-dropout-and-projection/summary_flowpic_dim_32.csv",
    header=[0, 1],
    index_col=[0, 1, 2],
)

df = df["acc"][["mean", "ci95"]]
df = df.T
df.columns.set_names("test_split_name", level=0, inplace=True)
df.columns.set_names("projection_layer_dim", level=1, inplace=True)
df.columns.set_names("with_dropout", level=2, inplace=True)
df = df.reorder_levels(
    ["test_split_name", "with_dropout", "projection_layer_dim"], axis=1
)

df = df[list(itertools.product(["test-script", "test-human"], [True, False], [30, 84]))]
df = df.round(2)

df.to_csv("table5_simclr_dropout_and_projectionlayer.csv")
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
      <th>30</th>
      <th>84</th>
      <th>30</th>
      <th>84</th>
      <th>30</th>
      <th>84</th>
      <th>30</th>
      <th>84</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mean</th>
      <td>91.81</td>
      <td>92.02</td>
      <td>92.18</td>
      <td>92.54</td>
      <td>72.12</td>
      <td>73.31</td>
      <td>74.69</td>
      <td>74.35</td>
    </tr>
    <tr>
      <th>ci95</th>
      <td>0.38</td>
      <td>0.36</td>
      <td>0.31</td>
      <td>0.33</td>
      <td>1.37</td>
      <td>1.04</td>
      <td>1.13</td>
      <td>1.38</td>
    </tr>
  </tbody>
</table>
</div>


