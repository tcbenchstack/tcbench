# Table 2: (G0) Baseline ML performance without augmentation in a supervised setting.

[:simple-jupyter: :material-download:](../../paper_tables_and_figures/table2_xgboost_baseline/table2_xgboost_baseline.ipynb)


```python
import pandas as pd
```


```python
df = pd.read_csv(
    "./campaigns/ucdavis-icdm19/xgboost/noaugmentation-flowpic/campaign_summary/1684951896/summary_flowpic_dim_32.csv"
)
# df
```


```python
# this is just reformatting to extracting the right values
df.columns = [col.split(".")[0] for col in df.columns]
df = df.set_index(["test_split_name", "aug_name"], drop=True)
df.columns = pd.MultiIndex.from_arrays([df.columns, df.iloc[0].values])
df = df.loc[["test-script", "test-human"]].droplevel(1, axis=0).astype(float).round(2)

df["acc"].T.loc[["mean", "ci95"]]
```




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
      <th>test_split_name</th>
      <th>test-script</th>
      <th>test-human</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mean</th>
      <td>96.53</td>
      <td>71.81</td>
    </tr>
    <tr>
      <th>ci95</th>
      <td>0.15</td>
      <td>2.85</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = pd.read_csv(
    "./campaigns/ucdavis-icdm19/xgboost/noaugmentation-timeseries/campaign_summary/1685008005/summary_max_n_pkts_10.csv"
)
# df
```


```python
# this is just reformatting to extracting the right values
df.columns = [col.split(".")[0] for col in df.columns]
df = df.set_index(["test_split_name", "aug_name"], drop=True)
df.columns = pd.MultiIndex.from_arrays([df.columns, df.iloc[0].values])
df = df.loc[["test-script", "test-human"]].droplevel(1, axis=0).astype(float).round(2)

df["acc"].T.loc[["mean", "ci95"]]
```




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
      <th>test_split_name</th>
      <th>test-script</th>
      <th>test-human</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mean</th>
      <td>94.53</td>
      <td>67.47</td>
    </tr>
    <tr>
      <th>ci95</th>
      <td>0.45</td>
      <td>1.51</td>
    </tr>
  </tbody>
</table>
</div>


