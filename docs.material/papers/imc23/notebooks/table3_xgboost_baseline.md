
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
# Table 3: (G0) Baseline ML performance without augmentation in a supervised setting.

[:simple-jupyter: :material-download:](/tcbench/papers/imc23/notebooks/table3_xgboost_baseline.ipynb)


```python
import pandas as pd
```

```python
df = pd.read_csv(
    "./campaigns/ucdavis-icdm19/xgboost/noaugmentation-flowpic/campaign_summary/noaugmentation-flowpic/summary_flowpic_dim_32.csv",
    header=[0, 1],
    index_col=[0, 1],
)
```

```python
# reformatting
df_tmp = df["acc"][["mean", "ci95"]].round(2)
df_tmp.loc[["test-script", "test-human"]].droplevel(1, axis=0).astype(float).round(2)
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
<td>96.80</td>
<td>0.37</td>
</tr>
<tr>
<th>test-human</th>
<td>73.65</td>
<td>2.14</td>
</tr>
</tbody>
</table>
</div>
</div>



```python
df = pd.read_csv(
    "./campaigns/ucdavis-icdm19/xgboost/noaugmentation-timeseries/campaign_summary/noaugmentation-timeseries/summary_max_n_pkts_10.csv",
    header=[0, 1],
    index_col=[0, 1],
)
```

```python
# reformatting
df_tmp = df["acc"][["mean", "ci95"]].round(2)
df_tmp.loc[["test-script", "test-human"]].droplevel(1, axis=0).astype(float).round(2)
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
<td>94.53</td>
<td>0.56</td>
</tr>
<tr>
<th>test-human</th>
<td>66.91</td>
<td>1.40</td>
</tr>
</tbody>
</table>
</div>
</div>

