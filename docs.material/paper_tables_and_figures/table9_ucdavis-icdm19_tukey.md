# Table 9: Performance comparison across augmentations for different flowpic sizes.

[:simple-jupyter: :material-download:](../../paper_tables_and_figures/table9_ucdavis-icdm19_tukey/table9_ucdavis-icdm19_tukey.ipynb)


```python
import pandas as pd
import numpy as np
```


```python
from scipy.stats import tukey_hsd
```


```python
df = pd.read_parquet('campaigns/ucdavis-icdm19/augmentation-at-loading-with-dropout/campaign_summary/1684447037/merged_runsinfo.parquet')
```


```python
df_script = df[df['test_split_name'] == 'test-script']

acc_32 = df_script[df_script['flowpic_dim'] == 32]['acc'].values.tolist()
acc_64 = df_script[df_script['flowpic_dim'] == 64]['acc'].values.tolist()
acc_1500 = df_script[df_script['flowpic_dim'] == 1500]['acc'].values.tolist()
```


```python
res = tukey_hsd(acc_32, acc_64, acc_1500)
```


```python
df = pd.DataFrame(np.array([res.pvalue[0, 1], res.pvalue[0, 2], res.pvalue[1,2]]).reshape(-1, 1), columns=['pvalue'],
            index=pd.MultiIndex.from_arrays([('32x32', '32x32', '64x64'), ('64x64', '1500x1500', '1500x1500')]))
df = df.assign(is_different=df['pvalue']<0.05)
```


```python
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

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>pvalue</th>
      <th>is_different</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">32x32</th>
      <th>64x64</th>
      <td>9.380580e-01</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1500x1500</th>
      <td>3.718318e-07</td>
      <td>True</td>
    </tr>
    <tr>
      <th>64x64</th>
      <th>1500x1500</th>
      <td>6.270783e-08</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>


