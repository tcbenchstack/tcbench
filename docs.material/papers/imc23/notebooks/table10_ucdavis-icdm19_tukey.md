
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
# Table 10: Performance comparison across augmentations for different flowpic sizes.

[:simple-jupyter: :material-download:](/tcbench/papers/imc23/notebooks/table10_ucdavis-icdm19_tukey.ipynb)


```python
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import tukey_hsd
```

```python
folder = pathlib.Path(
    "campaigns/ucdavis-icdm19/augmentation-at-loading-with-dropout/campaign_summary/augment-at-loading-with-dropout"
)
df = pd.concat(
    (
        pd.read_parquet(folder / "runsinfo_flowpic_dim_1500.parquet"),
        pd.read_parquet(folder / "runsinfo_flowpic_dim_64.parquet"),
        pd.read_parquet(folder / "runsinfo_flowpic_dim_32.parquet"),
    )
)
```

```python
# df = pd.read_parquet('campaigns/ucdavis-icdm19/augmentation-at-loading-with-dropout/campaign_summary/1684447037/merged_runsinfo.parquet')
```

```python
df_script = df[df["test_split_name"] == "test-script"]

acc_32 = df_script[df_script["flowpic_dim"] == 32]["acc"].values.tolist()
acc_64 = df_script[df_script["flowpic_dim"] == 64]["acc"].values.tolist()
acc_1500 = df_script[df_script["flowpic_dim"] == 1500]["acc"].values.tolist()
```

```python
res = tukey_hsd(acc_32, acc_64, acc_1500)
```

```python
df = pd.DataFrame(
    np.array([res.pvalue[0, 1], res.pvalue[0, 2], res.pvalue[1, 2]]).reshape(-1, 1),
    columns=["pvalue"],
    index=pd.MultiIndex.from_arrays(
        [("32x32", "32x32", "64x64"), ("64x64", "1500x1500", "1500x1500")]
    ),
)
df = df.assign(is_different=df["pvalue"] < 0.05)
```

```python
df
```



<div class="md-typeset__scrollwrap">
<div class="md-typeset__table">
<table>
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
<td>5.772842e-01</td>
<td>False</td>
</tr>
<tr>
<th>1500x1500</th>
<td>1.936038e-06</td>
<td>True</td>
</tr>
<tr>
<th>64x64</th>
<th>1500x1500</th>
<td>1.044272e-08</td>
<td>True</td>
</tr>
</tbody>
</table>
</div>
</div>

