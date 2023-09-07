# Table 9: Macro-average Accuracy with different retraining dataset and different sampling methods

[:simple-jupyter: :material-download:](../../paper_tables_and_figures/table9_icdm_finetuning_per_class_metrics_on_human/table9_icdm_finetuning_per_class_metrics_on_human.ipynb)


```python
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.stats.api as sms

%matplotlib inline
%config InlineBackend.figure_format='retina'
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
path = pathlib.Path("./campaigns/ucdavis-icdm19-git-repo-forked/artifacts/")

class_repss = list(path.glob("*10/"))
```


```python
data = dict()

for path in class_repss:
    if "script" in str(path):
        class_reps = list(path.glob("*class_rep.csv"))
        accs = [pd.read_csv(file).iloc[6].values[2] for file in class_reps]

        augmentation_name = path.name.split("_")[0].replace("Sampling", "")
        data[augmentation_name] = (
            np.mean(accs) * 100,
            compute_confidence_intervals(accs),
        )

df_script = pd.DataFrame(data, index=["mean", "ci95"]).T.round(2)
df_script.columns = pd.MultiIndex.from_arrays([["script", "script"], df_script.columns])
# df_script
```


```python
data = dict()
for path in class_repss:
    if "human" in str(path):
        class_reps = list(path.glob("*class_rep.csv"))
        accs = [pd.read_csv(file).iloc[6].values[2] for file in class_reps]

        augmentation_name = path.name.split("_")[0].replace("Sampling", "")
        data[augmentation_name] = (
            np.mean(accs) * 100,
            compute_confidence_intervals(accs),
        )

df_human = pd.DataFrame(data, index=["mean", "ci95"]).T.round(2)
df_human.columns = pd.MultiIndex.from_arrays([["human", "human"], df_human.columns])
```


```python
df_tmp = pd.concat((df_script, df_human), axis=1).T
display(df_tmp)
df_tmp.to_csv("icdm_finetuning_per_class_metrics_on_human.csv")
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
      <th>FixedStep</th>
      <th>Random</th>
      <th>Incremental</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">script</th>
      <th>mean</th>
      <td>87.11</td>
      <td>94.63</td>
      <td>96.22</td>
    </tr>
    <tr>
      <th>ci95</th>
      <td>0.09</td>
      <td>0.02</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">human</th>
      <th>mean</th>
      <td>82.60</td>
      <td>87.29</td>
      <td>92.56</td>
    </tr>
    <tr>
      <th>ci95</th>
      <td>0.03</td>
      <td>0.04</td>
      <td>0.03</td>
    </tr>
  </tbody>
</table>
</div>

