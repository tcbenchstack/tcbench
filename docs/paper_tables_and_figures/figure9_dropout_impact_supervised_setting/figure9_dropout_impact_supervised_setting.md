# Figure 9: Accuracy difference w/ and w/o Dropout in supervised learning.

[:simple-jupyter: :material-download:](../../paper_tables_and_figures/figure9_dropout_impact_supervised_setting/figure9_dropout_impact_supervised_setting.ipynb)


```python
import itertools

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
df_with_dropout = pd.concat(
    [
        pd.read_parquet(
            "campaigns/ucdavis-icdm19/augmentation-at-loading-with-dropout/campaign_summary/1684447037/runsinfo_flowpic_dim_1500.parquet"
        ),
        pd.read_parquet(
            "campaigns/ucdavis-icdm19/augmentation-at-loading-with-dropout/campaign_summary/1684447037/runsinfo_flowpic_dim_32.parquet"
        ),
    ]
)
```


```python
df_no_dropout = pd.concat(
    [
        pd.read_parquet(
            "campaigns/ucdavis-icdm19/augmentation-at-loading-suppress-dropout/campaign_summary/1684566558/runsinfo_flowpic_dim_1500.parquet"
        ),
        pd.read_parquet(
            "campaigns/ucdavis-icdm19/augmentation-at-loading-suppress-dropout/campaign_summary/1684566558/runsinfo_flowpic_dim_32.parquet"
        ),
    ]
)
```


```python
df_tmp1 = df_with_dropout[
    [
        "flowpic_dim",
        "test_split_name",
        "aug_name",
        "seed",
        "split_index",
        "acc",
    ]
].rename(columns={"acc": "withdropout_acc"})
df_tmp2 = df_no_dropout[
    [
        "flowpic_dim",
        "test_split_name",
        "aug_name",
        "seed",
        "split_index",
        "acc",
    ]
].rename(columns={"acc": "nodropout_acc"})
df = pd.merge(
    df_tmp1,
    df_tmp2,
    on=[
        "flowpic_dim",
        "test_split_name",
        "aug_name",
        "seed",
        "split_index",
    ],
    suffixes=["withdropout_", "nodropout_"],
)
```


```python
df = df.iloc[df["nodropout_acc"].dropna().index]
```


```python
df["acc_diff"] = df["withdropout_acc"] - df["nodropout_acc"]
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
df_merged = df.groupby(["flowpic_dim", "test_split_name", "aug_name"]).agg(
    {"acc_diff": ["mean", "std", "count", "min", "max", compute_confidence_intervals]}
)
df_merged = df_merged.rename(
    columns={"compute_confidence_intervals": "confidence_interval"}
)
df_merged = df_merged.droplevel(0, axis=1)
```


```python
df_merged
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
      <th></th>
      <th>mean</th>
      <th>std</th>
      <th>count</th>
      <th>min</th>
      <th>max</th>
      <th>confidence_interval</th>
    </tr>
    <tr>
      <th>flowpic_dim</th>
      <th>test_split_name</th>
      <th>aug_name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="14" valign="top">32</th>
      <th rowspan="7" valign="top">test-human</th>
      <th>changertt</th>
      <td>8.835341e-01</td>
      <td>3.421913</td>
      <td>15</td>
      <td>-6.024096</td>
      <td>6.024096</td>
      <td>1.894992</td>
    </tr>
    <tr>
      <th>colorjitter</th>
      <td>1.285141e+00</td>
      <td>5.750530</td>
      <td>15</td>
      <td>-8.433735</td>
      <td>10.843373</td>
      <td>3.184537</td>
    </tr>
    <tr>
      <th>horizontalflip</th>
      <td>-9.473903e-16</td>
      <td>2.694058</td>
      <td>15</td>
      <td>-4.819277</td>
      <td>6.024096</td>
      <td>1.491919</td>
    </tr>
    <tr>
      <th>noaug</th>
      <td>-1.204819e+00</td>
      <td>3.020642</td>
      <td>15</td>
      <td>-4.819277</td>
      <td>3.614458</td>
      <td>1.672776</td>
    </tr>
    <tr>
      <th>packetloss</th>
      <td>-8.032129e-01</td>
      <td>2.215345</td>
      <td>15</td>
      <td>-3.614458</td>
      <td>3.614458</td>
      <td>1.226817</td>
    </tr>
    <tr>
      <th>rotate</th>
      <td>-1.285141e+00</td>
      <td>4.998116</td>
      <td>15</td>
      <td>-12.048193</td>
      <td>6.024096</td>
      <td>2.767864</td>
    </tr>
    <tr>
      <th>timeshift</th>
      <td>-1.606426e-01</td>
      <td>3.806343</td>
      <td>15</td>
      <td>-4.819277</td>
      <td>6.024096</td>
      <td>2.107882</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">test-script</th>
      <th>changertt</th>
      <td>3.555556e-01</td>
      <td>0.791489</td>
      <td>15</td>
      <td>-0.666667</td>
      <td>2.000000</td>
      <td>0.438312</td>
    </tr>
    <tr>
      <th>colorjitter</th>
      <td>5.777778e-01</td>
      <td>1.256517</td>
      <td>15</td>
      <td>-2.000000</td>
      <td>3.333333</td>
      <td>0.695836</td>
    </tr>
    <tr>
      <th>horizontalflip</th>
      <td>-1.333333e-01</td>
      <td>1.104105</td>
      <td>15</td>
      <td>-2.666667</td>
      <td>1.333333</td>
      <td>0.611433</td>
    </tr>
    <tr>
      <th>noaug</th>
      <td>-6.222222e-01</td>
      <td>0.990964</td>
      <td>15</td>
      <td>-2.000000</td>
      <td>0.666667</td>
      <td>0.548778</td>
    </tr>
    <tr>
      <th>packetloss</th>
      <td>2.666667e-01</td>
      <td>1.176489</td>
      <td>15</td>
      <td>-2.000000</td>
      <td>2.666667</td>
      <td>0.651518</td>
    </tr>
    <tr>
      <th>rotate</th>
      <td>-1.777778e-01</td>
      <td>0.924676</td>
      <td>15</td>
      <td>-1.333333</td>
      <td>1.333333</td>
      <td>0.512069</td>
    </tr>
    <tr>
      <th>timeshift</th>
      <td>4.444444e-02</td>
      <td>1.053088</td>
      <td>15</td>
      <td>-2.000000</td>
      <td>2.000000</td>
      <td>0.583181</td>
    </tr>
    <tr>
      <th rowspan="14" valign="top">1500</th>
      <th rowspan="7" valign="top">test-human</th>
      <th>changertt</th>
      <td>7.228916e-01</td>
      <td>2.762465</td>
      <td>15</td>
      <td>-4.819277</td>
      <td>4.819277</td>
      <td>1.529802</td>
    </tr>
    <tr>
      <th>colorjitter</th>
      <td>1.847390e+00</td>
      <td>2.446646</td>
      <td>15</td>
      <td>-1.204819</td>
      <td>6.024096</td>
      <td>1.354908</td>
    </tr>
    <tr>
      <th>horizontalflip</th>
      <td>5.622490e-01</td>
      <td>1.633450</td>
      <td>15</td>
      <td>-2.409639</td>
      <td>3.614458</td>
      <td>0.904575</td>
    </tr>
    <tr>
      <th>noaug</th>
      <td>-1.204819e+00</td>
      <td>3.727437</td>
      <td>15</td>
      <td>-6.024096</td>
      <td>7.228916</td>
      <td>2.064186</td>
    </tr>
    <tr>
      <th>packetloss</th>
      <td>1.847390e+00</td>
      <td>3.779005</td>
      <td>15</td>
      <td>-3.614458</td>
      <td>9.638554</td>
      <td>2.092743</td>
    </tr>
    <tr>
      <th>rotate</th>
      <td>-5.622490e-01</td>
      <td>2.488664</td>
      <td>15</td>
      <td>-3.614458</td>
      <td>4.819277</td>
      <td>1.378176</td>
    </tr>
    <tr>
      <th>timeshift</th>
      <td>-1.124498e+00</td>
      <td>1.903171</td>
      <td>15</td>
      <td>-3.614458</td>
      <td>2.409639</td>
      <td>1.053941</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">test-script</th>
      <th>changertt</th>
      <td>4.444444e-02</td>
      <td>1.167460</td>
      <td>15</td>
      <td>-1.333333</td>
      <td>2.666667</td>
      <td>0.646518</td>
    </tr>
    <tr>
      <th>colorjitter</th>
      <td>8.000000e-01</td>
      <td>2.645151</td>
      <td>15</td>
      <td>-3.333333</td>
      <td>6.666667</td>
      <td>1.464836</td>
    </tr>
    <tr>
      <th>horizontalflip</th>
      <td>7.555556e-01</td>
      <td>1.668887</td>
      <td>15</td>
      <td>-1.333333</td>
      <td>4.000000</td>
      <td>0.924199</td>
    </tr>
    <tr>
      <th>noaug</th>
      <td>-1.777778e-01</td>
      <td>1.521625</td>
      <td>15</td>
      <td>-2.000000</td>
      <td>3.333333</td>
      <td>0.842648</td>
    </tr>
    <tr>
      <th>packetloss</th>
      <td>7.555556e-01</td>
      <td>2.150920</td>
      <td>15</td>
      <td>-2.000000</td>
      <td>4.000000</td>
      <td>1.191140</td>
    </tr>
    <tr>
      <th>rotate</th>
      <td>-8.888889e-02</td>
      <td>0.903842</td>
      <td>15</td>
      <td>-2.000000</td>
      <td>1.333333</td>
      <td>0.500531</td>
    </tr>
    <tr>
      <th>timeshift</th>
      <td>0.000000e+00</td>
      <td>0.666667</td>
      <td>15</td>
      <td>-0.666667</td>
      <td>1.333333</td>
      <td>0.369188</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 4))

for ax, (flowpic_dim, test_split_name) in zip(
    axes.flatten(), itertools.product((32, 1500), ("test-human", "test-script"))
):
    # df_merged.loc[(flowpic_dim, test_split_name)]['mean'].plot(kind='bar', ax=ax)

    ax.bar(
        list(df_merged.loc[(flowpic_dim, test_split_name)].index),
        df_merged.loc[(flowpic_dim, test_split_name)]["mean"],
        yerr=df_merged.loc[(flowpic_dim, test_split_name)]["confidence_interval"],
        align="center",
        alpha=0.5,
        ecolor="black",
        capsize=10,
    )

    ax.set_title(f"{test_split_name} @ {flowpic_dim}x{flowpic_dim}")

    ax.set_xticklabels(
        list(df_merged.loc[(flowpic_dim, test_split_name)].index), rotation=90
    )


plt.tight_layout()
```

    /tmp/ipykernel_62018/2491681095.py:13: UserWarning: FixedFormatter should only be used together with FixedLocator
      ax.set_xticklabels(list(df_merged.loc[(flowpic_dim, test_split_name)].index), rotation=90)



    
![png](figure9_dropout_impact_supervised_setting_files/figure9_dropout_impact_supervised_setting_11_1.png)
    

