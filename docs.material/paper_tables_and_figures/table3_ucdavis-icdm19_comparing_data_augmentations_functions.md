# Table 3: Comparing data augmentation functions applied in supervised training.

[:simple-jupyter: :material-download:](../../paper_tables_and_figures/table3_ucdavis-icdm19_comparing_data_augmentations_functions/table3_ucdavis-icdm19_comparing_data_augmentations_functions.ipynb)


```python
import numpy as np
import pandas as pd
import statsmodels.stats.api as sms
```


```python
import itertools
import pathlib
import tempfile
```


```python
def compute_ci95(ser):
    low, high = sms.DescrStatsW(ser.values).tconfint_mean(alpha=0.05)
    mean = ser.mean()
    ci = high - mean
    return ci
```


```python
folder_campaign_summary = pathlib.Path(
    "campaigns/ucdavis-icdm19/augmentation-at-loading-with-dropout/campaign_summary/1684447037/"
)
```


```python
# load results
df = pd.concat(
    [
        pd.read_parquet(folder_campaign_summary / "runsinfo_flowpic_dim_32.parquet"),
        pd.read_parquet(folder_campaign_summary / "runsinfo_flowpic_dim_64.parquet"),
        pd.read_parquet(folder_campaign_summary / "runsinfo_flowpic_dim_1500.parquet"),
    ]
)
```


```python
df_agg_dict = dict()
for flowpic_dim in (32, 64, 1500):
    df_tmp = df[df["flowpic_dim"] == flowpic_dim]
    df_agg = df_tmp.groupby(["test_split_name", "aug_name"]).agg(
        {"acc": ["count", "mean", "std", compute_ci95]}
    )
    df_agg = df_agg.droplevel(0, axis=1).rename({"compute_ci95": "ci95"}, axis=1)
    fname = folder_campaign_summary / f"summary_flowpic_dim_{flowpic_dim}.csv"
    df_agg_dict[flowpic_dim] = df_agg
```


```python
# loading imc22-paper results
# (there are oviously copied)

IMC22_TABLE_TEST_SCRIPT = """
aug_name,32,64,1500
No augmentation,98.67,99.1,96.22
Rotate,98.6,98.87,94.89
Horizontal flip,98.93,99.27,97.33
Color jitter,96.73,96.4,94.0
Packet loss,98.73,99.6,96.22
Time shift,99.13,99.53,97.56
Change rtt,99.4,100.0,98.44
"""

IMC22_TABLE_TEST_HUMAN = """
aug_name,32,64,1500
No augmentation,92.4,85.6,73.3
Rotate,93.73,87.07,77.3
Horizontal flip,94.67,79.33,87.9
Color jitter,82.93,74.93,68.0
Packet loss,90.93,85.6,84.0
Time shift,92.8,87.33,77.3
Change rtt,96.4,88.6,90.7
"""

with tempfile.NamedTemporaryFile("w") as f_tmp:
    f_tmp.write(IMC22_TABLE_TEST_SCRIPT)
    f_tmp.seek(0)
    df_imc22_table_test_script = pd.read_csv(f_tmp.name)
    df_imc22_table_test_script = df_imc22_table_test_script.set_index("aug_name")
    df_imc22_table_test_script.columns = pd.MultiIndex.from_product(
        [["imc22-paper"], df_imc22_table_test_script.columns, ["mean"]]
    )

with tempfile.NamedTemporaryFile("w") as f_tmp:
    f_tmp.write(IMC22_TABLE_TEST_HUMAN)
    f_tmp.seek(0)
    df_imc22_table_test_human = pd.read_csv(f_tmp.name)
    df_imc22_table_test_human = df_imc22_table_test_human.set_index("aug_name")
    df_imc22_table_test_human.columns = pd.MultiIndex.from_product(
        [["imc22-paper"], df_imc22_table_test_human.columns, ["mean"]]
    )
```


```python
RENAMING = {
    "test-human": "human",
    "test-script": "script",
    "test-train-val-leftover": "leftover",
    "noaug": "No augmentation",
    "changertt": "Change rtt",
    "colorjitter": "Color jitter",
    "horizontalflip": "Horizontal flip",
    "packetloss": "Packet loss",
    "rotate": "Rotate",
    "timeshift": "Time shift",
}

AUG_NAME_ORDER = [
    "No augmentation",
    "Rotate",
    "Horizontal flip",
    "Color jitter",
    "Packet loss",
    "Time shift",
    "Change rtt",
]

partial_dfs = {
    "human": dict(),
    "script": dict(),
    "leftover": dict(),
}
for flowpic_dim in (32, 64, 1500):
    df_tmp = df_agg_dict[flowpic_dim][["mean", "ci95"]].round(2).reset_index()
    df_tmp = df_tmp.assign(
        test_split_name=df_tmp["test_split_name"].replace(RENAMING),
        aug_name=df_tmp["aug_name"].replace(RENAMING),
    )
    df_tmp = df_tmp.set_index("test_split_name", drop=True)
    for split_name in ("script", "human", "leftover"):
        df_partial = df_tmp.loc[split_name].copy()
        df_partial = df_partial.set_index("aug_name", drop=True)
        df_partial = df_partial.loc[AUG_NAME_ORDER]
        partial_dfs[split_name][flowpic_dim] = df_partial
```


```python
df_ours_script = pd.concat(partial_dfs["script"], axis=1)
df_ours_script.columns = pd.MultiIndex.from_product(
    [["ours"], *df_ours_script.columns.levels]
)

df_ours_human = pd.concat(partial_dfs["human"], axis=1)
df_ours_human.columns = pd.MultiIndex.from_product(
    [["ours"], *df_ours_human.columns.levels]
)

df_ours_leftover = pd.concat(partial_dfs["leftover"], axis=1)
df_ours_leftover.columns = pd.MultiIndex.from_product(
    [["ours"], *df_ours_leftover.columns.levels]
)
```


```python
print("=== test on script ===")
df_tmp = pd.concat((df_imc22_table_test_script, df_ours_script), axis=1)

df_tmp.loc["mean_diff", :] = np.nan
df_tmp.loc["mean_diff", ("ours", 32, "mean")] = (
    (df_tmp[("ours", 32, "mean")] - df_tmp[("imc22-paper", "32", "mean")])
    .mean()
    .round(2)
)
df_tmp.loc["mean_diff", ("ours", 64, "mean")] = (
    (df_tmp[("ours", 64, "mean")] - df_tmp[("imc22-paper", "64", "mean")])
    .mean()
    .round(2)
)
df_tmp.loc["mean_diff", ("ours", 1500, "mean")] = (
    (df_tmp[("ours", 1500, "mean")] - df_tmp[("imc22-paper", "1500", "mean")])
    .mean()
    .round(2)
)
df_tmp.fillna("")
```

    === test on script ===





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

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">imc22-paper</th>
      <th colspan="6" halign="left">ours</th>
    </tr>
    <tr>
      <th></th>
      <th>32</th>
      <th>64</th>
      <th>1500</th>
      <th colspan="2" halign="left">32</th>
      <th colspan="2" halign="left">64</th>
      <th colspan="2" halign="left">1500</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>mean</th>
      <th>mean</th>
      <th>mean</th>
      <th>ci95</th>
      <th>mean</th>
      <th>ci95</th>
      <th>mean</th>
      <th>ci95</th>
    </tr>
    <tr>
      <th>aug_name</th>
      <th></th>
      <th></th>
      <th></th>
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
      <th>No augmentation</th>
      <td>98.67</td>
      <td>99.1</td>
      <td>96.22</td>
      <td>95.73</td>
      <td>0.27</td>
      <td>95.96</td>
      <td>0.29</td>
      <td>94.44</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th>Rotate</th>
      <td>98.6</td>
      <td>98.87</td>
      <td>94.89</td>
      <td>96.36</td>
      <td>0.39</td>
      <td>96.89</td>
      <td>0.39</td>
      <td>95.47</td>
      <td>0.47</td>
    </tr>
    <tr>
      <th>Horizontal flip</th>
      <td>98.93</td>
      <td>99.27</td>
      <td>97.33</td>
      <td>95.11</td>
      <td>0.41</td>
      <td>95.96</td>
      <td>0.49</td>
      <td>95.11</td>
      <td>0.68</td>
    </tr>
    <tr>
      <th>Color jitter</th>
      <td>96.73</td>
      <td>96.4</td>
      <td>94.0</td>
      <td>97.87</td>
      <td>0.45</td>
      <td>97.42</td>
      <td>0.67</td>
      <td>94.89</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>Packet loss</th>
      <td>98.73</td>
      <td>99.6</td>
      <td>96.22</td>
      <td>96.98</td>
      <td>0.48</td>
      <td>96.89</td>
      <td>0.53</td>
      <td>95.96</td>
      <td>0.7</td>
    </tr>
    <tr>
      <th>Time shift</th>
      <td>99.13</td>
      <td>99.53</td>
      <td>97.56</td>
      <td>96.71</td>
      <td>0.51</td>
      <td>97.11</td>
      <td>0.36</td>
      <td>96.80</td>
      <td>0.32</td>
    </tr>
    <tr>
      <th>Change rtt</th>
      <td>99.4</td>
      <td>100.0</td>
      <td>98.44</td>
      <td>97.33</td>
      <td>0.39</td>
      <td>97.29</td>
      <td>0.35</td>
      <td>96.80</td>
      <td>0.35</td>
    </tr>
    <tr>
      <th>mean_diff</th>
      <td></td>
      <td></td>
      <td></td>
      <td>-2.01</td>
      <td></td>
      <td>-2.18</td>
      <td></td>
      <td>-0.74</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>




```python
print("=== test on human ===")
df_tmp = pd.concat((df_imc22_table_test_human, df_ours_human), axis=1)

df_tmp.loc["mean_diff", :] = np.nan
df_tmp.loc["mean_diff", ("ours", 32, "mean")] = (
    (df_tmp[("ours", 32, "mean")] - df_tmp[("imc22-paper", "32", "mean")])
    .mean()
    .round(2)
)
df_tmp.loc["mean_diff", ("ours", 64, "mean")] = (
    (df_tmp[("ours", 64, "mean")] - df_tmp[("imc22-paper", "64", "mean")])
    .mean()
    .round(2)
)
df_tmp.loc["mean_diff", ("ours", 1500, "mean")] = (
    (df_tmp[("ours", 1500, "mean")] - df_tmp[("imc22-paper", "1500", "mean")])
    .mean()
    .round(2)
)
df_tmp.fillna("")
```

    === test on human ===





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

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">imc22-paper</th>
      <th colspan="6" halign="left">ours</th>
    </tr>
    <tr>
      <th></th>
      <th>32</th>
      <th>64</th>
      <th>1500</th>
      <th colspan="2" halign="left">32</th>
      <th colspan="2" halign="left">64</th>
      <th colspan="2" halign="left">1500</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>mean</th>
      <th>mean</th>
      <th>mean</th>
      <th>ci95</th>
      <th>mean</th>
      <th>ci95</th>
      <th>mean</th>
      <th>ci95</th>
    </tr>
    <tr>
      <th>aug_name</th>
      <th></th>
      <th></th>
      <th></th>
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
      <th>No augmentation</th>
      <td>92.4</td>
      <td>85.6</td>
      <td>73.3</td>
      <td>69.48</td>
      <td>1.17</td>
      <td>69.88</td>
      <td>1.26</td>
      <td>68.67</td>
      <td>1.07</td>
    </tr>
    <tr>
      <th>Rotate</th>
      <td>93.73</td>
      <td>87.07</td>
      <td>77.3</td>
      <td>71.57</td>
      <td>1.95</td>
      <td>71.00</td>
      <td>1.35</td>
      <td>67.87</td>
      <td>0.86</td>
    </tr>
    <tr>
      <th>Horizontal flip</th>
      <td>94.67</td>
      <td>79.33</td>
      <td>87.9</td>
      <td>69.80</td>
      <td>1.39</td>
      <td>70.92</td>
      <td>1.83</td>
      <td>73.82</td>
      <td>0.82</td>
    </tr>
    <tr>
      <th>Color jitter</th>
      <td>82.93</td>
      <td>74.93</td>
      <td>68.0</td>
      <td>68.84</td>
      <td>2.59</td>
      <td>71.33</td>
      <td>1.86</td>
      <td>68.59</td>
      <td>1.76</td>
    </tr>
    <tr>
      <th>Packet loss</th>
      <td>90.93</td>
      <td>85.6</td>
      <td>84.0</td>
      <td>71.00</td>
      <td>1.02</td>
      <td>73.17</td>
      <td>0.89</td>
      <td>72.13</td>
      <td>1.04</td>
    </tr>
    <tr>
      <th>Time shift</th>
      <td>92.8</td>
      <td>87.33</td>
      <td>77.3</td>
      <td>70.36</td>
      <td>1.65</td>
      <td>72.53</td>
      <td>1.02</td>
      <td>70.84</td>
      <td>1.34</td>
    </tr>
    <tr>
      <th>Change rtt</th>
      <td>96.4</td>
      <td>88.6</td>
      <td>90.7</td>
      <td>70.04</td>
      <td>2.44</td>
      <td>72.05</td>
      <td>1.16</td>
      <td>72.69</td>
      <td>1.48</td>
    </tr>
    <tr>
      <th>mean_diff</th>
      <td></td>
      <td></td>
      <td></td>
      <td>-21.82</td>
      <td></td>
      <td>-12.51</td>
      <td></td>
      <td>-9.13</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>




```python
print("=== test on leftover ===")
df_ours_leftover
```

    === test on leftover ===





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

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="6" halign="left">ours</th>
    </tr>
    <tr>
      <th></th>
      <th colspan="2" halign="left">32</th>
      <th colspan="2" halign="left">64</th>
      <th colspan="2" halign="left">1500</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>ci95</th>
      <th>mean</th>
      <th>ci95</th>
      <th>mean</th>
      <th>ci95</th>
    </tr>
    <tr>
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
      <th>No augmentation</th>
      <td>96.05</td>
      <td>0.19</td>
      <td>96.22</td>
      <td>0.31</td>
      <td>95.62</td>
      <td>0.51</td>
    </tr>
    <tr>
      <th>Rotate</th>
      <td>97.01</td>
      <td>0.24</td>
      <td>97.28</td>
      <td>0.34</td>
      <td>95.93</td>
      <td>0.41</td>
    </tr>
    <tr>
      <th>Horizontal flip</th>
      <td>95.88</td>
      <td>0.25</td>
      <td>96.38</td>
      <td>0.50</td>
      <td>96.47</td>
      <td>0.57</td>
    </tr>
    <tr>
      <th>Color jitter</th>
      <td>97.46</td>
      <td>0.33</td>
      <td>96.82</td>
      <td>0.41</td>
      <td>95.79</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>Packet loss</th>
      <td>97.47</td>
      <td>0.35</td>
      <td>97.48</td>
      <td>0.28</td>
      <td>97.29</td>
      <td>0.27</td>
    </tr>
    <tr>
      <th>Time shift</th>
      <td>97.44</td>
      <td>0.42</td>
      <td>97.78</td>
      <td>0.38</td>
      <td>97.94</td>
      <td>0.19</td>
    </tr>
    <tr>
      <th>Change rtt</th>
      <td>98.24</td>
      <td>0.31</td>
      <td>98.29</td>
      <td>0.39</td>
      <td>98.43</td>
      <td>0.12</td>
    </tr>
  </tbody>
</table>
</div>


