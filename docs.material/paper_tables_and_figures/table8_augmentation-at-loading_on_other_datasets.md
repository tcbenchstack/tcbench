# Table 8 (G3) Data augmentation in supervised setting on other datasets.

[:simple-jupyter: :material-download:](../../paper_tables_and_figures/table8_augmentation-at-loading_on_other_datasets/table8_augmentation-at-loading_on_other_datasets.ipynb)


```python
import pathlib

import pandas as pd

AUGMENTATIONS_ORDER = [
    "noaug",
    "rotate",
    "horizontalflip",
    "colorjitter",
    "packetloss",
    "timeshift",
    "changertt",
]

RENAME = {
    "noaug": "No augmentation",
    "changertt": "Change rtt",
    "horizontalflip": "Horizontal flip",
    "colorjitter": "Color jitter",
    "packetloss": "Packet loss",
    "rotate": "Rotate",
    "timeshift": "Time shift",
}
```


```python
def load_summary_report(fname, level0):
    df = pd.read_csv(fname, header=[0, 1], index_col=[0, 1]).droplevel(0, axis=0)
    df = df["f1"]
    df = df[["mean", "ci95"]]
    df = df.loc[AUGMENTATIONS_ORDER].rename(RENAME)
    df.columns = pd.MultiIndex.from_arrays([[level0, level0], df.columns])

    return df
```


```python
df = pd.concat(
    (
        load_summary_report(
            "campaigns/mirage22/augmentation-at-loading-no-dropout/minpkts10/campaign_summary/augment-at-loading/summary_flowpic_dim_32.csv",
            "mirage22 - minpkts10",
        ),
        load_summary_report(
            "campaigns/mirage22/augmentation-at-loading-no-dropout/minpkts1000/campaign_summary/augment-at-loading/summary_flowpic_dim_32.csv",
            "mirage22 - minpkts1000",
        ),
        load_summary_report(
            "campaigns/utmobilenet21/augmentation-at-loading-no-dropout/minpkts10/campaign_summary/augment-at-loading/summary_flowpic_dim_32.csv",
            "utmobilenet21 - minpkts10",
        ),
        load_summary_report(
            "campaigns/mirage19/augmentation-at-loading-no-dropout/minpkts10/campaign_summary/augment-at-loading/summary_flowpic_dim_32.csv",
            "mirage19 - minpkts10",
        ),
    ),
    axis=1,
)
df = (df * 100).round(2)
display(df)
df.to_csv("table8_augmentation-at-loading_on_other_datasets.csv")
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
      <th colspan="2" halign="left">mirage22 - minpkts10</th>
      <th colspan="2" halign="left">mirage22 - minpkts1000</th>
      <th colspan="2" halign="left">utmobilenet21 - minpkts10</th>
      <th colspan="2" halign="left">mirage19 - minpkts10</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No augmentation</th>
      <td>90.97</td>
      <td>1.15</td>
      <td>83.35</td>
      <td>3.13</td>
      <td>86.52</td>
      <td>1.89</td>
      <td>69.91</td>
      <td>1.57</td>
    </tr>
    <tr>
      <th>Rotate</th>
      <td>88.25</td>
      <td>1.20</td>
      <td>87.32</td>
      <td>2.24</td>
      <td>89.00</td>
      <td>0.89</td>
      <td>60.35</td>
      <td>1.17</td>
    </tr>
    <tr>
      <th>Horizontal flip</th>
      <td>91.90</td>
      <td>0.84</td>
      <td>83.82</td>
      <td>2.26</td>
      <td>86.20</td>
      <td>2.57</td>
      <td>69.78</td>
      <td>1.28</td>
    </tr>
    <tr>
      <th>Color jitter</th>
      <td>89.77</td>
      <td>1.16</td>
      <td>81.40</td>
      <td>3.62</td>
      <td>88.42</td>
      <td>1.09</td>
      <td>67.00</td>
      <td>1.11</td>
    </tr>
    <tr>
      <th>Packet loss</th>
      <td>92.34</td>
      <td>1.10</td>
      <td>87.19</td>
      <td>2.52</td>
      <td>85.03</td>
      <td>1.70</td>
      <td>67.55</td>
      <td>1.46</td>
    </tr>
    <tr>
      <th>Time shift</th>
      <td>92.80</td>
      <td>1.21</td>
      <td>86.73</td>
      <td>3.88</td>
      <td>89.18</td>
      <td>1.30</td>
      <td>70.33</td>
      <td>1.26</td>
    </tr>
    <tr>
      <th>Change rtt</th>
      <td>93.75</td>
      <td>0.83</td>
      <td>91.48</td>
      <td>2.12</td>
      <td>88.25</td>
      <td>1.34</td>
      <td>74.28</td>
      <td>1.22</td>
    </tr>
  </tbody>
</table>
</div>

