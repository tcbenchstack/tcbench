# Table 7: (G3) Data augmentation in supervised setting on other datasets.

[:simple-jupyter: :material-download:](../../paper_tables_and_figures/table7_augmentation-at-loading_on_other_datasets/table7_augmentation-at-loading_on_other_datasets.ipynb)


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
    df = (
        pd.read_csv(fname)
        .drop("test_split_name", axis=1)
        .set_index("aug_name", drop=True)
    )
    df = df[["f1", "f1.1", "f1.2", "f1.3"]]
    df.columns = df.iloc[0].values
    df = df.iloc[1:]
    df = df[["mean", "ci95"]].astype(float).round(4) * 100
    df = df.loc[AUGMENTATIONS_ORDER].rename(RENAME)
    df.columns = pd.MultiIndex.from_arrays([[level0, level0], df.columns])

    return df
```


```python
pd.concat(
    (
        load_summary_report(
            "campaigns/mirage22/augmentation-at-loading-no-dropout/minpkts10/campaign_summary/1684958367/summary_flowpic_dim_32.csv",
            "mirage22 - minpkts10",
        ),
        load_summary_report(
            "campaigns/mirage22/augmentation-at-loading-no-dropout/minpkts1000/campaign_summary/1684958367/summary_flowpic_dim_32.csv",
            "mirage22 - minpkts1000",
        ),
        load_summary_report(
            "campaigns/utmobilenet21/augmentation-at-loading-no-dropout/minpkts10/campaign_summary/1684958367/summary_flowpic_dim_32.csv",
            "utmobilenet21 - minpkts10",
        ),
        load_summary_report(
            "campaigns/mirage19/augmentation-at-loading-no-dropout/minpkts10/campaign_summary/1684958367/summary_flowpic_dim_32.csv",
            "mirage19 - minpkts10",
        ),
    ),
    axis=1,
)
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

    .dataframe thead tr:last-of-type th {
        text-align: right;
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No augmentation</th>
      <td>90.31</td>
      <td>1.08</td>
      <td>79.85</td>
      <td>2.80</td>
      <td>84.23</td>
      <td>1.82</td>
      <td>69.61</td>
      <td>1.62</td>
    </tr>
    <tr>
      <th>Rotate</th>
      <td>88.95</td>
      <td>1.32</td>
      <td>86.54</td>
      <td>2.84</td>
      <td>88.29</td>
      <td>1.04</td>
      <td>61.20</td>
      <td>1.39</td>
    </tr>
    <tr>
      <th>Horizontal flip</th>
      <td>91.07</td>
      <td>0.84</td>
      <td>81.47</td>
      <td>3.17</td>
      <td>85.93</td>
      <td>1.89</td>
      <td>70.35</td>
      <td>1.41</td>
    </tr>
    <tr>
      <th>Color jitter</th>
      <td>89.40</td>
      <td>1.57</td>
      <td>80.69</td>
      <td>2.94</td>
      <td>86.22</td>
      <td>2.22</td>
      <td>67.48</td>
      <td>1.71</td>
    </tr>
    <tr>
      <th>Packet loss</th>
      <td>91.91</td>
      <td>0.96</td>
      <td>84.20</td>
      <td>3.79</td>
      <td>85.79</td>
      <td>1.23</td>
      <td>67.50</td>
      <td>1.91</td>
    </tr>
    <tr>
      <th>Time shift</th>
      <td>92.53</td>
      <td>0.84</td>
      <td>84.67</td>
      <td>3.71</td>
      <td>88.64</td>
      <td>1.03</td>
      <td>70.68</td>
      <td>1.64</td>
    </tr>
    <tr>
      <th>Change rtt</th>
      <td>94.11</td>
      <td>0.75</td>
      <td>90.96</td>
      <td>1.77</td>
      <td>88.55</td>
      <td>1.63</td>
      <td>74.07</td>
      <td>1.48</td>
    </tr>
  </tbody>
</table>
</div>


