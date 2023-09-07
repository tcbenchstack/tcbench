# Table 2 : Datasets properties

[:simple-jupyter: :material-download:](../../paper_tables_and_figures/table2_datasets_properties/table2_datasets_properties.ipynb)


```python
import pandas as pd
import tcbench as tcb
```

## ucdavis-icdm19


```python
df = tcb.load_parquet(tcb.DATASETS.UCDAVISICDM19)

# add number of packets
df = df.assign(packets=df["pkts_size"].apply(len))

# number of samples
df_tmp = pd.DataFrame(
    df.groupby(["partition", "app"])["app"].value_counts()
).reset_index()
df_tmp = df_tmp.pivot(index="partition", columns="app", values="count")
df_tmp = df_tmp.assign(
    count=df_tmp.sum(axis=1),
    flows_min=df_tmp.min(axis=1),
    flows_max=df_tmp.max(axis=1),
    rho=(df_tmp.max(axis=1) / df_tmp.min(axis=1)).round(1),
    classes=len(df["app"].cat.categories),
)

# mean pkts per flow
mean_pkts = df.groupby("partition")["packets"].mean().round(0)
mean_pkts.name = "mean_pkts"
flows_all = df.groupby("partition")["partition"].count()
flows_all.name = "flows_all"

# combining everything together
df_tmp = pd.concat((df_tmp, mean_pkts, flows_all), axis=1)
df_tmp = df_tmp[["classes", "flows_all", "flows_min", "flows_max", "rho", "mean_pkts"]]
display(df_tmp)

stats_ucdavis19 = df_tmp
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
      <th>classes</th>
      <th>flows_all</th>
      <th>flows_min</th>
      <th>flows_max</th>
      <th>rho</th>
      <th>mean_pkts</th>
    </tr>
    <tr>
      <th>partition</th>
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
      <th>pretraining</th>
      <td>5</td>
      <td>6439</td>
      <td>592</td>
      <td>1915</td>
      <td>3.2</td>
      <td>6653.0</td>
    </tr>
    <tr>
      <th>retraining-human-triggered</th>
      <td>5</td>
      <td>83</td>
      <td>15</td>
      <td>20</td>
      <td>1.3</td>
      <td>7666.0</td>
    </tr>
    <tr>
      <th>retraining-script-triggered</th>
      <td>5</td>
      <td>150</td>
      <td>30</td>
      <td>30</td>
      <td>1.0</td>
      <td>7131.0</td>
    </tr>
  </tbody>
</table>
</div>


## mirage19

The unfiltered version of the dataset has an extra class, which corresponds to `"background"` traffic


```python
# unfiltered
df = tcb.load_parquet(tcb.DATASETS.MIRAGE19)

ser = df["app"].value_counts()
df_unfiltered = pd.DataFrame(
    [
        dict(
            classes=len(ser),
            flows_all=ser.sum(),
            flows_min=ser.min(),
            flows_max=ser.max(),
            rho=(ser.max() / ser.min()).round(1),
            mean_pkts=df["packets"].mean().round(0),
        )
    ],
    index=["unfiltered"],
)
```


```python
# min_pkts = 10
df = tcb.load_parquet(tcb.DATASETS.MIRAGE19, min_pkts=10)

ser = df["app"].value_counts()
df_minpkts10 = pd.DataFrame(
    [
        dict(
            classes=len(ser),
            flows_all=ser.sum(),
            flows_min=ser.min(),
            flows_max=ser.max(),
            rho=(ser.max() / ser.min()).round(1),
            mean_pkts=df["packets"].mean().round(0),
        )
    ],
    index=["min_pkts=10"],
)
```


```python
df_tmp = pd.concat((df_unfiltered, df_minpkts10), axis=0)
display(df_tmp)
stats_mirage19 = df_tmp
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
      <th>classes</th>
      <th>flows_all</th>
      <th>flows_min</th>
      <th>flows_max</th>
      <th>rho</th>
      <th>mean_pkts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>unfiltered</th>
      <td>21</td>
      <td>122007</td>
      <td>1986</td>
      <td>11737</td>
      <td>5.9</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>min_pkts=10</th>
      <td>20</td>
      <td>64172</td>
      <td>1013</td>
      <td>7505</td>
      <td>7.4</td>
      <td>17.0</td>
    </tr>
  </tbody>
</table>
</div>


## mirage22

The unfiltered version of the dataset has an extra class, which corresponds to `"background"` traffic


```python
# unfiltered
df = tcb.load_parquet(tcb.DATASETS.MIRAGE22)

ser = df["app"].value_counts()
df_unfiltered = pd.DataFrame(
    [
        dict(
            classes=len(ser),
            flows_all=ser.sum(),
            flows_min=ser.min(),
            flows_max=ser.max(),
            rho=(ser.max() / ser.min()).round(1),
            mean_pkts=df["packets"].mean().round(0),
        )
    ],
    index=["unfiltered"],
)
```


```python
# min_pkts = 10
df = tcb.load_parquet(tcb.DATASETS.MIRAGE22, min_pkts=10)

ser = df["app"].value_counts()
df_minpkts10 = pd.DataFrame(
    [
        dict(
            classes=len(ser),
            flows_all=ser.sum(),
            flows_min=ser.min(),
            flows_max=ser.max(),
            rho=(ser.max() / ser.min()).round(1),
            mean_pkts=df["packets"].mean().round(0),
        )
    ],
    index=["min_pkts=10"],
)
```


```python
# min_pkts = 1000
df = tcb.load_parquet(tcb.DATASETS.MIRAGE22, min_pkts=1000)

ser = df["app"].value_counts()
df_minpkts1000 = pd.DataFrame(
    [
        dict(
            classes=len(ser),
            flows_all=ser.sum(),
            flows_min=ser.min(),
            flows_max=ser.max(),
            rho=(ser.max() / ser.min()).round(1),
            mean_pkts=df["packets"].mean().round(0),
        )
    ],
    index=["min_pkts=1000"],
)
```


```python
df_tmp = pd.concat((df_unfiltered, df_minpkts10, df_minpkts1000), axis=0)
display(df_tmp)
stats_mirage22 = df_tmp
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
      <th>classes</th>
      <th>flows_all</th>
      <th>flows_min</th>
      <th>flows_max</th>
      <th>rho</th>
      <th>mean_pkts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>unfiltered</th>
      <td>10</td>
      <td>59071</td>
      <td>2252</td>
      <td>18882</td>
      <td>8.4</td>
      <td>3068.0</td>
    </tr>
    <tr>
      <th>min_pkts=10</th>
      <td>9</td>
      <td>26773</td>
      <td>970</td>
      <td>4437</td>
      <td>4.6</td>
      <td>6598.0</td>
    </tr>
    <tr>
      <th>min_pkts=1000</th>
      <td>9</td>
      <td>4569</td>
      <td>190</td>
      <td>2220</td>
      <td>11.7</td>
      <td>38321.0</td>
    </tr>
  </tbody>
</table>
</div>


## utmobilenet21


```python
# unfiltered
df = tcb.load_parquet(tcb.DATASETS.UTMOBILENET21)

ser = df["app"].value_counts()
df_unfiltered = pd.DataFrame(
    [
        dict(
            classes=len(ser),
            flows_all=ser.sum(),
            flows_min=ser.min(),
            flows_max=ser.max(),
            rho=(ser.max() / ser.min()).round(1),
            mean_pkts=df["packets"].mean().round(0),
        )
    ],
    index=["unfiltered"],
)
```


```python
# unfiltered
df = tcb.load_parquet(tcb.DATASETS.UTMOBILENET21, min_pkts=10)

ser = df["app"].value_counts()
df_minpkts10 = pd.DataFrame(
    [
        dict(
            classes=len(ser),
            flows_all=ser.sum(),
            flows_min=ser.min(),
            flows_max=ser.max(),
            rho=(ser.max() / ser.min()).round(1),
            mean_pkts=df["packets"].mean().round(0),
        )
    ],
    index=["minpkts=10"],
)
```


```python
df_tmp = pd.concat((df_unfiltered, df_minpkts10), axis=0)
display(df_tmp)
stats_utmobilenet21 = df_tmp
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
      <th>classes</th>
      <th>flows_all</th>
      <th>flows_min</th>
      <th>flows_max</th>
      <th>rho</th>
      <th>mean_pkts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>unfiltered</th>
      <td>17</td>
      <td>22429</td>
      <td>57</td>
      <td>4716</td>
      <td>82.7</td>
      <td>716.0</td>
    </tr>
    <tr>
      <th>minpkts=10</th>
      <td>10</td>
      <td>5685</td>
      <td>104</td>
      <td>2153</td>
      <td>20.7</td>
      <td>2741.0</td>
    </tr>
  </tbody>
</table>
</div>


# alltogether


```python
df_tmp = pd.concat(
    (
        (stats_ucdavis19.assign(dataset="ucdavis-icdm19")).set_index(
            ["dataset", stats_ucdavis19.index]
        ),
        (stats_mirage19.assign(dataset="mirage19")).set_index(
            ["dataset", stats_mirage19.index]
        ),
        (stats_mirage22.assign(dataset="mirage22")).set_index(
            ["dataset", stats_mirage22.index]
        ),
        (stats_utmobilenet21.assign(dataset="utmobilenet21")).set_index(
            ["dataset", stats_utmobilenet21.index]
        ),
    )
).rename(
    {
        "retraining-human-triggered": "human",
        "retraining-script-triggered": "script",
    },
    axis=0,
)
display(df_tmp)
df_tmp.to_csv("table2_datasets_properties.csv")
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
      <th>classes</th>
      <th>flows_all</th>
      <th>flows_min</th>
      <th>flows_max</th>
      <th>rho</th>
      <th>mean_pkts</th>
    </tr>
    <tr>
      <th>dataset</th>
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
      <th rowspan="3" valign="top">ucdavis-icdm19</th>
      <th>pretraining</th>
      <td>5</td>
      <td>6439</td>
      <td>592</td>
      <td>1915</td>
      <td>3.2</td>
      <td>6653.0</td>
    </tr>
    <tr>
      <th>human</th>
      <td>5</td>
      <td>83</td>
      <td>15</td>
      <td>20</td>
      <td>1.3</td>
      <td>7666.0</td>
    </tr>
    <tr>
      <th>script</th>
      <td>5</td>
      <td>150</td>
      <td>30</td>
      <td>30</td>
      <td>1.0</td>
      <td>7131.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">mirage19</th>
      <th>unfiltered</th>
      <td>21</td>
      <td>122007</td>
      <td>1986</td>
      <td>11737</td>
      <td>5.9</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>min_pkts=10</th>
      <td>20</td>
      <td>64172</td>
      <td>1013</td>
      <td>7505</td>
      <td>7.4</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">mirage22</th>
      <th>unfiltered</th>
      <td>10</td>
      <td>59071</td>
      <td>2252</td>
      <td>18882</td>
      <td>8.4</td>
      <td>3068.0</td>
    </tr>
    <tr>
      <th>min_pkts=10</th>
      <td>9</td>
      <td>26773</td>
      <td>970</td>
      <td>4437</td>
      <td>4.6</td>
      <td>6598.0</td>
    </tr>
    <tr>
      <th>min_pkts=1000</th>
      <td>9</td>
      <td>4569</td>
      <td>190</td>
      <td>2220</td>
      <td>11.7</td>
      <td>38321.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">utmobilenet21</th>
      <th>unfiltered</th>
      <td>17</td>
      <td>22429</td>
      <td>57</td>
      <td>4716</td>
      <td>82.7</td>
      <td>716.0</td>
    </tr>
    <tr>
      <th>minpkts=10</th>
      <td>10</td>
      <td>5685</td>
      <td>104</td>
      <td>2153</td>
      <td>20.7</td>
      <td>2741.0</td>
    </tr>
  </tbody>
</table>
</div>

