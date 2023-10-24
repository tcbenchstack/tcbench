
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
# Tutorial: loading datasets APIs

[:simple-jupyter: :material-download:](/tcbench/datasets/guides/tutorial_load_datasets.ipynb)

Let's import `tcbench` and map its alias `tcb`

The module automatically import a few functions and constants.


```python
import tcbench as tcb
```
## The `.get_datasets_root_folder()` method

You can first discover the <root> path where the datasets are
installed using `.get_datasets_root_folder()`


```python
root_folder = tcb.get_datasets_root_folder()
root_folder
```



<pre><code class="outputcode">PosixPath('./envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets')
</code></pre>


The function returns a [`pathlib` path](https://docs.python.org/3/library/pathlib.html?highlight=pathlib)
so you can take advantage of it to navigate the subfolders structure.

For instance:


```python
list(root_folder.iterdir())
```



<pre><code class="outputcode">[PosixPath('./envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/utmobilenet21.BACKUP'),
PosixPath('./envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/utmobilenet21'),
PosixPath('./envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/mirage22'),
PosixPath('./envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19'),
PosixPath('./envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/mirage19')]
</code></pre>


As from the output, each dataset is mapped to a different folder 
named after the dataset itself. Meaning, again taking advantage of `pathlib`, 
you can compose path based on strings.

For instance:


```python
list((root_folder / 'ucdavis-icdm19').iterdir())
```



<pre><code class="outputcode">[PosixPath('./envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19/preprocessed'),
PosixPath('./envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19/raw')]
</code></pre>


## The `.DATASETS` enum

A more polished way to reference datasets is via the `tcbench.DATASETS` attribute which corresponds to a [python enumeration](https://docs.python.org/3/library/enum.html?highlight=enum#enum.Enum) object


```python
type(tcb.DATASETS), list(tcb.DATASETS)
```



<pre><code class="outputcode">(enum.EnumMeta,
[< DATASETS.UCDAVISICDM19: 'ucdavis-icdm19' >,
< DATASETS.UTMOBILENET21: 'utmobilenet21' >,
< DATASETS.MIRAGE19: 'mirage19' >,
< DATASETS.MIRAGE22: 'mirage22' >])
</code></pre>


## The `.get_dataset_folder()` method

For instance, you can bypass the composition of a dataset folder path
and call directly `.get_dataset_folder()` to find the specific 
dataset folder you look for.


```python
dataset_folder = tcb.get_dataset_folder(tcb.DATASETS.UCDAVISICDM19)
dataset_folder
```



<pre><code class="outputcode">PosixPath('./envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19')
</code></pre>


## Listing files

Via `pathlib` you can easily discover all parquet files composing a dataset


```python
list(dataset_folder.rglob('*.parquet'))
```



<pre><code class="outputcode">[PosixPath('./envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19/preprocessed/ucdavis-icdm19.parquet'),
PosixPath('./envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19/preprocessed/imc23/train_split_0.parquet'),
PosixPath('./envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19/preprocessed/imc23/train_split_1.parquet'),
PosixPath('./envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19/preprocessed/imc23/test_split_human.parquet'),
PosixPath('./envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19/preprocessed/imc23/train_split_4.parquet'),
PosixPath('./envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19/preprocessed/imc23/train_split_3.parquet'),
PosixPath('./envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19/preprocessed/imc23/test_split_script.parquet'),
PosixPath('./envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19/preprocessed/imc23/train_split_2.parquet')]
</code></pre>


But you can also programmatically call the the `datasets lsparquet` subcommand of the CLI using `get_rich_tree_parquet_files()`


```python
from tcbench.libtcdatasets.datasets_utils import get_rich_tree_parquet_files
get_rich_tree_parquet_files(tcb.DATASETS.UCDAVISICDM19)
```



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"class="outputcode">Datasets
└── ucdavis-icdm19
    └── 📁 preprocessed/
        ├── ucdavis-icdm19.parquet
        ├── LICENSE
        └── 📁 imc23/
            ├── test_split_human.parquet
            ├── test_split_script.parquet
            ├── train_split_0.parquet
            ├── train_split_1.parquet
            ├── train_split_2.parquet
            ├── train_split_3.parquet
            └── train_split_4.parquet
</pre>



## The `.load_parquet()` method

Finally, the generic `.load_parquet()` can be used to load one of the parquet files.

For instance, the following load the unfiltered monolithic file of the `ucdavis-icdm19` dataset


```python
df = tcb.load_parquet(tcb.DATASETS.UCDAVISICDM19)
```

```python
df.head(2)
```



<div class="md-typeset__scrollwrap">
<div class="md-typeset__table">
<table>
<thead>
<tr style="text-align: right;">
<th></th>
<th>row_id</th>
<th>app</th>
<th>flow_id</th>
<th>partition</th>
<th>num_pkts</th>
<th>duration</th>
<th>bytes</th>
<th>unixtime</th>
<th>timetofirst</th>
<th>pkts_size</th>
<th>pkts_dir</th>
<th>pkts_iat</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>0</td>
<td>google-doc</td>
<td>GoogleDoc-100</td>
<td>pretraining</td>
<td>2925</td>
<td>116.348</td>
<td>816029</td>
<td>[1527993495.652867, 1527993495.685678, 1527993...</td>
<td>[0.0, 0.0328109, 0.261392, 0.262656, 0.263943,...</td>
<td>[354, 87, 323, 1412, 1412, 107, 1412, 180, 141...</td>
<td>[1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, ...</td>
<td>[0.0, 0.0328109, 0.2285811, 0.0012639999999999...</td>
</tr>
<tr>
<th>1</th>
<td>1</td>
<td>google-doc</td>
<td>GoogleDoc-1000</td>
<td>pretraining</td>
<td>2813</td>
<td>116.592</td>
<td>794628</td>
<td>[1527987720.40456, 1527987720.422811, 15279877...</td>
<td>[0.0, 0.0182509, 0.645106, 0.646344, 0.647689,...</td>
<td>[295, 87, 301, 1412, 1412, 1412, 180, 113, 141...</td>
<td>[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, ...</td>
<td>[0.0, 0.0182509, 0.6268551, 0.0012380000000000...</td>
</tr>
</tbody>
</table>
</div>
</div>



```python
df.groupby(['partition', 'app'])['app'].value_counts()
```



<pre><code class="outputcode">partition                    app
pretraining                  google-doc       1221
google-drive     1634
google-music      592
google-search    1915
youtube          1077
retraining-human-triggered   google-doc         15
google-drive       18
google-music       15
google-search      15
youtube            20
retraining-script-triggered  google-doc         30
google-drive       30
google-music       30
google-search      30
youtube            30
Name: count, dtype: int64
</code></pre>


Beside the dataset name, the function only has 2 other parameters, but
their semantic and values are "mingled" with the curation process adopted.


```python
tcb.load_parquet?
```

<pre><code class="outputcode"><span class="ansi-red-fg">Signature:</span>
tcb<span class="ansi-blue-fg">.</span>load_parquet<span class="ansi-blue-fg">(</span>
    dataset_name<span class="ansi-blue-fg">:</span> <span class="ansi-blue-fg">'str | DATASETS'</span><span class="ansi-blue-fg">,</span>
    min_pkts<span class="ansi-blue-fg">:</span> <span class="ansi-blue-fg">'int'</span> <span class="ansi-blue-fg">=</span> <span class="ansi-blue-fg">-</span><span class="ansi-cyan-fg">1</span><span class="ansi-blue-fg">,</span>
    split<span class="ansi-blue-fg">:</span> <span class="ansi-blue-fg">'str'</span> <span class="ansi-blue-fg">=</span> <span class="ansi-green-fg">None</span><span class="ansi-blue-fg">,</span>
    columns<span class="ansi-blue-fg">:</span> <span class="ansi-blue-fg">'List[str]'</span> <span class="ansi-blue-fg">=</span> <span class="ansi-green-fg">None</span><span class="ansi-blue-fg">,</span>
    animation<span class="ansi-blue-fg">:</span> <span class="ansi-blue-fg">'bool'</span> <span class="ansi-blue-fg">=</span> <span class="ansi-green-fg">False</span><span class="ansi-blue-fg">,</span>
<span class="ansi-blue-fg">)</span> <span class="ansi-blue-fg">-&gt;</span> <span class="ansi-blue-fg">'pd.DataFrame'</span>
<span class="ansi-red-fg">Docstring:</span>
Load and returns a dataset parquet file

Arguments:
    dataset_name: The name of the dataset
    min_pkts: the filtering rule applied when curating the datasets.
        If -1, load the unfiltered dataset
    split: if min_pkts!=-1, is used to request the loading of
        the split file. For DATASETS.UCDAVISICDM19
        values can be "human", "script" or a number
        between 0 and 4.
        For all other dataset split can be anything
        which is not None (e.g., True)
    columns: A list of columns to load (if None, load all columns)
    animation: if True, create a loading animation on the console

Returns:
    A pandas dataframe and the related parquet file used to load the dataframe
<span class="ansi-red-fg">File:</span>      ~/.conda/envs/super-tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets_utils.py
<span class="ansi-red-fg">Type:</span>      function
</code></pre>

## How `.load_parquet()` maps to parquet files

The logic to follow to load specific files can be confusing. The table below report a global view across datasets:

| Dataset | min_pkts=-1 | min_pkts=10 | min_pkts=1000 | split=True | split=0..4 | split=human | split=script |
|:-------:|:-------------:|:-------------:|:---------------:|:------------:|:------------:|:-------------:|:--------------:|
|`ucdavis-icdm19`| yes | - | - | - | yes (train+val) | yes (test)| yes (test)|
|`mirage19`| yes | yes| - | yes (train/val/test) | - | - | - |
|`mirage22`| yes | yes |yes|yes (train/val/test) | - | - | - | 
|`utmobilenet21`| yes | yes |-|yes (train/val/test) | - | - | - |

* `min_pkts=-1` is set by default and corresponds to loading the unfiltered parquet files, i.e., the files stored immediately under `/preprocessed`. All other files are stored under the `imc23` subfolders

* For `ucdavis-icdm19`, the parameter `min_pkts` is not used. The loading of training(+validation) and test data is controlled by `split`

* For all other datasets, `min_pkts` specifies which filtered version of the data to use, while `split=True` load the split indexes

## Loading `ucdavis-icdm19`

For instance, to load the `human` test split of `ucdavid-icdm19` you can run


```python
df = tcb.load_parquet(tcb.DATASETS.UCDAVISICDM19, split='human')
df['app'].value_counts()
```



<pre><code class="outputcode">app
youtube          20
google-drive     18
google-doc       15
google-music     15
google-search    15
Name: count, dtype: int64
</code></pre>


And the logic is very similar for the `script` partition


```python
df = tcb.load_parquet(tcb.DATASETS.UCDAVISICDM19, split='script')
df['app'].value_counts()
```



<pre><code class="outputcode">app
google-doc       30
google-drive     30
google-music     30
google-search    30
youtube          30
Name: count, dtype: int64
</code></pre>


However to load a specific train split


```python
df = tcb.load_parquet(tcb.DATASETS.UCDAVISICDM19, split='0')
df['app'].value_counts()
```



<pre><code class="outputcode">app
google-doc       100
google-drive     100
google-music     100
google-search    100
youtube          100
Name: count, dtype: int64
</code></pre>


## Loading other datasets

By default, without any parameter beside the dataset name, the function loads the unfiltered version of a dataset


```python
df = tcb.load_parquet(tcb.DATASETS.MIRAGE19)
df.shape
```



<pre><code class="outputcode">(122007, 135)
</code></pre>


Recall the structure of the `mirage19` dataset


```python
get_rich_tree_parquet_files(tcb.DATASETS.MIRAGE19)
```



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"class="outputcode">Datasets
└── mirage19
    └── 📁 preprocessed/
        ├── mirage19.parquet
        └── 📁 imc23/
            ├── mirage19_filtered_minpkts10.parquet
            └── mirage19_filtered_minpkts10_splits.parquet
</pre>



So there is only one filtering with `min_pkts=10`


```python
df = tcb.load_parquet(tcb.DATASETS.MIRAGE19, min_pkts=10)
df.shape
```



<pre><code class="outputcode">(64172, 20)
</code></pre>


Based on the dataframe shape, we can see that (indeed) we loaded a reduced version of the unfiltered dataset.

While for `ucdavis-icdm19` the "split" files represent 100 samples selected for training (because there are two ad-hoc test split), for all other dataset the "split" files contains indexes indicating the rows to use for train/val/test.

Thus, issuing `split=True` is enough to indicate the need to load the split table.


```python
df_split = tcb.load_parquet(tcb.DATASETS.MIRAGE19, min_pkts=10, split=True)
```
