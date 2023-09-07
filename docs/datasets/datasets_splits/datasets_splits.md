The splits described here are specific for our submission
and the aim to replicate the previous IMC22 paper.


### ucdavis-icdm19

Differently from the other datasets inhere described,
`ucdavis-icdm19` does NOT require any filtering/adaptation
after transforming the original CSV into a monolithic parquet.

The testing partition are also predefined ("human" and "script").

We need however to define splits of 100 samples per class
for modeling. To do so we perform a random shuffle of 
the data and generate 5 non overlapping groups of 100 samples.

```
python datasets/generate_splits.py --config config.yml
```

???+ note "output"
    ```
    loading: datasets/ucdavis-icdm19/ucdavis-icdm19.parquet
    saving: datasets/ucdavis-icdm19/train_split_0.parquet
    saving: datasets/ucdavis-icdm19/train_split_1.parquet
    saving: datasets/ucdavis-icdm19/train_split_2.parquet
    saving: datasets/ucdavis-icdm19/train_split_3.parquet
    saving: datasets/ucdavis-icdm19/train_split_4.parquet
    loading: datasets/ucdavis-icdm19/ucdavis-icdm19.parquet
    saving: datasets/ucdavis-icdm19/test_split_human.parquet
    saving: datasets/ucdavis-icdm19/test_split_script.parquet
    ```

