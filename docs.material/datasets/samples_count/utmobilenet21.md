# `utmobilenet21`

Below we report the samples count for each version of the dataset.

!!! tip "Semantic of the splits"

    The split available for this datasets relate to our [:material-file-document-outline: IMC23 paper](/tcbench/papers/imc23).

### unfiltered

The unfitered version contains all data before curation.

```
tcbench datasets samples-count --name utmobilenet21
```

!!! note "Output"
	```
    unfiltered
    ┏━━━━━━━━━━━━━━┳━━━━━━━━━┓
    ┃ app          ┃ samples ┃
    ┡━━━━━━━━━━━━━━╇━━━━━━━━━┩
    │ youtube      │    5591 │
    │ reddit       │    4370 │
    │ google-maps  │    4347 │
    │ spotify      │    2550 │
    │ netflix      │    2237 │
    │ pinterest    │    2165 │
    │ hulu         │    1839 │
    │ instagram    │    1778 │
    │ dropbox      │    1752 │
    │ facebook     │    1654 │
    │ twitter      │    1494 │
    │ gmail        │    1133 │
    │ pandora      │     949 │
    │ messenger    │     837 │
    │ google-drive │     803 │
    │ hangout      │     720 │
    │ skype        │     159 │
    ├──────────────┼─────────┤
    │ __total__    │   34378 │
    └──────────────┴─────────┘
	```


### First training split

```
tcbench datasets samples-count --name utmobilenet21 --split 0
```

!!! note "Output"
	```
    min_pkts: 10, split: 0
    ┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
    ┃ app          ┃ train_samples ┃ val_samples ┃ test_samples ┃ all_samples ┃
    ┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
    │ youtube      │          2021 │         225 │          250 │        2496 │
    │ google-maps  │          1456 │         162 │          180 │        1798 │
    │ hulu         │           947 │         105 │          117 │        1169 │
    │ reddit       │           661 │          73 │           82 │         816 │
    │ spotify      │           538 │          60 │           66 │         664 │
    │ netflix      │           391 │          44 │           48 │         483 │
    │ pinterest    │           353 │          39 │           44 │         436 │
    │ twitter      │           296 │          33 │           36 │         365 │
    │ instagram    │           222 │          25 │           27 │         274 │
    │ hangout      │           206 │          23 │           25 │         254 │
    │ dropbox      │           193 │          21 │           24 │         238 │
    │ pandora      │           162 │          18 │           20 │         200 │
    │ facebook     │           111 │          12 │           14 │         137 │
    │ google-drive │           105 │          12 │           13 │         130 │
    ├──────────────┼───────────────┼─────────────┼──────────────┼─────────────┤
    │ __total__    │          7662 │         852 │          946 │        9460 │
    └──────────────┴───────────────┴─────────────┴──────────────┴─────────────┘
	```
