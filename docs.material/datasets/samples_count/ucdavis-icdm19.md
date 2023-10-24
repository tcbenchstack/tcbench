# `ucdavis-icdm19`

Below we report the samples count for each version of the dataset.

!!! tip "Semantic of the splits"

    The split available for this datasets relate to our [:material-file-document-outline: IMC23 paper](/tcbench/papers/imc23).

### unfiltered

The unfitered version contains all data before curation.

```
tcbench datasets samples-count --name ucdavis-icdm19
```

!!! note "Output"
	```
	unfiltered
	┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━┓
	┃ partition                   ┃ app           ┃ samples ┃
	┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━┩
	│ pretraining                 │ google-doc    │    1221 │
	│                             │ google-drive  │    1634 │
	│                             │ google-music  │     592 │
	│                             │ google-search │    1915 │
	│                             │ youtube       │    1077 │
	│                             │ __total__     │    6439 │
	├─────────────────────────────┼───────────────┼─────────┤
	│ retraining-human-triggered  │ google-doc    │      15 │
	│                             │ google-drive  │      18 │
	│                             │ google-music  │      15 │
	│                             │ google-search │      15 │
	│                             │ youtube       │      20 │
	│                             │ __total__     │      83 │
	├─────────────────────────────┼───────────────┼─────────┤
	│ retraining-script-triggered │ google-doc    │      30 │
	│                             │ google-drive  │      30 │
	│                             │ google-music  │      30 │
	│                             │ google-search │      30 │
	│                             │ youtube       │      30 │
	│                             │ __total__     │     150 │
	└─────────────────────────────┴───────────────┴─────────┘
	```


### First training split

```
tcbench datasets samples-count --name ucdavis-icdm19 --split 0
```

!!! note "Output"
	```
	filtered, split: 0
	┏━━━━━━━━━━━━━━━┳━━━━━━━━━┓
	┃ app           ┃ samples ┃
	┡━━━━━━━━━━━━━━━╇━━━━━━━━━┩
	│ google-doc    │     100 │
	│ google-drive  │     100 │
	│ google-music  │     100 │
	│ google-search │     100 │
	│ youtube       │     100 │
	├───────────────┼─────────┤
	│ __total__     │     500 │
	└───────────────┴─────────┘
	```

### `human` test split

This is equivalent to the `human` partition of the unfiltered dataset.

```
tcbench datasets samples-count --name ucdavis-icdm19 --split human
```

!!! note "Output"
	```
	filtered, split: human
	┏━━━━━━━━━━━━━━━┳━━━━━━━━━┓
	┃ app           ┃ samples ┃
	┡━━━━━━━━━━━━━━━╇━━━━━━━━━┩
	│ youtube       │      20 │
	│ google-drive  │      18 │
	│ google-doc    │      15 │
	│ google-music  │      15 │
	│ google-search │      15 │
	├───────────────┼─────────┤
	│ __total__     │      83 │
	└───────────────┴─────────┘
	```

### `script` test split

This is equivalent to the `script` partition of the unfiltered dataset.

```
tcbench datasets samples-count --name ucdavis-icdm19 --split script
```

!!! note "Output"
    ```
    filtered, split: script
    ┏━━━━━━━━━━━━━━━┳━━━━━━━━━┓
    ┃ app           ┃ samples ┃
    ┡━━━━━━━━━━━━━━━╇━━━━━━━━━┩
    │ google-doc    │      30 │
    │ google-drive  │      30 │
    │ google-music  │      30 │
    │ google-search │      30 │
    │ youtube       │      30 │
    ├───────────────┼─────────┤
    │ __total__     │     150 │
    └───────────────┴─────────┘
    ```
