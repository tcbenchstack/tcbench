---
icon: octicons/number-24
---

# Samples count report

An important dataset property to keep an eye
on when aiming for modeling is the number of 
samples for each class available in the datasets.

You can easily recover this using the `datasets samples-count` subcommand.

For instance, 
the following command computes the samples count for the *unfitered* 
version of the [`ucdavis-icdm19`](/datasets/install/ucdavis-icdm19) dataset.

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

While to obtain the breakdown of the first train split

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

...or the `human` test split

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
