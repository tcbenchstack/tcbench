# `mirage22`

Below we report the samples count for each version of the dataset.

!!! tip "Semantic of the splits"

    The split available for this datasets relate to our [:material-file-document-outline: IMC23 paper](/tcbench/papers/imc23).

### unfiltered

The unfitered version contains all data before curation.

```
tcbench datasets samples-count --name mirage22
```

!!! note "Output"
	```
    unfiltered
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
    ┃ app                              ┃ samples ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
    │ background                       │   18882 │
    │ com.microsoft.teams              │    6541 │
    │ com.skype.raider                 │    6203 │
    │ us.zoom.videomeetings            │    5066 │
    │ com.cisco.webex.meetings         │    4789 │
    │ com.discord                      │    4337 │
    │ com.facebook.orca                │    4321 │
    │ com.gotomeeting                  │    3695 │
    │ com.Slack                        │    2985 │
    │ com.google.android.apps.meetings │    2252 │
    ├──────────────────────────────────┼─────────┤
    │ __total__                        │   59071 │
    └──────────────────────────────────┴─────────┘
	```


### First training split @ `min_pkts=10`

```
tcbench datasets samples-count --name mirage22 --min-pkts 10 --split 0
```

!!! note "Output"
	```
    min_pkts: 10, split: 0
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
    ┃ app                              ┃ train_samples ┃ val_samples ┃ test_samples ┃ all_samples ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
    │ com.cisco.webex.meetings         │          3594 │         399 │          444 │        4437 │
    │ com.skype.raider                 │          3334 │         371 │          412 │        4117 │
    │ com.microsoft.teams              │          3124 │         347 │          386 │        3857 │
    │ us.zoom.videomeetings            │          2905 │         323 │          359 │        3587 │
    │ com.discord                      │          2743 │         305 │          339 │        3387 │
    │ com.facebook.orca                │          2125 │         236 │          262 │        2623 │
    │ com.gotomeeting                  │          2072 │         230 │          255 │        2557 │
    │ com.google.android.apps.meetings │          1002 │         112 │          124 │        1238 │
    │ com.Slack                        │           786 │          87 │           97 │         970 │
    ├──────────────────────────────────┼───────────────┼─────────────┼──────────────┼─────────────┤
    │ __total__                        │         21685 │        2410 │         2678 │       26773 │
    └──────────────────────────────────┴───────────────┴─────────────┴──────────────┴─────────────┘
	```

### First training split @ `min_pkts=1000`

```
tcbench datasets samples-count --name mirage22 --min-pkts 1000 --split 0
```

!!! note "Output"
	```
    min_pkts: 1000, split: 0
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
    ┃ app                              ┃ train_samples ┃ val_samples ┃ test_samples ┃ all_samples ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
    │ com.discord                      │          1798 │         200 │          222 │        2220 │
    │ us.zoom.videomeetings            │           344 │          39 │           42 │         425 │
    │ com.google.android.apps.meetings │           307 │          34 │           38 │         379 │
    │ com.microsoft.teams              │           260 │          29 │           32 │         321 │
    │ com.gotomeeting                  │           240 │          27 │           30 │         297 │
    │ com.facebook.orca                │           227 │          25 │           28 │         280 │
    │ com.cisco.webex.meetings         │           210 │          23 │           26 │         259 │
    │ com.Slack                        │           160 │          18 │           20 │         198 │
    │ com.skype.raider                 │           154 │          17 │           19 │         190 │
    ├──────────────────────────────────┼───────────────┼─────────────┼──────────────┼─────────────┤
    │ __total__                        │          3700 │         412 │          457 │        4569 │
    └──────────────────────────────────┴───────────────┴─────────────┴──────────────┴─────────────┘
	```

