# `mirage19`

Below we report the samples count for each version of the dataset.

!!! tip "Semantic of the splits"

    The split available for this datasets relate to our [:material-file-document-outline: IMC23 paper](/papers/imc23).

### unfiltered

The unfitered version contains all data before curation.

```
tcbench datasets samples-count --name mirage19
```

!!! note "Output"
	```
    unfiltered
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
    ┃ app                         ┃ samples ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
    │ com.waze                    │   11737 │
    │ de.motain.iliga             │   10810 │
    │ com.accuweather.android     │   10631 │
    │ com.duolingo                │    8319 │
    │ it.subito                   │    8167 │
    │ com.contextlogic.wish       │    6507 │
    │ com.spotify.music           │    6431 │
    │ com.joelapenna.foursquared  │    6399 │
    │ com.google.android.youtube  │    6346 │
    │ com.iconology.comics        │    5516 │
    │ com.facebook.katana         │    5368 │
    │ com.dropbox.android         │    4815 │
    │ com.twitter.android         │    4734 │
    │ background                  │    4439 │
    │ com.pinterest               │    4078 │
    │ com.facebook.orca           │    4018 │
    │ com.tripadvisor.tripadvisor │    3572 │
    │ air.com.hypah.io.slither    │    3088 │
    │ com.viber.voip              │    2740 │
    │ com.trello                  │    2306 │
    │ com.groupon                 │    1986 │
    ├─────────────────────────────┼─────────┤
    │ __total__                   │  122007 │
    └─────────────────────────────┴─────────┘
	```


### First training split

```
tcbench datasets samples-count --name mirage19 --split 0
```

!!! note "Output"
	```
    min_pkts: 10, split: 0
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
    ┃ app                         ┃ train_samples ┃ val_samples ┃ test_samples ┃ all_samples ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
    │ de.motain.iliga             │          6079 │         675 │          751 │        7505 │
    │ com.waze                    │          5844 │         649 │          721 │        7214 │
    │ com.duolingo                │          3712 │         413 │          458 │        4583 │
    │ it.subito                   │          3482 │         387 │          430 │        4299 │
    │ com.contextlogic.wish       │          3181 │         353 │          393 │        3927 │
    │ com.accuweather.android     │          3027 │         336 │          374 │        3737 │
    │ com.joelapenna.foursquared  │          2938 │         326 │          363 │        3627 │
    │ com.spotify.music           │          2673 │         297 │          330 │        3300 │
    │ com.dropbox.android         │          2583 │         287 │          319 │        3189 │
    │ com.facebook.katana         │          2331 │         259 │          288 │        2878 │
    │ com.iconology.comics        │          2278 │         253 │          281 │        2812 │
    │ com.twitter.android         │          2272 │         252 │          281 │        2805 │
    │ com.google.android.youtube  │          2209 │         246 │          273 │        2728 │
    │ com.pinterest               │          1984 │         221 │          245 │        2450 │
    │ com.tripadvisor.tripadvisor │          1662 │         185 │          205 │        2052 │
    │ com.facebook.orca           │          1444 │         161 │          178 │        1783 │
    │ com.viber.voip              │          1310 │         146 │          162 │        1618 │
    │ com.trello                  │          1197 │         133 │          148 │        1478 │
    │ com.groupon                 │           951 │         106 │          117 │        1174 │
    │ air.com.hypah.io.slither    │           821 │          91 │          101 │        1013 │
    ├─────────────────────────────┼───────────────┼─────────────┼──────────────┼─────────────┤
    │ __total__                   │         51978 │        5776 │         6418 │       64172 │
    └─────────────────────────────┴───────────────┴─────────────┴──────────────┴─────────────┘
	```
