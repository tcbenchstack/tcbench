# `mirage22`

The dataset collect traffic from __9 mobile Android apps__ 
(webex, skype, microsoft teams,
zoom, discord, messenger,
gotomeeting, google meetings, slack).

The authors of the dataset (*Guarino et. al*) describe it as follows

=== "Quote"
    :fontawesome-solid-quote-left:
    *The dataset was collected by students and researchers within
    April–June 2021 leveraging the MIRAGE architecture [16]
    (conveniently optimized to capture traffic of communication
    and collaboration apps) in the ARCLAB laboratory at the
    University of Napoli “Federico II”.1 Experimenters used three
    mobile devices: a Google Nexus 6 (Android 10) and two Samsung Galaxy A5 (Android 6.0.1). In each capture session—
    whose duration spanned from 15 to 80 minutes based on
    the activity—the experimenters performed a specific activity,
    so as to obtain a traffic dataset that reflects the common
    usage of considered apps.2 Each session resulted in a PCAP
    traffic trace and additional system log-files with ground-truth
    information. Based on the latter, each biflow3 was reliably
    labeled with the corresponding Android package-name by
    considering established network-connections (via netstat).*

    [...]

    *Communication and collaboration apps—used for business
    meeting, classes, and social interaction—have experienced a
    huge utilization increment when “stay-at-home” orders were
    issued worldwide. Based on both popularity and utilization
    boost, herein we focus on five of them: GotoMeeting (Gm),
    Skype (Sk), Teams (Tm), Webex (Wb), and Zoom (Zm).
    Indeed, Zoom has obtained the steepest increment with its
    traffic scaling to orders of magnitude, followed by Webex,
    GotoMeeting, Teams, BlueJeans (whose traffic we are
    currently collecting), and Skype [17]*
    :fontawesome-solid-quote-right:

=== "Bibtex"
    ```
    @INPROCEEDINGS{guarino2021classification,  
    author={Guarino, Idio and Aceto, Giuseppe and Ciuonzo, Domenico and Montieri, Antonio and Persico, Valerio and PescapÃ©, Antonio},  
    booktitle={2021 IEEE 26th International Workshop on Computer Aided Modeling and Design of Communication Links and Networks (CAMAD)},   
    title={Classification of Communication and Collaboration Apps via Advanced Deep-Learning Approaches},   
    year={2021},  
    volume={},  
    number={},  
    pages={1-6},  
    doi={10.1109/CAMAD52502.2021.9617789}
    }
    ```

As suggested by the name, the dataset is from the same research group of [`mirage19`](/datasets/install/mirage19) 
so the two datasets share many properties.

The major difference is the target of applications as
`mirage22` focuses only on on video meeting Android apps with experiments
annotated with respect to different interactions
the the apps (voice, chat, etc.) while `mirage19`
is more diversified set of apps.


## Raw data

The dataset is a single zip.
Once unpacked it has the following structure
```
MIRAGE-COVID-CCMA-2022
├── Preprocessed_pickle
└── Raw_JSON
    ├── Discord
    ├── GotoMeeting
    ├── Meet
    ├── Messenger
    ├── Skype
    ├── Slack
    ├── Teams
    ├── Webex
    └── Zoom
```

Notice the two subfolders:

* `Raw_JSON` gathers the nested JSON files for each experiment.

* `Preprocessed_pickle` is a pickle serialization of the 
data but unfortunately is undocumented.

## Curation & splits

We follow the same processes described for
[`mirage19` curation](/datasets/install/mirage19/#curation), i.e.,
consolidation and flattening of the JSON files, removal of the 
background, etc.

However, next to the *unfiltered* and *filtered* version imposing
a minimum of 10 packets per flow, we also create a second filtered
version imposing a minimum of 1,000 packets per flow.

Once the parquet files are generate we create 80/10/10
train/validation/test splits with the same process 
described for the [`mirage19` splits](/datasets/install/mirage19/#splits).

## Install

The installation does not requires you to pre-download the dataset tarball
and can be triggered with the following command

```
tcbench datasets install --name mirage22
```

!!! note "Output"
	```
	╭─────────────────╮
	│download & unpack│
	╰─────────────────╯
	Downloading... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.1 GB / 3.1 GB eta 0:00:00
	opening: /tmp/tmp3marsp7l/MIRAGE-COVID-CCMA-2022.zip
	opening: ./envs/tcbench/lib/python3.10/site-packages/libtcdatasets/datasets/mirage22/raw/MIRAGE-COVID-CCMA-2022/Raw_JSON/Discord.zip
	opening: ./envs/tcbench/lib/python3.10/site-packages/libtcdatasets/datasets/mirage22/raw/MIRAGE-COVID-CCMA-2022/Raw_JSON/GotoMeeting.zip
	opening: ./envs/tcbench/lib/python3.10/site-packages/libtcdatasets/datasets/mirage22/raw/MIRAGE-COVID-CCMA-2022/Raw_JSON/Meet.zip
	opening: ./envs/tcbench/lib/python3.10/site-packages/libtcdatasets/datasets/mirage22/raw/MIRAGE-COVID-CCMA-2022/Raw_JSON/Messenger.zip
	opening: ./envs/tcbench/lib/python3.10/site-packages/libtcdatasets/datasets/mirage22/raw/MIRAGE-COVID-CCMA-2022/Raw_JSON/Skype.zip
	opening: ./envs/tcbench/lib/python3.10/site-packages/libtcdatasets/datasets/mirage22/raw/MIRAGE-COVID-CCMA-2022/Raw_JSON/Slack.zip
	opening: ./envs/tcbench/lib/python3.10/site-packages/libtcdatasets/datasets/mirage22/raw/MIRAGE-COVID-CCMA-2022/Raw_JSON/Teams.zip
	opening: ./envs/tcbench/lib/python3.10/site-packages/libtcdatasets/datasets/mirage22/raw/MIRAGE-COVID-CCMA-2022/Raw_JSON/Webex.zip
	opening: ./envs/tcbench/lib/python3.10/site-packages/libtcdatasets/datasets/mirage22/raw/MIRAGE-COVID-CCMA-2022/Raw_JSON/Zoom.zip

	╭──────────╮
	│preprocess│
	╰──────────╯
	found 998 JSON files to load
	Converting JSONs... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 998/998 0:00:28
	merging files...
	saving: ./envs/tcbench/lib/python3.10/site-packages/libtcdatasets/datasets/mirage22/preprocessed/mirage22.parquet

	╭────────────────────────╮
	│filter & generate splits│
	╰────────────────────────╯
	loading: ./envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/mirage22/preprocessed/mirage22.parquet
	samples count : unfiltered
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
	stats : number packets per-flow (unfiltered)
	┏━━━━━━━┳━━━━━━━━━━━┓
	┃ stat  ┃     value ┃
	┡━━━━━━━╇━━━━━━━━━━━┩
	│ count │   59071.0 │
	│ mean  │   3068.32 │
	│ std   │  25416.43 │
	│ min   │       1.0 │
	│ 25%   │      20.0 │
	│ 50%   │      27.0 │
	│ 75%   │      42.0 │
	│ max   │ 1665842.0 │
	└───────┴───────────┘

	filtering min_pkts=10...
	saving: ./envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/mirage22/preprocessed/imc23/mirage22_filtered_minpkts10.parquet
	samples count : filtered (min_pkts=10)
	┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
	┃ app                              ┃ samples ┃
	┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
	│ com.cisco.webex.meetings         │    4437 │
	│ com.skype.raider                 │    4117 │
	│ com.microsoft.teams              │    3857 │
	│ us.zoom.videomeetings            │    3587 │
	│ com.discord                      │    3387 │
	│ com.facebook.orca                │    2623 │
	│ com.gotomeeting                  │    2557 │
	│ com.google.android.apps.meetings │    1238 │
	│ com.Slack                        │     970 │
	├──────────────────────────────────┼─────────┤
	│ __total__                        │   26773 │
	└──────────────────────────────────┴─────────┘
	stats : number packets per-flow (min_pkts=10)
	┏━━━━━━━┳━━━━━━━━━━━┓
	┃ stat  ┃     value ┃
	┡━━━━━━━╇━━━━━━━━━━━┩
	│ count │   26773.0 │
	│ mean  │   6598.23 │
	│ std   │  37290.08 │
	│ min   │      11.0 │
	│ 25%   │      15.0 │
	│ 50%   │      21.0 │
	│ 75%   │     186.0 │
	│ max   │ 1665842.0 │
	└───────┴───────────┘
	saving: ./envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/mirage22/preprocessed/imc23/mirage22_filtered_minpkts10_splits.parquet
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

	filtering min_pkts=1000...
	saving: ./envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/mirage22/preprocessed/imc23/mirage22_filtered_minpkts1000.parquet
	samples count : filtered (min_pkts=1000)
	┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
	┃ app                              ┃ samples ┃
	┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
	│ com.discord                      │    2220 │
	│ us.zoom.videomeetings            │     425 │
	│ com.google.android.apps.meetings │     379 │
	│ com.microsoft.teams              │     321 │
	│ com.gotomeeting                  │     297 │
	│ com.facebook.orca                │     280 │
	│ com.cisco.webex.meetings         │     259 │
	│ com.Slack                        │     198 │
	│ com.skype.raider                 │     190 │
	├──────────────────────────────────┼─────────┤
	│ __total__                        │    4569 │
	└──────────────────────────────────┴─────────┘
	stats : number packets per-flow (min_pkts=1000)
	┏━━━━━━━┳━━━━━━━━━━━┓
	┃ stat  ┃     value ┃
	┡━━━━━━━╇━━━━━━━━━━━┩
	│ count │    4569.0 │
	│ mean  │  38321.32 │
	│ std   │   83282.0 │
	│ min   │    1001.0 │
	│ 25%   │    2863.0 │
	│ 50%   │    6303.0 │
	│ 75%   │   35392.0 │
	│ max   │ 1665842.0 │
	└───────┴───────────┘
	saving: ./envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/mirage22/preprocessed/imc23/mirage22_filtered_minpkts1000_splits.parquet
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

The console output is showing a few samples count reports
related to the processing performed on the datasets

1. The first report relates to the *unfiltered* dataset, i.e., 
the monolithic parquet files obtained consolidating all JSON
files but *before* applying any curation.
At first glance, it looks like this dataset has a lot of flows.
However, the following report shows the number of packets
per flow and suggests that there are many flows which are very short.

2. The second and third group of reports show similar information
to the first group but relates to the filtering
out of flows with less than 10 and 1,000 packets.

3. The last report shows the number of train/validation/test samples
by each application for the first split (the same counters are true for all
splits) when focusing on flows with more than 1,000 packets.
