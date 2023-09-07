## Datasets references

We refer to the following datasets

{{ read_csv('./datasets.csv') }}

!!! warning
    Run the following commands from the root folder
    of the code repository (one of the scripts, namely
    mirage22_json_to_parquet.py has an harded dependency
    from mirage19_json_to_parquet.py).


## Why and how to preprocess the raw data from each dataset

Each dataset comes as either CSV or JSON files, with a mixed
preference between per-packet and per-flow formating. Moreover,
files can be organized in subfolders---namely *partitions*---
to reflect some aspect of the measuring campaign.

We preprocess all dataset to create monolitich per-flow
parquet files, associating to each flow numpy arrays
for the packets time series used for modeling.

The scripts for the conversion are collected in the `/datasets` 
subfolder of the repository.
The same folder is expected to gather the output parquet files
and (later) the splits used for modeling.

!!! note
    Our modeling framework provides some flexibility to bypass this
    limitation but, as of now, this is not fully supported yet.

!!! note
    The code for generating the charts is in `/notebooks/datasets_properties.ipynb`

### ucdavis-icdm19

The dataset comprises 3 partitions with the following structure

```
<root folder>
├── pretraining
│   ├── Google Doc
│   ├── Google Drive
│   ├── Google Music
│   ├── Google Search
│   └── Youtube
├── Retraining(human-triggered)
│   ├── Google Doc
│   ├── Google Drive
│   ├── Google Music
│   ├── Google Search
│   └── Youtube
└── Retraining(script-triggered)
    ├── Google Doc
    ├── Google Drive
    ├── Google Music
    ├── Google Search
    └── Youtube
```

Inside each nested folder there is a collection CSV files.
Each file corresponds to a different flow, 
where each row represent individual packet information.

The aim of the pre-processing is to load all CSV into a
single parquet file `ucdavis-icdm19.parquet`, and "transpose" the representation---
rather than having indivial packet for each row, we
create one row per flow (the flow_id is the filename itself)
encoding the packet time series into numpy arrays.

The final dataset has the following columns

* `row_id`: a unique row id
* `app`: the label of the flow, encoded as pandas `category`
* `flow_id`: the original filename
* `partition`: the partition related to the flow
* `num_pkts`: number of packets in the flow
* `duration`: the duration of the flow
* `bytes`: the number of bytes of the flow
* `unixtime`: numpy array with the absolute time of each packet
* `timetofirst`: numpy array with the delta between a packet the first packet of the flow
* `pkts_size`: numpy array for the packet size time series
* `pkts_dir`: numpy array for the packet direction time series
* `pkts_iat`: numpy array for the packet inter-arrival time series

---

```
python datasets/ucdavis-icdm19_csv-to-parquet.py \
    --input-folder <where-you-unpacked-the-csvs> \
    --output-folder datasets/ucdavis-icdm19
```

???+ note "output"
    ```
    found 6672 files to load

    ................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................
    loading complete
    saving datasets/ucdavis-icdm19/ucdavis-icdm19.parquet
    ```

Here some aggregate stats about traffic volumes per-class (click to magnify the charts)

[![dataset_properties_ucdavis-icdm19]][dataset_properties_ucdavis-icdm19]

  [dataset_properties_ucdavis-icdm19]: figs/dataset_properties_ucdavis-icdm19.png



### mirage19

The dataset is a collection of JSON files for 20 applications

!!! note
    Despite the website and the related paper mention that
    the dataset contains 40 application, the public version
    has only 20. With separate communication with the authors
    of the dataset, we understood that the remaining 20
    are available only upon request (altough not explicitly
    specified). As result, we considered only the 20 publicly
    available.

The files collection is organized as follows
```
/mnt/storage/nfs/TLS/mirage-unina/mirage-19
└─ MIRAGE-2019_traffic_dataset_downloadable
   ├── Mi5_38_a4_ed_18_cc_bf
   └── Nexus7_bc_ee_7b_a4_09_47
```

Inside each nested folder there is a collection of JSON files
with some semantical information embedded in the names themselves.
    
Each JSON has fairly complicated nested structure
which makes it very difficult to work with.

The purpose of the preprocessing is to

1. Combine all JSON into a monolithic parquet file `mirage19.parquet`
2. Flatten the nested structure. For instance, a dictionary
    such as {"a":{"b":1, "c":2}} is transformed into two separate
    columns "a_b" and "a_c" with the respective values
3. Add a `"background"` class by processing the original
    label compared the JSON filenames. More specifically,
    each JSON file detail the name of the app used during
    a measurement campaign. But the traffic in an experiment
    can be related to a different app/service running in parallel.
    The decoupling of the two is facilitated by the column
    `flow_metadata_bf_label` which is collected using `netstat`
    from the mobile phone: if `flow_metadata_bf_label` != 
    the expected app name, we mark the flow as `background`
4. The dataset contains raw packet bytes across multiple packets
    of a flow. We process these series to search for ASCII strings.
    This can be usefull for extract (in a lazy way) TLS
    handshake information (e.g., SNI or certificate info)

The final parquet files has 127 columns, and most of
them comes from the original dataset itself.
They are not documented but fairly easy to understand
based on the name.

The most important one are

* `packet_data_packet_dir`: the time series of the packet direction
* `packet_data_l4_payload_bytes`: the time series of the packet size
* `packet_data_iat`: the time series of the packet inter-arrival time
* `flow_metadata_bf_label`: the label gathered via netstat
* `strings`: the ASCII string recovered from the payload analysis
* `android_name`: the app used for an experiment
* `app`: the final label encoded as a pandas `category`

---

```
python datasets/mirage19_json_to_parquet.py \
    --input-folder <where-you-unpacked-the-json>/MIRAGE-2019_traffic_dataset_downloadable \
    --output-f0lder datasets/mirage19 \
    --workers 30
```

???+ note "output"
    ```
    found 1642 files
    ..........................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................done
    saving: mirage19/mirage19.parquet
    ```

Here some aggregate stats about traffic volumes per-class (click to magnify the charts)

[![dataset_properties_mirage19]][dataset_properties_mirage19]

  [dataset_properties_mirage19]: figs/dataset_properties_mirage19.png

### mirage22

The data has the same format as mirage-19, i.e., a collection of JSON files.

The dataset contains the following structure

```
/mnt/storage/nfs/TLS/mirage-unina/mirage-22
└── MIRAGE-COVID-CCMA-2022
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

!!! warning:
    We disegarded the pickle preprocessed version because (from what we reverse engineered) 
    encodes a series of object in the same pickle, but we found it cumbersome to work with.


Please refer to mirage-19 for details about pre-processing


---

```
python datasets/mirage22_json_to_parquet.py \
    --input-folder <where-you-unpacked-the-json>/MIRAGE-COVID-CCMA-2022/Raw_JSON/ \
    --output-f0lder datasets/mirage22
    --workers 30
```

???+ note "output"
    ```
    found 998 files
    ......................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................done
    saving: datasets/mirage22/mirage22.parquet
    ```

Here some aggregate stats about traffic volumes per-class (click to magnify the charts)

[![dataset_properties_mirage22]][dataset_properties_mirage22]

  [dataset_properties_mirage22]: figs/dataset_properties_mirage22.png

### utmobilenet21

The dataset is a collection of per-packet CSV files divided into 4 partitions
representing 4 different measurement campaigns.

The files are organized as follows
```
<root>
└── csvs
    ├── Action-Specific Wild Test Data
    ├── Deterministic Automated Data
    ├── Randomized Automated Data
    └── Wild Test Data
```

Broadly speaking, the dataset has the same preprocessing needs
of ucdavid-icmd19, i.e., being formatted per-packet, we
pre-process it into per-flow and create numpy time series.

The final `utmobilenet21.parquet` files contains the following columns

* `row_id`: a unique flow id
* `src_ip`: the source ip of the flow
* `src_port`: the source port of the flow
* `dst_ip`: the destination ip of the flow
* `dst_port`: the destination port of the flow
* `ip_proto`: the protocol of the flow (TCP or UDP)
* `first`: timestamp of the first packet
* `last`: timestamp of the last packet
* `duration`: duration of the flow
* `packets`: number of packets in the flow
* `bytes`: number of bytes in the flow
* `partition`: from which folder the flow was originally stored
* `location`: a label originally provided by the dataset (see the related paper for details)
* `fname`: the original filename where the packets of the flow come from 
* `app`: the final label of the flow, encoded as pandas `category`
* `pkts_size`: the numpy array for the packet size time series
* `pkts_dir`: the numpy array for the packet diretion time series
* `timetofirst`: the numpy array for the delta between the each packet timestamp the first packet of the flow


---

```
python datasets/utmobilenet21_csv_to_parquet.py \
    --input-folder <where-you-unpacked-the-dataset>/csvs \
    --output-folder ./datasets/utmobilenet21 \
    --tmp-staging-folder /tmp/processing-utmobilenet21 \
    --num-workers 10
```

???+ note "output"
    ```
    processing: /mnt/storage/nfs/TLS/utmobilenet-21/csvs/Wild Test Data
    found 14 files
    ..............
    stage1 completed
    stage2 completed
    stage3 completed
    stage4 completed
    saving: /tmp/processing-utmobilenet21/wild_test_data.parquet

    processing: /mnt/storage/nfs/TLS/utmobilenet-21/csvs/Deterministic Automated Data
    found 3438 files
    ..............................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................
    stage1 completed
    stage2 completed
    stage3 completed
    stage4 completed
    saving: /tmp/processing-utmobilenet21/deterministic_automated_data.parquet


    processing: /mnt/storage/nfs/TLS/utmobilenet-21/csvs/Action-Specific Wild Test Data
    found 43 files
    ...........................................
    stage1 completed
    stage2 completed
    stage3 completed
    stage4 completed
    saving: /tmp/processing-utmobilenet21/action-specific_wild_test_data.parquet

    processing: /mnt/storage/nfs/TLS/utmobilenet-21/csvs/Randomized Automated Data
    found 288 files
    ................................................................................................................................................................................................................................................................................................
    stage1 completed
    stage2 completed
    stage3 completed
    stage4 completed
    saving: /tmp/processing-utmobilenet21/randomized_automated_data.parquet
    merging all partitions
    saving: datasets/utmobilenet21/utmobilenet21.parquet
    ```

Here some aggregate stats about traffic volumes per-class (click to magnify the charts)

[![dataset_properties_utmobilenet21]][dataset_properties_utmobilenet21]

  [dataset_properties_utmobilenet21]: figs/dataset_properties_utmobilenet21.png
