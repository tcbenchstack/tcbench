# `ucdavis-icdm19`

Below we report all schemas for all datasets.
The section expanded suggest the datasets to be used,
while ==highlighted rows== suggest which fields
are more useful for modeling.

For `ucdavis-icdm19` the three types (unfiltered, filtered, splits)
have the same schema because the splits are materialized.

```
tcbench datasets schema --name ucdavis-icdm19 --type unfiltered
```

!!! note "Output"
	```hl_lines="5 12 13 14"
	┏━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
	┃ Field       ┃ Dtype    ┃ Description                                         ┃
	┡━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
	│ row_id      │ int      │ Unique row id                                       │
	│ app         │ category │ Label of the flow                                   │
	│ flow_id     │ str      │ Original filename                                   │
	│ partition   │ str      │ Partition related to the flow                       │
	│ num_pkts    │ int      │ Number of packets in the flow                       │
	│ duration    │ float    │ Duration of the flow                                │
	│ bytes       │ int      │ Number of bytes of the flow                         │
	│ unixtime    │ str      │ Absolute time of each packet                        │
	│ timetofirst │ np.array │ Delta between a packet the first packet of the flow │
	│ pkts_size   │ np.array │ Packet size time series                             │
	│ pkts_dir    │ np.array │ Packet direction time series                        │
	│ pkts_iat    │ np.array │ Packet inter-arrival time series                    │
	└─────────────┴──────────┴─────────────────────────────────────────────────────┘
	```

??? note "tcbench datasets schema --name ucdavis-icdm19 --type filtered"
	```hl_lines="5 12 13 14"
	┏━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
	┃ Field       ┃ Dtype    ┃ Description                                         ┃
	┡━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
	│ row_id      │ int      │ Unique row id                                       │
	│ app         │ category │ Label of the flow                                   │
	│ flow_id     │ str      │ Original filename                                   │
	│ partition   │ str      │ Partition related to the flow                       │
	│ num_pkts    │ int      │ Number of packets in the flow                       │
	│ duration    │ float    │ Duration of the flow                                │
	│ bytes       │ int      │ Number of bytes of the flow                         │
	│ unixtime    │ str      │ Absolute time of each packet                        │
	│ timetofirst │ np.array │ Delta between a packet the first packet of the flow │
	│ pkts_size   │ np.array │ Packet size time series                             │
	│ pkts_dir    │ np.array │ Packet direction time series                        │
	│ pkts_iat    │ np.array │ Packet inter-arrival time series                    │
	└─────────────┴──────────┴─────────────────────────────────────────────────────┘
	```

??? note "tcbench datasets schema --name ucdavis-icdm19 --type splits"
	```hl_lines="5 12 13 14"
	┏━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
	┃ Field       ┃ Dtype    ┃ Description                                         ┃
	┡━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
	│ row_id      │ int      │ Unique row id                                       │
	│ app         │ category │ Label of the flow                                   │
	│ flow_id     │ str      │ Original filename                                   │
	│ partition   │ str      │ Partition related to the flow                       │
	│ num_pkts    │ int      │ Number of packets in the flow                       │
	│ duration    │ float    │ Duration of the flow                                │
	│ bytes       │ int      │ Number of bytes of the flow                         │
	│ unixtime    │ str      │ Absolute time of each packet                        │
	│ timetofirst │ np.array │ Delta between a packet the first packet of the flow │
	│ pkts_size   │ np.array │ Packet size time series                             │
	│ pkts_dir    │ np.array │ Packet direction time series                        │
	│ pkts_iat    │ np.array │ Packet inter-arrival time series                    │
	└─────────────┴──────────┴─────────────────────────────────────────────────────┘
	```

