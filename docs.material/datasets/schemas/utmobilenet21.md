# `utmobilenet21`

Below we report all schemas for all datasets.
The section expanded suggest the datasets to be used,
while ==highlighted rows== suggest which fields
are more useful for modeling.

??? note "tcbench datasets schema --name utmobilenet21 --type unfiltered"
	```
	┏━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
	┃ Field       ┃ Dtype    ┃ Description                                                                  ┃
	┡━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
	│ row_id      │ int      │ Unique flow id                                                               │
	│ src_ip      │ str      │ Source ip of the flow                                                        │
	│ src_port    │ int      │ Source port of the flow                                                      │
	│ dst_ip      │ str      │ Destination ip of the flow                                                   │
	│ dst_port    │ int      │ Destination port of the flow                                                 │
	│ ip_proto    │ int      │ Protocol of the flow (TCP or UDP)                                            │
	│ first       │ float    │ Timestamp of the first packet                                                │
	│ last        │ float    │ Timestamp of the last packet                                                 │
	│ duration    │ float    │ Duration of the flow                                                         │
	│ packets     │ int      │ Number of packets in the flow                                                │
	│ bytes       │ int      │ Number of bytes in the flow                                                  │
	│ partition   │ str      │ From which folder the flow was originally stored                             │
	│ location    │ str      │ Label originally provided by the dataset (see the related paper for details) │
	│ fname       │ str      │ Original filename where the packets of the flow come from                    │
	│ app         │ category │ Final label of the flow, encoded as pandas category                          │
	│ pkts_size   │ np.array │ Packet size time series                                                      │
	│ pkts_dir    │ np.array │ Packet diretion time series                                                  │
	│ timetofirst │ np.array │ Delta between the each packet timestamp the first packet of the flow         │
	└─────────────┴──────────┴──────────────────────────────────────────────────────────────────────────────┘
	```

!!! note "tcbench datasets schema --name utmobilenet21 --type filtered"
	```hl_lines="4 18 19 20 21"
	┏━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
	┃ Field       ┃ Dtype    ┃ Description                                                                  ┃
	┡━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
	│ row_id      │ int      │ Unique flow id                                                               │
	│ src_ip      │ str      │ Source ip of the flow                                                        │
	│ src_port    │ int      │ Source port of the flow                                                      │
	│ dst_ip      │ str      │ Destination ip of the flow                                                   │
	│ dst_port    │ int      │ Destination port of the flow                                                 │
	│ ip_proto    │ int      │ Protocol of the flow (TCP or UDP)                                            │
	│ first       │ float    │ Timestamp of the first packet                                                │
	│ last        │ float    │ Timestamp of the last packet                                                 │
	│ duration    │ float    │ Duration of the flow                                                         │
	│ packets     │ int      │ Number of packets in the flow                                                │
	│ bytes       │ int      │ Number of bytes in the flow                                                  │
	│ partition   │ str      │ From which folder the flow was originally stored                             │
	│ location    │ str      │ Label originally provided by the dataset (see the related paper for details) │
	│ fname       │ str      │ Original filename where the packets of the flow come from                    │
	│ app         │ category │ Final label of the flow, encoded as pandas category                          │
	│ pkts_size   │ np.array │ Packet size time series                                                      │
	│ pkts_dir    │ np.array │ Packet diretion time series                                                  │
	│ timetofirst │ np.array │ Delta between the each packet timestamp the first packet of the flow         │
	└─────────────┴──────────┴──────────────────────────────────────────────────────────────────────────────┘
	```

!!! note "tcbench datasets schema --name utmobilenet21 --type splits"
	```
	┏━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
	┃ Field         ┃ Dtype    ┃ Description                  ┃
	┡━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
	│ train_indexes │ np.array │ row_id of training samples   │
	│ val_indexes   │ np.array │ row_id of validation samples │
	│ test_indexes  │ np.array │ row_id of test samples       │
	│ split_index   │ int      │ Split id                     │
	└───────────────┴──────────┴──────────────────────────────┘
	```

