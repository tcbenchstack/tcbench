# `mirage22`

Below we report all schemas for all datasets.
The section expanded suggest the datasets to be used,
while ==highlighted rows== suggest which fields
are more useful for modeling.

??? note "tcbench datasets schema --name mirage22 --type unfiltered"
	```
	┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
	┃ Field                                                     ┃ Dtype    ┃ Description                                                ┃
	┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
	│ row_id                                                    │ int      │ Unique flow id                                             │
	│ conn_id                                                   │ str      │ Flow 5-tuple                                               │
	│ packet_data_timestamp                                     │ np.array │ Time series of packet unixtime                             │
	│ packet_data_src_port                                      │ np.array │ Time series of the source ports                            │
	│ packet_data_dst_port                                      │ np.array │ Time series of the destination ports                       │
	│ packet_data_packet_dir                                    │ np.array │ Time series of pkts direction (0 or 1)                     │
	│ packet_data_ip_packet_bytes                               │ np.array │ Time series pkts bytes (as from IP len field)              │
	│ packet_data_ip_header_bytes                               │ np.array │ Time series of IP header bytes                             │
	│ packet_data_l4_payload_bytes                              │ np.array │ Time series of payload pkts size                           │
	│ packet_data_l4_header_bytes                               │ np.array │ Time series of L4 header bytes                             │
	│ packet_data_iat                                           │ np.array │ Time series of pkts inter arrival times                    │
	│ packet_data_tcp_win_size                                  │ np.array │ Time series of TCP window size                             │
	│ packet_data_tcp_flags                                     │ np.array │ Time series of TCP flags                                   │
	│ packet_data_l4_raw_payload                                │ np.array │ List of list with each packet payload                      │
	│ packet_data_is_clear                                      │ np.array │ n.a.                                                       │
	│ packet_data_heuristic                                     │ str      │ n.a.                                                       │
	│ packet_data_annotations                                   │ str      │ n.a.                                                       │
	│ flow_features_packet_length_biflow_min                    │ float    │ Bidirectional min frame (i.e., pkt with headers) size      │
	│ flow_features_packet_length_biflow_max                    │ float    │ Bidirectional max frame size                               │
	│ flow_features_packet_length_biflow_mean                   │ float    │ Bidirectional mean frame size                              │
	│ flow_features_packet_length_biflow_std                    │ float    │ Bidirectional std frame size                               │
	│ flow_features_packet_length_biflow_var                    │ float    │ Bidirectional variance frame size                          │
	│ flow_features_packet_length_biflow_mad                    │ float    │ Bidirectional median absolute deviation frame size         │
	│ flow_features_packet_length_biflow_skew                   │ float    │ Bidirection skew frame size                                │
	│ flow_features_packet_length_biflow_kurtosis               │ float    │ Bidirectional kurtosi frame size                           │
	│ flow_features_packet_length_biflow_10_percentile          │ float    │ Bidirection 10%-ile of frame size                          │
	│ flow_features_packet_length_biflow_20_percentile          │ float    │ Bidirection 20%-ile of frame size                          │
	│ flow_features_packet_length_biflow_30_percentile          │ float    │ Bidirection 30%-ile of frame size                          │
	│ flow_features_packet_length_biflow_40_percentile          │ float    │ Bidirection 40%-ile of frame size                          │
	│ flow_features_packet_length_biflow_50_percentile          │ float    │ Bidirection 50%-ile of frame size                          │
	│ flow_features_packet_length_biflow_60_percentile          │ float    │ Bidirection 60%-le of frame size                           │
	│ flow_features_packet_length_biflow_70_percentile          │ float    │ Bidirection 70%-ile of frame size                          │
	│ flow_features_packet_length_biflow_80_percentile          │ float    │ Bidirection 80%-ile of frame size                          │
	│ flow_features_packet_length_biflow_90_percentile          │ float    │ Bidirection 90%-ile of frame size                          │
	│ flow_features_packet_length_upstream_flow_min             │ float    │ Upstream min frame (i.e., pkt with headers) size           │
	│ flow_features_packet_length_upstream_flow_max             │ float    │ Upstream max frame size                                    │
	│ flow_features_packet_length_upstream_flow_mean            │ float    │ Upstream mean frame size                                   │
	│ flow_features_packet_length_upstream_flow_std             │ float    │ Upstream std frame size                                    │
	│ flow_features_packet_length_upstream_flow_var             │ float    │ Upstream variance frame size                               │
	│ flow_features_packet_length_upstream_flow_mad             │ float    │ Upstream median absolute deviation frame size              │
	│ flow_features_packet_length_upstream_flow_skew            │ float    │ Upstream skew frame size                                   │
	│ flow_features_packet_length_upstream_flow_kurtosis        │ float    │ Upstream kurtosi frame size                                │
	│ flow_features_packet_length_upstream_flow_10_percentile   │ float    │ Upstream 10%-ile frame size                                │
	│ flow_features_packet_length_upstream_flow_20_percentile   │ float    │ Upstream 20%-ile frame size                                │
	│ flow_features_packet_length_upstream_flow_30_percentile   │ float    │ Upstream 30%-ile frame size                                │
	│ flow_features_packet_length_upstream_flow_40_percentile   │ float    │ Upstream 40%-ile frame size                                │
	│ flow_features_packet_length_upstream_flow_50_percentile   │ float    │ Upstream 50%-ile frame size                                │
	│ flow_features_packet_length_upstream_flow_60_percentile   │ float    │ Upstream 60%-ile frame size                                │
	│ flow_features_packet_length_upstream_flow_70_percentile   │ float    │ Upstream 70%-ile frame size                                │
	│ flow_features_packet_length_upstream_flow_80_percentile   │ float    │ Upstream 80%-ile frame size                                │
	│ flow_features_packet_length_upstream_flow_90_percentile   │ float    │ Upstream 90%-ile frame size                                │
	│ flow_features_packet_length_downstream_flow_min           │ float    │ Downstream min frame (i.e., pkt with headers) size         │
	│ flow_features_packet_length_downstream_flow_max           │ float    │ Downstream max frame size                                  │
	│ flow_features_packet_length_downstream_flow_mean          │ float    │ Downstream mean frame size                                 │
	│ flow_features_packet_length_downstream_flow_std           │ float    │ Downstream std frame size                                  │
	│ flow_features_packet_length_downstream_flow_var           │ float    │ Downstream variance frame size                             │
	│ flow_features_packet_length_downstream_flow_mad           │ float    │ Downstream max frame size                                  │
	│ flow_features_packet_length_downstream_flow_skew          │ float    │ Downstream skew frame size                                 │
	│ flow_features_packet_length_downstream_flow_kurtosis      │ float    │ Downstream kurtosi frame size                              │
	│ flow_features_packet_length_downstream_flow_10_percentile │ float    │ Downstream 10%-ile frame size                              │
	│ flow_features_packet_length_downstream_flow_20_percentile │ float    │ Downstream 20%-ile frame size                              │
	│ flow_features_packet_length_downstream_flow_30_percentile │ float    │ Downstream 30%-ile frame size                              │
	│ flow_features_packet_length_downstream_flow_40_percentile │ float    │ Downstream 40%-ile frame size                              │
	│ flow_features_packet_length_downstream_flow_50_percentile │ float    │ Downstream 50%-ile frame size                              │
	│ flow_features_packet_length_downstream_flow_60_percentile │ float    │ Downstream 60%-ile frame size                              │
	│ flow_features_packet_length_downstream_flow_70_percentile │ float    │ Downstream 70%-ile frame size                              │
	│ flow_features_packet_length_downstream_flow_80_percentile │ float    │ Downstream 80%-ile frame size                              │
	│ flow_features_packet_length_downstream_flow_90_percentile │ float    │ Downstream 90%-ile frame size                              │
	│ flow_features_iat_biflow_min                              │ float    │ Bidirectional min inter arrival time                       │
	│ flow_features_iat_biflow_max                              │ float    │ Bidirectional max inter arrival time                       │
	│ flow_features_iat_biflow_mean                             │ float    │ Bidirectional mean inter arrival time                      │
	│ flow_features_iat_biflow_std                              │ float    │ Bidirectional std inter arrival time                       │
	│ flow_features_iat_biflow_var                              │ float    │ Bidirectional variance inter arrival time                  │
	│ flow_features_iat_biflow_mad                              │ float    │ Bidirectional median absolute deviation inter arrival time │
	│ flow_features_iat_biflow_skew                             │ float    │ Bidirectional skew inter arrival time                      │
	│ flow_features_iat_biflow_kurtosis                         │ float    │ Bidirectional kurtosi inter arrival time                   │
	│ flow_features_iat_biflow_10_percentile                    │ float    │ Bidirectional 10%-tile inter arrival time                  │
	│ flow_features_iat_biflow_20_percentile                    │ float    │ Bidirectional 20%-tile inter arrival time                  │
	│ flow_features_iat_biflow_30_percentile                    │ float    │ Bidirectional 30%-tile inter arrival time                  │
	│ flow_features_iat_biflow_40_percentile                    │ float    │ Bidirectional 40%-tile inter arrival time                  │
	│ flow_features_iat_biflow_50_percentile                    │ float    │ Bidirectional 50%-tile inter arrival time                  │
	│ flow_features_iat_biflow_60_percentile                    │ float    │ Bidirectional 60%-tile inter arrival time                  │
	│ flow_features_iat_biflow_70_percentile                    │ float    │ Bidirectional 70%-tile inter arrival time                  │
	│ flow_features_iat_biflow_80_percentile                    │ float    │ Bidirectional 80%-tile inter arrival time                  │
	│ flow_features_iat_biflow_90_percentile                    │ float    │ Bidirectional 90%-tile inter arrival time                  │
	│ flow_features_iat_upstream_flow_min                       │ float    │ Upstream min inter arrival time                            │
	│ flow_features_iat_upstream_flow_max                       │ float    │ Upstream max inter arrival time                            │
	│ flow_features_iat_upstream_flow_mean                      │ float    │ Upstream avg inter arrival time                            │
	│ flow_features_iat_upstream_flow_std                       │ float    │ Upstream std inter arrival time                            │
	│ flow_features_iat_upstream_flow_var                       │ float    │ Upstream variance inter arrival time                       │
	│ flow_features_iat_upstream_flow_mad                       │ float    │ Upstream median absolute deviation inter arrival time      │
	│ flow_features_iat_upstream_flow_skew                      │ float    │ Upstream skew inter arrival time                           │
	│ flow_features_iat_upstream_flow_kurtosis                  │ float    │ Upstream kurtosi inter arrival time                        │
	│ flow_features_iat_upstream_flow_10_percentile             │ float    │ Upstream 10%-ile inter arrival time                        │
	│ flow_features_iat_upstream_flow_20_percentile             │ float    │ Upstream 20%-ile inter arrival time                        │
	│ flow_features_iat_upstream_flow_30_percentile             │ float    │ Upstream 30%-ile inter arrival time                        │
	│ flow_features_iat_upstream_flow_40_percentile             │ float    │ Upstream 40%-ile inter arrival time                        │
	│ flow_features_iat_upstream_flow_50_percentile             │ float    │ Upstream 50%-ile inter arrival time                        │
	│ flow_features_iat_upstream_flow_60_percentile             │ float    │ Upstream 60%-ile inter arrival time                        │
	│ flow_features_iat_upstream_flow_70_percentile             │ float    │ Upstream 70%-ile inter arrival time                        │
	│ flow_features_iat_upstream_flow_80_percentile             │ float    │ Upstream 80%-ile inter arrival time                        │
	│ flow_features_iat_upstream_flow_90_percentile             │ float    │ Upstream 90%-ile inter arrival time                        │
	│ flow_features_iat_downstream_flow_min                     │ float    │ Downstream min inter arrival time                          │
	│ flow_features_iat_downstream_flow_max                     │ float    │ Downstream max inter arrival time                          │
	│ flow_features_iat_downstream_flow_mean                    │ float    │ Downstream mean inter arrival time                         │
	│ flow_features_iat_downstream_flow_std                     │ float    │ Downstream std inter arrival time                          │
	│ flow_features_iat_downstream_flow_var                     │ float    │ Downstream variance inter arrival time                     │
	│ flow_features_iat_downstream_flow_mad                     │ float    │ Downstream median absolute deviation inter arrival time    │
	│ flow_features_iat_downstream_flow_skew                    │ float    │ Downstream skew inter arrival time                         │
	│ flow_features_iat_downstream_flow_kurtosis                │ float    │ Downstream kurtosi inter arrival time                      │
	│ flow_features_iat_downstream_flow_10_percentile           │ float    │ Downstream 10%-ile inter arrival time                      │
	│ flow_features_iat_downstream_flow_20_percentile           │ float    │ Downstream 20%-ile inter arrival time                      │
	│ flow_features_iat_downstream_flow_30_percentile           │ float    │ Downstream 30%-ile inter arrival time                      │
	│ flow_features_iat_downstream_flow_40_percentile           │ float    │ Downstream 40%-ile inter arrival time                      │
	│ flow_features_iat_downstream_flow_50_percentile           │ float    │ Downstream 50%-ile inter arrival time                      │
	│ flow_features_iat_downstream_flow_60_percentile           │ float    │ Downstream 60%-ile inter arrival time                      │
	│ flow_features_iat_downstream_flow_70_percentile           │ float    │ Downstream 70%-ile inter arrival time                      │
	│ flow_features_iat_downstream_flow_80_percentile           │ float    │ Downstream 80%-ile inter arrival time                      │
	│ flow_features_iat_downstream_flow_90_percentile           │ float    │ Downstream 90%-ile inter arrival time                      │
	│ flow_metadata_bf_device                                   │ str      │ Ethernet address                                           │
	│ flow_metadata_bf_label_source                             │ str      │ Constant value 'netstate'                                  │
	│ flow_metadata_bf_label                                    │ str      │ original mirage label                                      │
	│ flow_metadata_bf_sublabel                                 │ str      │ n.a.                                                       │
	│ flow_metadata_bf_label_version_code                       │ str      │ n.a.                                                       │
	│ flow_metadata_bf_label_version_name                       │ str      │ n.a.                                                       │
	│ flow_metadata_bf_labeling_type                            │ str      │ exact=via netstat; most-common=via experiment              │
	│ flow_metadata_bf_num_packets                              │ int      │ Bidirectional number of pkts                               │
	│ flow_metadata_bf_ip_packet_bytes                          │ int      │ Bidirectional bytes (including headers)                    │
	│ flow_metadata_bf_l4_payload_bytes                         │ int      │ Bidirectional payload bytes                                │
	│ flow_metadata_bf_duration                                 │ float    │ Bidirectional duration                                     │
	│ flow_metadata_uf_num_packets                              │ int      │ Upload number of pkts                                      │
	│ flow_metadata_uf_ip_packet_bytes                          │ int      │ Upload bytes (including headers)                           │
	│ flow_metadata_uf_l4_payload_bytes                         │ int      │ Upload payload bytes                                       │
	│ flow_metadata_uf_duration                                 │ float    │ Upload duration                                            │
	│ flow_metadata_uf_mss                                      │ float    │ Upload maximum segment size                                │
	│ flow_metadata_uf_ws                                       │ float    │ Upload window scaling                                      │
	│ flow_metadata_df_num_packets                              │ int      │ Download number of packets                                 │
	│ flow_metadata_df_ip_packet_bytes                          │ int      │ Download bytes (including headers)                         │
	│ flow_metadata_df_l4_payload_bytes                         │ int      │ Download payload bytes                                     │
	│ flow_metadata_df_duration                                 │ float    │ Download duration                                          │
	│ flow_metadata_df_mss                                      │ float    │ Download maximum segment size                              │
	│ flow_metadata_df_ws                                       │ float    │ Download window scaling                                    │
	│ flow_metadata_bf_activity                                 │ str      │ Experiment activity                                        │
	│ strings                                                   │ list     │ ASCII string extracted from payload                        │
	│ android_name                                              │ str      │ app name (based on filename)                               │
	│ device_name                                               │ str      │ device name (based on filename)                            │
	│ app                                                       │ category │ label (background|android app)                             │
	│ src_ip                                                    │ str      │ Source IP                                                  │
	│ src_port                                                  │ str      │ Source port                                                │
	│ dst_ip                                                    │ str      │ Destination IP                                             │
	│ dst_port                                                  │ str      │ Destination port                                           │
	│ proto                                                     │ str      │ L4 protol                                                  │
	│ packets                                                   │ int      │ Number of (bidirectional) packets                          │
	└───────────────────────────────────────────────────────────┴──────────┴────────────────────────────────────────────────────────────┘
	```

!!! note "tcbench datasets schema --name mirage22 --type filtered"
	```hl_lines="4 15 22 23 24"
	┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
	┃ Field                             ┃ Dtype    ┃ Description                                                          ┃
	┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
	│ row_id                            │ int      │ Unique flow id                                                       │
	│ conn_id                           │ str      │ Flow 5-tuple                                                         │
	│ packet_data_l4_raw_payload        │ np.array │ List of list with each packet payload                                │
	│ flow_metadata_bf_label            │ str      │ original mirage label                                                │
	│ flow_metadata_bf_activity         │ str      │ Experiment activity                                                  │
	│ flow_metadata_bf_labeling_type    │ str      │ exact=via netstat; most-common=via experiment                        │
	│ flow_metadata_bf_l4_payload_bytes │ int      │ Bidirectional payload bytes                                          │
	│ flow_metadata_bf_duration         │ float    │ Bidirectional duration                                               │
	│ strings                           │ list     │ ASCII string extracted from payload                                  │
	│ android_name                      │ str      │ app name (based on filename)                                         │
	│ device_name                       │ str      │ device name (based on filename)                                      │
	│ app                               │ category │ label (background|android app)                                       │
	│ src_ip                            │ str      │ Source IP                                                            │
	│ src_port                          │ str      │ Source port                                                          │
	│ dst_ip                            │ str      │ Destination IP                                                       │
	│ dst_port                          │ str      │ Destination port                                                     │
	│ proto                             │ str      │ L4 protocol                                                          │
	│ packets                           │ int      │ Number of (bidirectional) packets                                    │
	│ pkts_size                         │ str      │ Packet size time series                                              │
	│ pkts_dir                          │ str      │ Packet diretion time series                                          │
	│ timetofirst                       │ str      │ Delta between the each packet timestamp the first packet of the flow │
	└───────────────────────────────────┴──────────┴──────────────────────────────────────────────────────────────────────┘
	```

!!! note "tcbench datasets schema --name mirage22 --type splits"
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

