from __future__ import annotations

import ipaddress
import functools

import polars as pl
import numpy as np

from typing import Any

from tcbench.libtcdatasets.constants import (
    APP_LABEL_BACKGROUND,
    APP_LABEL_ALL,
)


def add_is_private_ip_columns(
    df: pl.DataFrame, src_ip_colname: str = "src_ip", dst_ip_colname: str = "dst_ip"
) -> pl.DataFrame:
    def is_private(ip_addr: str) -> bool:
        return ipaddress.ip_address(ip_addr).is_private

    df_ip_private = (
        pl.concat(
            (
                df[src_ip_colname].unique().rename("ip_addr"),
                df[src_ip_colname].unique().rename("ip_addr"),
            )
        )
        .to_frame()
        .with_columns(
            pl.col("ip_addr")
            .map_elements(is_private, return_dtype=pl.Boolean())
            .alias("is_private")
        )
    )

    return (
        df.join(
            df_ip_private,
            left_on=src_ip_colname,
            right_on="ip_addr",
            how="left",
        )
        .rename(
            {
                "is_private": f"{src_ip_colname}_is_private",
            }
        )
        .join(
            df_ip_private,
            left_on=dst_ip_colname,
            right_on="ip_addr",
            how="left",
        )
        .rename({"is_private": f"{dst_ip_colname}_is_private"})
    )


def add_is_valid_tcp_handshake_heuristic(
    df: pl.DataFrame,
    tcp_handshake_size: int = 40,
    direction_upload: int = 1,
    direction_download: int = -1,
) -> pl.DataFrame:
    def _is_valid_tcp_handshake(struct):
        return (struct["proto"] == "udp") | (
            (
                struct["pkts_size"]
                == [tcp_handshake_size, tcp_handshake_size, tcp_handshake_size]
            )
            & (
                struct["pkts_dir"]
                == [direction_upload, direction_download, direction_upload]
            )
        )

    return df.with_columns(
        pl.struct(
            pl.col("pkts_size").list.head(3),
            pl.col("pkts_dir").list.head(3),
            pl.col("proto"),
        )
        .map_elements(_is_valid_tcp_handshake, return_dtype=pl.Boolean())
        .alias("is_valid_handshake")
    )

def add_is_valid_tcp_handshake_from_flags(
    df: pl.DataFrame,
    colname_pkts_flags: str = "pkts_tcp_flags",
    colname_pkts_dir: str = "pkts_dir",
    colname_proto: str = "proto", 
    proto_udp: str = 17,
    direction_upload: int = 0,
    direction_download: int = 1, 
) -> pl.DataFrame:
    def _is_valid_tcp_handshake(struct):
        return (struct[colname_proto] == proto_udp) | (
            # outgoing SYN
            ((struct["flags_pkt1"] == "S") & (struct["dir_pkt1"] == 0)) 
            # incoming SYN+ACK 
            & ((struct["flags_pkt2"] == "SA") & (struct["dir_pkt2"] == 1))
            # outgoing ACK 
            & (
                ((struct["flags_pkt3"] == "A") | (struct["flags_pkt3"] == "PA"))
                & (struct["dir_pkt3"] == 0) 
            )
        )

    return df.with_columns(
        pl.struct(
            colname_proto,
            pl.col(colname_pkts_flags).list.get(0).alias("flags_pkt1"), 
            pl.col(colname_pkts_flags).list.get(1, null_on_oob=True).alias("flags_pkt2"), 
            pl.col(colname_pkts_flags).list.get(2, null_on_oob=True).alias("flags_pkt3"), 
            pl.col(colname_pkts_dir).list.get(0).alias("dir_pkt1"), 
            pl.col(colname_pkts_dir).list.get(1, null_on_oob=True).alias("dir_pkt2"), 
            pl.col(colname_pkts_dir).list.get(2, null_on_oob=True).alias("dir_pkt3"), 
        )
        .map_elements(_is_valid_tcp_handshake, return_dtype=pl.Boolean())
        .alias("is_valid_handshake")
      )


def get_stats(df: pl.DataFrame) -> pl.DataFrame:
    def _get_samples_counters():
        return pl.struct(
            samples=pl.col("row_id").count(),
            samples_tcp=(pl.col("proto") == "tcp").sum(),
            samples_udp=(pl.col("proto") == "udp").sum(),
            samples_is_valid_handshake=pl.col("is_valid_handshake").sum(),
        )

    def _get_stats(colname):
        return pl.struct(
            **{
                f"{colname}_min": pl.col(colname).min(),
                f"{colname}_max": pl.col(colname).max(),
                f"{colname}_q5": pl.col(colname).quantile(0.05),
                f"{colname}_q25": pl.col(colname).quantile(0.25),
                f"{colname}_q50": pl.col(colname).quantile(0.5),
                f"{colname}_q75": pl.col(colname).quantile(0.75),
                f"{colname}_q95": pl.col(colname).quantile(0.95),
            }
        )

    # stats by app
    df_stats_app = (
        df.group_by("app")
        .agg(
            _get_samples_counters().alias("samples_counters"),
            pl.col("pkts_size").list.min().alias("pkts_size_min"),
            pl.col("pkts_size").list.max().alias("pkts_size_max"),
            _get_stats("packets").alias("packets_stats"),
            _get_stats("duration").alias("duration_stats"),
        )
        .with_columns(
            pl.col("pkts_size_min").list.min(),
            pl.col("pkts_size_max").list.max(),
        )
        .unnest(
            "samples_counters",
            "packets_stats",
            "duration_stats",
        )
        .sort("samples", descending=True)
    )

    # stats all
    df_stats = (
        df.select(
            _get_samples_counters().alias("samples_counters"),
            pl.col("pkts_size").list.min().min().alias("pkts_size_min"),
            pl.col("pkts_size").list.max().max().alias("pkts_size_max"),
            _get_stats("packets").alias("packets_stats"),
            _get_stats("duration").alias("duration_stats"),
        )
        .unnest(
            "samples_counters",
            "packets_stats",
            "duration_stats",
        )
        .with_columns(pl.lit(APP_LABEL_ALL).alias("app"))
        .select(df_stats_app.columns)
    )

    return pl.concat(
        (
            df_stats_app.filter(pl.col("app") != APP_LABEL_BACKGROUND),
            df_stats,
            df_stats_app.filter(pl.col("app") == APP_LABEL_BACKGROUND),
        )
    )


def expr_pkts_size_times_dir() -> pl.Expr:
    return pl.struct(["pkts_size", "pkts_dir"]).map_elements(
        lambda data: pl.Series(np.multiply(data["pkts_size"], data["pkts_dir"])),
        return_dtype=pl.List(pl.Int64),
    )


def helper_list_index_equal_value(data: pl.Series, value: Any) -> pl.Series:
    return pl.Series(np.where(data == value)[0])


def helper_list_index_not_equal_value(data: pl.Series, value: Any) -> pl.Series:
    return pl.Series(np.where(data != value)[0])


def helper_list_index_value_greather_than(data: pl.Series, value: Any, inclusive: bool = False) -> pl.Series:
    if inclusive:
        return pl.Series(np.where(data >= value)[0])
    return pl.Series(np.where(data > value)[0])
    

def helper_list_index_value_lower_than(data: pl.Series, value: Any, inclusive: bool = False) -> pl.Series:
    if inclusive:
        return pl.Series(np.where(data <= value)[0])
    return pl.Series(np.where(data < value)[0])


def expr_pkts_ack_idx(colname: str = "pkts_size", ack_size: int = 40) -> pl.Expr:
    func = functools.partial(helper_list_index_equal_value, value=ack_size)
    return pl.col(colname).map_elements(
        function=func, return_dtype=pl.List(pl.Int64)
    )


def expr_pkts_data_idx(colname: str = "pkts_size", ack_size: int = 40) -> pl.Expr:
    func = functools.partial(helper_list_index_not_equal_value, value=ack_size)
    return pl.col(colname).map_elements(
        function=func, return_dtype=pl.List(pl.Int64)
    )

def expr_indices_list_value_equal_to(colname: str, value: Any) -> pl.Expr:
    func = functools.partial(helper_list_index_equal_value, value=value)
    return pl.col(colname).map_elements(
        function=func, return_dtype=pl.List(pl.Int64)
    )

def expr_indices_list_value_not_equal_to(colname: str, value: Any) -> pl.Expr:
    func = functools.partial(helper_list_index_not_equal_value, value=value)
    return pl.col(colname).map_elements(
        function=func, return_dtype=pl.List(pl.Int64)
    )

def expr_indices_list_value_greater_than(colname: str, value: Any, inclusive: bool = False) -> pl.Expr:
    func = functools.partial(helper_list_index_value_greather_than, value=value, inclusive=inclusive)
    return pl.col(colname).map_elements(
        function=func, return_dtype=pl.List(pl.Int64)
    )

def expr_indices_list_value_lower_than(colname: str, value: Any, inclusive: bool = False) -> pl.Expr:
    func = functools.partial(helper_list_index_value_lower_than, value=value, inclusive=inclusive)
    return pl.col(colname).map_elements(
        function=func, return_dtype=pl.List(pl.Int64)
    )


def expr_list_len_upload(list_colname: str, list_idx_colname: str) -> pl.Expr:
    return (
        pl.col(list_colname)
        .list.gather(pl.col(list_idx_colname))
        .list.eval(pl.element() > 0)
        .list.len()
    )


def expr_list_len_download(list_colname: str, list_idx_colname: str) -> pl.Expr:
    return (
        pl.col(list_colname)
        .list.gather(pl.col(list_idx_colname))
        .list.eval(pl.element() < 0)
        .list.len()
    )
