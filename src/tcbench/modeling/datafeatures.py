from __future__ import annotations

import polars as pl
import numpy as np

from typing import List,Dict


def packet_series_colnames(df:pl.DataFrame) -> List[str]:
    return [
        col
        for col in df.columns
        if col.startswith("pkts_") and df.schema.get(col).is_nested()
    ]


def expr_pad_packet_series(
    colname: str,
    expected_len: int,
    return_dtype: pl.DataType,
    pad_value: int = 0,
) -> pl.Expr:
    return (
        pl.when(
            pl.col(colname).list.len() < expected_len
        )
        .then(
            pl.col(colname).map_elements(
                function=lambda data: pl.Series(
                    np.pad(
                        data, 
                        pad_width=(0, expected_len-min(len(data), expected_len)), 
                        constant_values=pad_value,
                    )
                ),
                return_dtype=return_dtype
            )
        )
    )


def packet_series_pad(
    df: pl.DataFrame,
    expected_len: int,
    pad_value: int = 0,
) -> pl.DataFrame:
    cols = packet_series_colnames(df)
    return df.with_columns(
        **{
            col: expr_pad_packet_series(
                col,
                expected_len,
                df.schema.get(col),
                pad_value,
            )
            for col in cols
        },
        is_padded=(
            pl.col(cols[0]).list.len() < expected_len
        )
    )

def packet_series_cut(
    df: pl.DataFrame,
    expected_len: int | Dict[str, int],
) -> pl.DataFrame:
    cols = packet_series_colnames(df)
    return df.with_columns(
        **{
            col: pl.col(col).list.head(expected_len)
            for col in cols
        },
        is_cut=(
            pl.col(cols[0]).list.len() < expected_len
        )
    )
