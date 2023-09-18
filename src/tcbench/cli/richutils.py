from __future__ import annotations

import pandas as pd

from rich.table import Table
from rich.panel import Panel
from rich import box
from typing import List

from tcbench import cli

console = cli.console


def _rich_table_from_series(ser:pd.Series, columns:List[str], with_total:bool=False) -> rich.table.Table:
    """Compose a rich Table from a pandas Series"""
    table = Table()
    table.add_column(columns[0])
    table.add_column(columns[1], justify="right")
    for index, value in zip(ser.index, ser.values):
        table.add_row(str(index), str(value))
    if with_total:
        table.add_section()
        table.add_row("__total__", str(ser.values.sum()))
    return table


def _rich_table_from_series_multiindex(ser:pd.Series, with_total:bool=False) -> rich.table.Table:
    """Compose a rich Table from a pandas Series with MultiIndex"""
    table = Table()
    for col in ser.index.names:
        table.add_column(col)
    table.add_column("samples", justify="right")

    for level0 in ser.index.levels[0]:
        tmp = ser.loc[level0]
        for level1, value in zip(tmp.index, tmp.values):
            table.add_row(str(level0), str(level1), str(value))
            level0 = ""
        if with_total:
            table.add_row("", "__total__", str(tmp.values.sum()))
        table.add_section()
    return table


def _rich_table_from_dataframe(df:pd.DataFrame, with_total:bool=False) -> rich.table.Table:
    """Compose a rich Table from a pandas DataFrame"""
    table = Table()
    table.add_column(df.index.name)
    for col in df.columns:
        table.add_column(col, justify="right")
    for idx in range(df.shape[0]):
        table.add_row(df.index[idx], *list(df.iloc[idx].values.astype(str)))
    if with_total:
        table.add_section()
        table.add_row("__total__", *list(map(str, df.sum().values)))
    return table


def rich_samples_count_report(obj:pd.Series | pd.DataFrame, columns:List[str]=None, title:str=None, with_total:bool=True) -> rich.table.Table:
    """Compute and format into a table the per-class samples count"""
    if columns is None:
        columns = ["app", "samples"]
    if isinstance(obj, pd.Series):
        if isinstance(obj.index, pd.MultiIndex):
            table = _rich_table_from_series_multiindex(obj, with_total=with_total)
        else:
            table = _rich_table_from_series(obj, columns, with_total=with_total)
    else:
        table = _rich_table_from_dataframe(obj, with_total=with_total)

    if title is not None:
        console.print(title)
    console.print(table)


def rich_splits_report(df:pd.DataFrame, df_split:pd.DataFrame, split_index:int=None, title:str=None, with_total:bool=True) -> rich.table.Table:
    min_split_idx = 0
    max_split_idx = df_split["split_index"].max()
    _title = f"split_index={min_split_idx} to {max_split_idx}"
    if split_index is not None:
        min_split_idx = int(split_index)
        _title = f"split_index={split_index}"

    df = df.copy()
    df = df.set_index("row_id", drop=False)
    ser_split = df_split[df_split["split_index"] == min_split_idx]
    df_tmp = pd.DataFrame(
        [
            df.loc[ser_split["train_indexes"].values[0]]["app"].value_counts(),
            df.loc[ser_split["val_indexes"].values[0]]["app"].value_counts(),
            df.loc[ser_split["test_indexes"].values[0]]["app"].value_counts(),
        ],
        index=["train_samples", "val_samples", "test_samples"],
    ).T
    df_tmp = df_tmp.assign(all_samples=df_tmp.sum(axis=1))
    if title:
        _title = title
    rich_samples_count_report(df_tmp, title=title, with_total=with_total)


def rich_packets_report(df:pd.DataFrame, packets_colname:str="packets", title:str=None) -> rich.table.Table:
    """Compute and reports stats for number of packets per flow"""
    ser = df[packets_colname].describe().round(2)
    rich_samples_count_report(
        ser, columns=["stat", "value"], title=title, with_total=False
    )


def rich_label(text:str, extra_new_line:bool=False) -> None:
    """Output on the console a formatted label"""
    if extra_new_line:
        console.print()
    console.print(Panel(text, box=box.ROUNDED, expand=False, padding=0))
