from __future__ import annotations

import polars as pl

from typing import Any, Dict

import pickle
import yaml
import pathlib

import tcbench.cli

def _check_file_exists(path: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path)
    if not path.exists():
        raise RuntimeError(f"FileNotFound: {path}")
    return path

def create_folder(folder: pathlib.Path, echo: bool = True) -> pathlib.Path:
    folder = pathlib.Path(folder)
    if not folder.exists():
        tcbench.cli.logger.log(f"creating folder: {folder}", echo=echo)
        folder.mkdir(parents=True)
    return folder

def load_pickle(path: pathlib.Path, echo: bool = True) -> Any:
    path = _check_file_exists(path)
    tcbench.cli.logger.log(f"reading: {path}", echo=echo)
    with open(path, "rb") as fin:
        data = pickle.load(fin)
    return data

def save_pickle(data: Any, save_as: pathlib.Path, echo: bool = True) -> Any:
    save_as = pathlib.Path(save_as)
    create_folder(save_as.parent)
    tcbench.cli.logger.log(f"saving: {save_as}", echo=echo)
    with open(save_as, "wb") as fout:
        pickle.dump(data, fout)
    return data

def load_yaml(path: pathlib.Path, echo: bool = True) -> Dict[Any, Any]:
    tcbench.cli.logger.log(f"reading: {path}", echo=echo)
    with open(path) as fin:
        return yaml.safe_load(fin)

def save_yaml(data: Any, save_as: pathlib.Path, echo: bool = True) -> None:
    save_as = pathlib.Path(save_as)
    create_folder(save_as.parent)
    tcbench.cli.logger.log(f"saving: {save_as}", echo=echo)
    with open(save_as, "w") as fout:
        yaml.dump(data, fout)

def load_csv(path: pathlib.Path, echo: bool = True) -> pl.DataFrame:
    tcbench.cli.logger.log(f"loading: {path}", echo=echo)
    return pl.read_csv(path)

def save_csv(df: pl.DataFrame, save_as: pathlib.Path, echo: bool = True) -> None:
    save_as = pathlib.Path(save_as)
    create_folder(save_as.parent)
    tcbench.cli.logger.log(f"saving: {save_as}", echo=echo)
    df.write_csv(save_as)

def load_parquet(path: pathlib.Path, echo: bool = True) -> pl.DataFrame:
    logger.log(f"loading: {path}", echo=echo)
    return pl.read_parquet(path)

def save_parquet(df: pl.DataFrame, save_as: pathlib.Path, echo: bool = True) -> None:
    save_as = pathlib.Path(save_as)
    create_folder(save_as.parent)
    tcbench.cli.logger.log(f"saving: {save_as}", echo=echo)
    df.write_parquet(save_as)


LOAD_FUNC_BY_SUFFIX = {
    ".yaml": load_yaml,
    ".pkl": load_pickle,
    ".csv": load_csv,
    ".parquet": load_parquet,
}


def load_if_exists(path: pathlib.Path, echo: bool = True, error_policy: str = "return") -> Any:
    path = pathlib.Path(path)
    if not path.exists():
        if error_policy == "return":
            return None
        raise RuntimeError(f"FileNotFound: {path}")

    func = LOAD_FUNC_BY_SUFFIX.get(path.suffix, None)
    if not func:
        raise RuntimeError(f"Unrecognized suffix {path.suffix}")
    return func(path, echo=echo)
