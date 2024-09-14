from __future__ import annotations

from typing import Any

import pickle
import yaml
import pathlib


def _check_file_exists(path: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path)
    if not path.exists():
        raise RuntimeError(f"FileNotFound: {path}")
    return path

def create_folder(folder: pathlib.Path) -> pathlib.Path:
    folder = pathlib.Path(folder)
    if not folder.exists():
        folder.mkdir(parents=True)
    return folder

def load_pickle(path: pathlib.Path) -> Any:
    path = _check_file_exists(path)
    with open(path, "rb") as fin:
        data = pickle.load(fin)
    return data

def save_pickle(data: Any, save_as: pathlib.Path) -> Any:
    save_as = pathlib.Path(save_as)
    create_folder(save_as.parent)
    with open(save_as, "wb") as fout:
        pickle.dump(data, fout)
    return data
