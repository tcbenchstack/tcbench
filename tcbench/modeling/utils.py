"""
This modules contains a set of utility function
to support a variety of tasks
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.stats.api as sms

from typing import Dict, Any

import numpy
from numpy.typing import NDArray

import aim
import yaml
import pathlib
import logging
import torch
import random
import argparse
import contextlib
import io
import torchsummary

from tcbench import DATASETS, MODELING_INPUT_REPR_TYPE
from tcbench.modeling import dataprep, backbone, methods

ARGPARSE_HELP_DEFAULT = "(default: %(default)s)"


def compose_cli_help_string(text: str) -> str:
    """Attach to the input text the default formatting string used by argparse to print help strings

    Arguments:
        text: the input text to process

    Return:
        The input text modified by appending "(default: %(default)s)"
    """
    return text + " " + ARGPARSE_HELP_DEFAULT


def load_yaml(fname: pathlib.Path) -> Dict[Any, Any]:
    """Load an input YAML file

    Arguments:
        fname: the YAML filename to load

    Return:
        The YAML object loaded
    """
    with open(fname) as fin:
        return yaml.safe_load(fin)


def load_config(fname: pathlib.Path) -> Dict:
    """Load the configuration file of the framework

    Arguments:
        fname: the YAML config file to load

    Return:
        The loaded config file
    """
    return load_yaml(fname)


def get_logger(fname: pathlib.Path) -> logging.Logger:
    """Create a logger attached to the console
    and also binds it to the filename passed as input.
    Anything printed via the logger will appear on the
    console and in the file

    Arguments:
        fname: the file name to bind to the logger

    Return:
        A new logger object associated to both
        console and the specified filename
    """
    fname = pathlib.Path(fname)

    logger = logging.getLogger("tcbench")

    # loggers are kept internally
    # and consistently returned based on name
    # thus, if handlers are set, it means
    # the logger was previously created.
    # In this case we close them
    # and create new ones
    if logger.handlers:
        logger.handlers[0].close()
        logger.handlers[1].close()
        logger.removeHandler(logger.handlers[0])
        logger.removeHandler(logger.handlers[0])

    # logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if not fname.parent.exists():
        fname.parent.mkdir(parents=True)

    fh = logging.FileHandler(fname)
    fh.setLevel(logging.DEBUG)
    print(f"opened log at {fname}")

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def log_msg(msg: str, logger: logging.Logger = None) -> None:
    """A generic function to print a message via a logger

    Arguments:
        msg: the text to print
        logger: the logger handling the printing
    """
    if logger is None:
        print(msg, flush=True)
    else:
        logger.debug(msg)


def seed_everything(seed: int) -> None:
    """Set the seed for pytorch, numpy and python

    Arguments:
        seed: the seed to use for the initialization
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_aim_run_hash(run: aim.Run) -> str:
    """Return the hash number of an AIM run based on its name

    Arguments:
        run: the AIM run from which extract the hash

    Return:
        The hash of the run
    """
    return run.name.split(" ")[1]


def dump_cli_args(
    cli_args: argparse.Namespace, save_as: pathlib.Path, logger: logging.Logger
) -> None:
    """Transform a argparse Namespace object into a dictionary
    which is saved as YAML file as well as printed by the logger

    Arguments:
        cli_args: a Namespace objected obtained by calling .parse_args() on a argparse parser
        save_as: a file where to save the arguments
        logger: a logger instance use for printing the parsed CLI arguments
    """
    params = dict()
    for key in dir(cli_args):
        if key[0] == "_":
            continue
        params[key] = getattr(cli_args, key)
        if isinstance(params[key], (pathlib.Path, DATASETS, MODELING_INPUT_REPR_TYPE)):
            params[key] = str(params[key])

    save_as = pathlib.Path(save_as)
    if not save_as.parent.exists():
        save_as.parent.mkdir(parents=True)
    log_msg(f"saving: {save_as}", logger)
    with open(save_as, "w") as fout:
        yaml.dump(params, fout)


def classification_reports(
    net: backbone.BaseNet,
    dset: dataprep.FlowpicDataset,
    batch_size: int,
    device: str = "cuda:0",
    context: str = "train",
    save_to: pathlib.Path = None,
    logger: pathlib.Path = None,
    method: str = "monolithic",
    xgboost_model: Any=None,
) -> Dict[str, pd.DataFrame]:
    """Compute scikit learn classification report
    and confusion matrix

    Arguments:
        net: the trained network to use
        dset: the dataset to use
        batch_size: the batch_size to use
        device: the device where to operate inferece
        context: a text string used for AIM tracking
        save_to: a folder where to store the report
        logger: the logger where to print the reports
        method: e.g. 'monolithic' (NN) or 'xgboost'
        xgboost_model: the xgboost model in case method=='xgboost'

    Return:
        A dictionary with two keys: "class_rep"
        and "conf_mtx". Each key is associated
        to a pandas DataFrame containing the
        classification report and confusion matrix
        computed based on inference
    """

    save_to = pathlib.Path(save_to)
    if not save_to.exists():
        save_to.mkdir(parents=True)

    dummy_trainer = methods.trainer_factory(
        method, net=net, device=device, logger=logger, xgboost_model=xgboost_model
    )

    loader = torch.utils.data.DataLoader(dset, batch_size, shuffle=False)
    _, reports = dummy_trainer.test_loop(loader, with_reports=True)

    labels = dset.df["app"].dtype.categories
    mapping = {str(idx): lab for idx, lab in enumerate(labels)}
    reports["class_rep"] = reports["class_rep"].rename(mapping, axis=0)
    ###
    mapping = {idx: lab for idx, lab in enumerate(labels)}
    reports["conf_mtx"] = reports["conf_mtx"].rename(mapping).rename(mapping, axis=1)

    log_msg("", logger)
    log_msg(f"---{context} reports---", logger)
    log_msg("", logger)
    log_msg(reports["class_rep"], logger)
    log_msg("", logger)
    log_msg(reports["conf_mtx"], logger)

    if save_to:
        log_msg("")
        fname = save_to / f"./{context}_class_rep.csv"
        log_msg(f"saving: {fname}", logger)
        reports["class_rep"].reset_index().to_csv(fname, index=None)

        fname = save_to / f"./{context}_conf_mtx.csv"
        log_msg(f"saving: {fname}", logger)
        reports["conf_mtx"].reset_index().to_csv(fname, index=None)
    return reports


def compute_confidence_intervals(array: NDArray, alpha: float = 0.05) -> float:
    """Computes the confidence intervasl from an array of values.

    Arguments:
        array: a list of values to process
        alpha: the alpha of the confidence interval

    Return:
        The function is based on statsmodels.stats.api.DescrStatsW()
        which reports lower and upper values of the interval.
        However we return the difference between mean value of the
        input array and the upper value of the confidence interval
    """
    array = np.array(array)
    low, high = sms.DescrStatsW(array).tconfint_mean(alpha)
    mean = array.mean()
    ci = high - mean
    return ci

def log_torchsummary(net:Any, input_shape:Any, logger:logging.Logger=None) -> None:
    """Log a model backbone architecture"""

    f_capture = io.StringIO()
    with contextlib.redirect_stdout(f_capture):
        torchsummary.summary(net, input_shape)

    log_msg(f_capture.getvalue(), logger)
