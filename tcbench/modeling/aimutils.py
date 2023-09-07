#!/usr/bin/env python
# coding: utf-8
"""
This module contains a collection 
of utility functions used to interact
with a AIM repository
"""
from __future__ import annotations
from typing import Tuple, List

import pandas as pd
import numpy as np

import aim
import pathlib
import subprocess
import sys
import itertools
import logging

from tcbench.modeling import utils
from tcbench.cli import console


def init_repository(folder:pathlib.Path, logger:logging.Logger=None) -> None:
    if str(folder).startswith("aim://"):
        return

    folder = pathlib.Path(folder)
    if (folder / ".aim").exists():
        return

    if not folder.exists():
        folder.mkdir(parents=True)

    cmd = f"aim init --repo {folder}"
    proc = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    stdout = stdout.decode("utf-8")
    stderr = stderr.decode("utf-8")
    if stdout != "":
        utils.log_msg(stdout, logger)
    if stderr != "":
        utils.log_msg(stderr, logger)
    if proc.returncode != 0:
        sys.exit(proc.returncode)

def list_repo(repo: aim.Repo) -> pd.DataFrame:
    """List all runs in the repository as pandas DataFrame"""
    #return pd.concat(run.dataframe() for run in repo.iter_runs())
    return pd.concat(
        item.run.dataframe()
        for item in repo.query_runs('', report_mode=0).iter_runs()
    )

def get_metric_names(repo: aim.Repo) -> List[str]:
    return sorted(list({ 
        item.name
        for item in repo.query_metrics(report_mode=0)
        if not item.name.startswith('__system')
    }))

def get_context_names(repo: aim.Repo) -> List[str]:
    return sorted(list({ 
        item.context['subset']
        for item in repo.query_metrics(report_mode=0)
        if not item.name.startswith('__system')
    }))


def get_repo_properties(repo: aim.Repo) -> Dict[str, Any]:
    with console.status(f"collecting repo properties...", spinner="dots"):
        df = list_repo(repo)
        metrics = get_metric_names(repo)
        contexts = get_context_names(repo)

    df = df[[
        col
        for col in df.columns
        if not col.startswith('__system')
    ]]
#    df = df.assign(
#        creation_time = pd.to_datetime(df['creation_time'], unit='s'),
#        end_time = pd.to_datetime(df['end_time'], unit='s')
#    )
#    df = df.assign(
#        duration = pd.to_datetime(df['end_time'], unit='s') - pd.to_datetime(df['creation_time'], unit='s')
#    )
    #df = df.sort_values(by='creation_time')

    properties = dict(
        df_run=df,
        metrics=metrics,
        contexts=contexts,
        run_duration=(
            pd.Timedelta((df['end_time'] - df['creation_time']).mean(), unit='s'),
            pd.Timedelta((df['end_time'] - df['creation_time']).std(), unit='s'),
        ),
    )

    for col in df.columns:
        properties[col] = sorted(df[col].unique().tolist())
    return properties
        

def get_latest_campaign_id(repo: aim.Repo, experiment_name: str) -> str:
    """Extract the latest campaign id (defined when launching
    a modeling campaign) from a AIM repo

    Arguments:
        repo: the AIM repository to use
        experiment_name: the experiment name from which extract the last campaign

    Return:
        The campaign id found
    """
    query = f"run.experiment == '{experiment_name}'"
    df = pd.concat(
        [
            entry.run.dataframe()
            for entry in repo.query_runs(query, report_mode=0).iter_runs()
        ]
    )
    campaign_ids = df["hparams.campaign_id"].replace({np.nan: "0"})
    return campaign_ids.unique().max()


def load_campaign(
    repo: aim.Repo, 
    campaign_id: str = None,
    experiment_name: str = None
) -> Tuple[pd.DataFrame, List[aim.Run]]:
    """Load the latest campaign of experiment into a pandas DataFrame

    Arguments:
        repo: the AIM repository to query
        campaign_id: the campaing_id of the runs to select (if None,
            it will search for the latest campaign_id)
        experiment_name: the experiment_name for the runs to select

    Return:
        A tuple with two objects: a DataFrame collecting the informatio
        for all the runs associated to the campaign, and a list
        of run object
    """
    if campaign_id is None:
        campaign_id = get_latest_campaign_id(repo, args.experiment_name)
        print(f"latest campaign_id: {campaign_id}")

    query = f"""
    run.hparams["campaign_id"] == '{campaign_id}'
    """
    if experiment_name != '':
        query += f"and run.experiment == '{experiment_name}'"

    runs = []
    l = []
    for entry in repo.query_runs(query, report_mode=0).iter_runs():
        l.append(entry.run.dataframe())
        runs.append(entry.run)

    df = pd.concat(l)

    ## remove __system columns
    cols_to_drop = [col for col in df.columns if col.startswith("__system")]
    if cols_to_drop:
        df = df.drop(cols_to_drop, axis=1)

    ## rename hparams.blablabla
    rename_cols = {col: col.replace("hparams.", "") for col in df.columns}
    if rename_cols:
        df = df.rename(rename_cols, axis=1)

    return df, runs


def query_metric(
    repo: aim.Repo,
    run_hashes: List[str],
    metric: str,
    context: str,
) -> Dict[str, Any]:
    """Collect all metrics associated to a list of runs

    Arguments:
        repo: the AIM repo to query
        run_hashes: a list of run hash values to search
        metric: the metric to extract (e.g., "acc")
        context: the split on which the metric was computed (e.g., "test")

    Return:
        A dictionary where keys maps to run hashes, and values
        to the related metric found
    """
    query = """
    metric.name == '{metric}' and 
    metric.context['subset'] == '{context}'
    """.format(
        metric=metric, context=context
    )
    query = f"run.hash in {run_hashes} and {query}"

    metrics = {
        item.run.hash: item.values.last()[1]
        for item in repo.query_metrics(query, report_mode=0)
    }

    if len(metrics) != len(run_hashes):
        missing_hashes = set(run_hashes) - set(metrics.keys())
        print(
            f"WARNING: found {len(metrics)} metrics for metric={metric} context={context}"
        )
        for run_hash in missing_hashes:
            if run_hash not in metrics:
                metrics[run_hash] = np.nan

    return metrics


def metrics_to_pandas(
    repo: aim.Repo,
    df_run: pd.DataFrame,
    metrics: List[str],
    contexts: List[str],
) -> pd.DataFrame:
    """Loads a set of metrics across multiple test splits into
    a dataframe already containing campaign related information

    Arguments:
        repo: the AIM repo to query
        df_run: a dataframe obtained invoking .load_latest_campaign()
        metrics: the list of metrics to load
        contexts: the set of split names to query metrics from

    Return:
        Expands df_campaign by adding columns related to the loaded metric values
    """

    df_campaign = df_run.copy()

    ## add (empty) metrics columns
    for mtr in metrics:
        df_campaign.loc[:, mtr] = np.nan

    df_campaign = df_campaign.assign(test_split_name=None)

    l = []
    for ctx in contexts: 
        with console.status(f"collecting metrics {ctx}...", spinner="dots"):
            df_new = df_campaign.copy()

            df_new = df_new.assign(test_split_name=ctx)
            for mtr in metrics:
                df_new.loc[:, mtr] = df_new["hash"].copy()
                metric_dict = query_metric(
                    repo=repo,
                    run_hashes=set(df_campaign["hash"].values),
                    metric=mtr,
                    context=ctx,
                )
                df_new.loc[:, mtr] = df_new[mtr].replace(metric_dict)
            l.append(df_new)

    df_new = pd.concat(l)

    return df_new

def track_metrics(
    tracker: aim.Run, metrics: Dict[str, Any], context: str, epoch: int = None
) -> None:
    """Save into a run metrics information

    Arguments:
        tracker: the AIM run where to save metrics
        metrics: a dictionary of key-value pairs to save
        context: the context related to the metrics (e.g., train/val/test)
        epoch: the epoch when the metrics where collected
    """
    for name, value in metrics.items():
        tracker.track(value, name, epoch=epoch, context=dict(subset=context))

# DEPRECATED
def summary_report(repo, groupby, metrics='acc', campaign_id=None, echo=True):
    df_campaign, runs = load_latest_campaign(
        repo, args.experiment_name, campaign_id
    )
    print(f"found: {len(runs)} runs")

    contexts = get_context_names(repo)

    if isinstance(metrics, str):
        metrics=[metrics]
    elif metrics is None:
        metrics = ['acc']

    test_split_name = [ctx for ctx in contexts if ctx.startswith('test')]

    for flowpic_dim in df_campaign["flowpic_dim"].unique():
        df_tmp = df_campaign[df_campaign["flowpic_dim"] == flowpic_dim]
        df = aimutils.metrics_to_pandas(
            repo, args.experiment_name, df_tmp, test_split_names, metrics_name
        )
        df = df.assign(duration=df["end_time"] - df["creation_time"])

        agg_funcs = {
            mtr_name: ["count", "mean", "std", utils.compute_confidence_intervals]
            for mtr_name in metrics_name
        }
        agg_funcs["duration"] = ["mean", "std", utils.compute_confidence_intervals]
        df_agg = df.groupby(by=["test_split_name", "aug_name"]).agg(agg_funcs)
        df_agg = df_agg.rename({"compute_confidence_intervals": "ci95"}, axis=1)

        for mtr_name in metrics_name:
            print()
            print(f"--- flowpic_dim {flowpic_dim} ({mtr_name}) ---")
            print(df_agg[mtr_name])
            print("---")

        fname = campaign_folder / f"summary_flowpic_dim_{flowpic_dim}.csv"
        print(f"saving: {fname}")
        df_agg.reset_index().to_csv(fname, index=None)

        fname = campaign_folder / f"runsinfo_flowpic_dim_{flowpic_dim}.parquet"
        print(f"saving: {fname}")
        df.reset_index(drop=True).to_parquet(fname)
