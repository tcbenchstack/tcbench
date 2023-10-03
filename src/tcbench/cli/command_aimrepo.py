from rich.table import Table
from rich import box

import rich_click as click

import pathlib
import aim
import shutil

from tcbench.cli import clickutils
from tcbench.cli.clickutils import (
    CLICK_TYPE_DATASET_NAME,
)

from tcbench import (
    DATASETS,
    DEFAULT_AIM_REPO,
)

#from tcbench.modeling import aimutils
from tcbench.cli import console

click.rich_click.SHOW_ARGUMENTS = True
#click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.USE_RICH_MARKUP = True

PLUSMINUS = u"\u00B1"

def _report_rich_table(df):
    table_out = Table(box=None)
    table_out.add_column('', justify='center')
    table_out.add_column('hparams', justify='center')
    table_out.add_column('', justify='center')

    table_split = Table(show_edge=False, expand=True, box=box.HORIZONTALS, pad_edge=False)
    table_split.add_column('split', justify='right')

    table_runs = Table(show_edge=False, expand=True, box=box.HORIZONTALS, pad_edge=False)
    table_runs.add_column('runs', justify='right')

    table_hparams = Table(show_edge=False, expand=True, box=box.HORIZONTALS, pad_edge=False)

    # assuming index is a pd.MultiIndex
    for hparam in df.index.names[1:]:
        table_hparams.add_column(hparam.replace('hparams.', ''), justify='right')

    prev_row = list(map(str, df.index[0]))
    table_split.add_row(prev_row[0])
    table_hparams.add_row(*prev_row[1:])

    section_indexes = set()
    for idx, items in enumerate(df.index[1:], start=1):
        items = list(map(str, items))
        curr_row = [
            '' if curr == prev else curr
            for prev, curr in zip(prev_row, items)
        ]
        if curr_row[0] != '':
            table_split.add_section()
            table_hparams.add_section()
            section_indexes.add(idx)
        elif len(curr_row) > 2 and curr_row[1] != '':
            table_split.add_row('')
            table_hparams.add_section()
            section_indexes.add(idx)

        table_split.add_row(curr_row[0])
        table_hparams.add_row(*curr_row[1:])
        prev_row = items

    for idx, value in enumerate(df[('hash', 'runs')]):
        if idx in section_indexes:
            table_runs.add_section()
        table_runs.add_row(str(int(value)))

    tables_metric = []
    for metric_name in df.columns.levels[0]:
        if metric_name == 'hash':
            continue
        df_tmp = df[metric_name]
        table = Table(show_edge=False, expand=True, box=box.HORIZONTALS, pad_edge=False)
        table_out.add_column(metric_name, justify='center')
        
        for agg_metric_name in df_tmp.columns:
            table.add_column(agg_metric_name, justify='right')
            
        for idx in range(len(df_tmp)):
            if idx in section_indexes:
                table.add_section()
            ser = df_tmp.iloc[idx]
            table.add_row(*list(map(str, ser.values)))

        tables_metric.append(table)

    table_out.add_row(table_split, table_hparams, table_runs, *tables_metric)
    console.print()
    console.print(table_out)
    console.print()

@click.group("aimrepo")
@click.pass_context
def aimrepo(ctx):
    """Investigate AIM repository content."""
    pass

@aimrepo.command("ls")
@click.pass_context
@click.option(
    "--aim-repo",
    "aim_repo_folder",
    type=pathlib.Path,
    default=DEFAULT_AIM_REPO,
    show_default=True,
    help="AIM repository location (local folder or URL).",
)
def ls(ctx, aim_repo_folder):
    """List a subset of properties of each run."""
    from tcbench.modeling import aimutils

    if not aim_repo_folder.exists():
        raise RuntimeError(f'Not found {aim_repo_folder}')
    if not (aim_repo_folder / '.aim'):
        raise RuntimeError(f'The input {aim_repo_folder} is not an AIM repository')

    repo = aim.Repo(str(aim_repo_folder))
    prop = aimutils.get_repo_properties(repo)
    df = prop['df_run']
    cols = ['hash', 'creation_time', 'end_time']
    if 'campaign_id' in df.columns:
        cols.insert(1, 'campaign_id')
    df = df[cols]
    df = df.astype(str)

    table = Table(box=None)
    for col in df.columns:
        table.add_column(col)
    for idx in range(len(df)):
        table.add_row(*df.iloc[idx].values)
    console.print(table)

@aimrepo.command("properties")
@click.pass_context
@click.option(
    "--aim-repo",
    "aim_repo_folder",
    type=pathlib.Path,
    default=DEFAULT_AIM_REPO,
    show_default=True,
    help="AIM repository location (local folder or URL).",
)
def properties(ctx, aim_repo_folder):
    """List properties across all runs."""
    from tcbench.modeling import aimutils

    def format_duration(timedelta):
        parts = timedelta.components
        text = ''
        if parts.days > 0:
            text += f'{parts.days}d '
        if parts.hours > 0:
            text += f'{parts.hours}h'
        text += f'{parts.minutes}m{parts.seconds}s'
        return text

    repo = aim.Repo(str(aim_repo_folder))
    prop = aimutils.get_repo_properties(repo)

    table = Table(box=box.ROUNDED)
    table.add_column('Name', overflow='fold')
    table.add_column('No. unique', overflow='fold', justify='right')
    table.add_column('Value', overflow='fold')

    table.add_row('runs', '-', str(len(prop['df_run']))) 
    duration_mean, duration_std = prop['run_duration']
    duration_mean = format_duration(duration_mean)
    duration_std = format_duration(duration_std)
    table.add_row(f'run duration (mean {PLUSMINUS} std)', '-', f'{duration_mean} {PLUSMINUS} {duration_std}')
    table.add_row('metrics', str(len(prop['metrics'])), str(prop['metrics']))
    table.add_row('contexts', str(len(prop['contexts'])), str(prop['contexts']))

    table.add_section()
    for hparam_name in sorted([name for name in prop if name.startswith('hparam')] + ['experiment']):
        values = prop[hparam_name]
        table.add_row(
            hparam_name.replace('hparams.',''),
            f'{len(values)}',
            str(values),
        )
    console.print(table)

@aimrepo.command("report")
@click.pass_context
@click.option(
    "--aim-repo",
    "aim_repo_folder",
    type=pathlib.Path,
    default=DEFAULT_AIM_REPO,
    show_default=True,
    help="AIM repository location (local folder or URL).",
)
@click.option(
    "--campaign-id",
    "campaign_id",
    default=None,
    show_default=True,
    help="Campaign ID to select. By default consider the latest registered campaign.",
)
@click.option(
    "--metrics",
    "metrics",
    default="acc",
    show_default=True,
    help="Coma separated list of metrics to consider.",
)
@click.option(
    "--contexts",
    "splits",
    default=None,
    show_default=True,
    help="Coma separated list of test split names to consider. By default consider only split which name starts with 'test'",
)
@click.option(
    "--groupby",
    "groupby_params",
    default=None,
    show_default=True,
    help="Coma separated list of parameters to aggregate campaign results. By default, try to guess the list from the properties with more than one value."
)
@click.option(
    "--output-folder",
    "output_folder",
    default=None,
    show_default=True,
    type=str,
    help="Folder where to store output reports.",
)
@click.option(
    "--precision",
    "float_precision",
    default=2,
    show_default=2,
    type=int,
    help='Number of floating point digits in console report.',
)
def report(ctx, **kwargs):
    """Summarize runs performance metrics."""
    import pandas as pd
    from tcbench.modeling import aimutils, utils

    repo = aim.Repo(str(kwargs['aim_repo_folder']))

    prop = aimutils.get_repo_properties(repo)
    metrics = kwargs['metrics'].split(',')
    #if "duration" not in metrics:
    #    metrics.append('duration')

    contexts = kwargs['splits']
    if contexts is None:
        contexts = [
            context_name
            for context_name in prop['contexts']
            if context_name.startswith('test')
        ]
    else:
        contexts = contexts.split(',')

    groupby_params = kwargs['groupby_params']
    if groupby_params is None:
        groupby_params = []
        for prop_name, prop_values in prop.items():
            if prop_name.startswith('hparams') and \
               prop_name not in {
                'hparams.campaign_exp_idx', 
                'hparams.seed', 
                'hparams.split_index'
                } and \
               len(prop_values) > 1:
                groupby_params.append(prop_name)
        if len(groupby_params) == 0:
            groupby_params = ['hparams.campaign_id']
    else:
        l = []
        for param_name in groupby_params.split(','):
            if param_name in prop:
                l.append(param_name)
            elif f'hparams.{param_name}' in prop:
                l.append(f'hparams.{param_name}')
            else:
                console.print(f'[yellow]WARNING: parameter {param_name} unknown and will be skipped')
        groupby_params = l

    if 'test_split_name' not in groupby_params:
        groupby_params.insert(0, 'test_split_name')


    if any(metric_name not in prop['metrics'] for metric_name in metrics):
        raise RuntimeError(f'Metric {metrics_name} not available: possible values are {prop["metrics"]}')
    if any(context_name not in prop['contexts'] for context_name in contexts):
        raise RuntimeError(f'Split {context_name} not available: possible values are {prop["contexts"]}')

    metrics2 = metrics[:]
    #if "duration" not in metrics2:
    #    metrics2.append("duration")
    df = aimutils.metrics_to_pandas(repo, prop['df_run'], metrics2, contexts)
    df = df.assign(run_duration=df["end_time"] - df["creation_time"])


    for campaign_id in prop['hparams.campaign_id']:
        df_campaign = df[df['hparams.campaign_id'] == campaign_id]
        if len(df_campaign) == 0:
            console.print(f'WARNING: missing campaign_id:{campaign_id}')
            continue

        console.print(f'campaign_id: {campaign_id}')
        console.print(f'runs: {df_campaign["hash"].nunique()}')

        columns = groupby_params + metrics + ['run_duration', 'hash']
        #if "duration" not in columns:
        #    columns.append("duration")

        df_tmp = df_campaign[columns]
        df_tmp = df_tmp.astype({mtr:float for mtr in metrics})
        agg_funcs = {
            mtr_name: ["count", "mean", "std", ('ci95', utils.compute_confidence_intervals)]
            for mtr_name in metrics
        }
        agg_funcs["run_duration"] = ["mean", "std", ('ci95', utils.compute_confidence_intervals)]
        agg_funcs['hash'] = [('runs', 'count')]
        df_agg = (
            df_tmp
                .groupby(by=groupby_params)
                .agg(agg_funcs)
        )

        ## table to console
        _report_rich_table((df_agg
            .round(kwargs['float_precision'])
            .drop([tpl for tpl in df_agg.columns if tpl[1] == 'count'], axis=1)
        ))

        

        ## dump to file
        df_agg.index.names = [name.replace('hparams.', '') for name in df_agg.index.names]
        df_agg = df_agg.drop([('hash', 'runs')], axis=1)


        folder = kwargs['output_folder'] 
        if folder is None:
            folder = pathlib.Path(kwargs['aim_repo_folder']) / 'campaign_summary' / campaign_id
        else:
            folder = pathlib.Path(folder) / 'campaign_summary' / campaign_id
        if not folder.exists():
            folder.mkdir(parents=True)

        is_xgboost_on_pktseries = (
            'hparams.flow_representation' in df.columns and
            (df['hparams.flow_representation'].unique() == ['pktseries']).all()
        )
        hparam_name = 'flowpic_dim'
        if is_xgboost_on_pktseries:
            hparam_name = 'max_n_pkts'

        for hparam_value in df_campaign[f'hparams.{hparam_name}'].unique():
            # run info
            df_tmp = df_campaign[df_campaign[f'hparams.{hparam_name}'] == hparam_value]
            df_tmp = df_tmp.reset_index(drop=True)
            df_tmp.columns = [col.replace('hparams.','') for col in df_tmp.columns]
            fname = folder / f'runsinfo_{hparam_name}_{hparam_value}.parquet'
            console.print(f'saving: {fname}')
            df_tmp.to_parquet(fname)

            # metrics report
            df_tmp = df_agg.reset_index()
            if (hparam_name, '') in df_tmp.columns:
                df_tmp = df_tmp[df_tmp[hparam_name] == hparam_value]
                df_tmp = df_tmp.drop([(hparam_name, '')], axis=1)
            fname = folder / f'summary_{hparam_name}_{hparam_value}.csv'
            console.print(f'saving: {fname}')
            df_tmp.to_csv(fname, index=None)

@aimrepo.command("merge")
@click.pass_context
@click.option(
    "--src",
    "src_paths",
    type=pathlib.Path,
    multiple=True,
    metavar="PATH",
    required=True,
    help="AIM repository to merge.",
)
@click.option(
    "--dst",
    "dst_path",
    type=pathlib.Path,
    default=DEFAULT_AIM_REPO,
    show_default=True,
    help="New AIM repository to create.",
)
def merge(ctx, src_paths, dst_path):
    """Coalesce different AIM repos into a single new repo."""
    from tcbench.modeling import aimutils

    if dst_path.exists():
        shutil.rmtree(dst_path)

    aimutils.init_repository(dst_path)
    dst_repo = aim.Repo(str(dst_path))

    for path in src_paths:
        src_repo = aim.Repo(str(path))
        src_runs = aimutils.list_repo(src_repo)
        hashes = src_runs['hash'].values.tolist()
        src_repo.copy_runs(hashes, dst_repo)

        if (path / 'artifacts').exists():
            dst_path_artifacts = dst_path / 'artifacts'
            if not dst_path_artifacts.exists():
                dst_path_artifacts.mkdir(parents=True)
            for run_hash in hashes:
                shutil.copytree((path / 'artifacts' / run_hash), dst_path / 'artifacts' / run_hash)
        
