from pkg_resources import iter_entry_points

import rich_click as click

import tcbench
from tcbench import cli
from click_plugins import with_plugins


@with_plugins(iter_entry_points('click_command_tree'))
@click.group(invoke_without_command=True)
@click.pass_context
@click.option(
    "--version", "show_version", is_flag=True, help="Show tcbench version and exit."
)
def main(ctx, show_version):
    if show_version:
        import sys
        cli.console.print(f"version: {tcbench.__version__}")
        sys.exit()


from tcbench.cli.command_datasets import datasets
from tcbench.cli.command_singlerun import singlerun
from tcbench.cli.command_campaign import campaign
from tcbench.cli.command_aimrepo import aimrepo
from tcbench.cli.command_fetchartifacts import fetchartifacts

main.add_command(datasets)
main.add_command(singlerun)
main.add_command(campaign)
main.add_command(aimrepo)
main.add_command(fetchartifacts)

if __name__ == "__main__":
    main()
