import rich_click as click

import pathlib
import shutil
import tempfile

from tcbench.cli import clickutils
from tcbench.cli import console

click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.USE_RICH_MARKUP = True

FIGSHARE_RESOURCES_FNAME = "FIGSHARE_RESOURCES.yml"

def _copy_file(src, dst):
    keyword = "installing"
    if pathlib.Path(dst).exists():
        keyword = "overwriting"
    print(f"{keyword}: {dst}")
    shutil.copy2(src, dst)

@click.command("fetch-artifacts")
@click.pass_context
def fetchartifacts(ctx):
    """Download from figshare and install all required artifacts."""
    from tcbench.libtcdatasets import datasets_utils
    import requests

    check_exists = [
        pathlib.Path("./src/tcbench"),
        pathlib.Path("./tests"),
        pathlib.Path("./notebooks/tutorials"),
        pathlib.Path("./pyproject.toml"),
    ]
    if any(not folder.exists() for folder in check_exists):
        raise RuntimeError("Run the command from within the cloned github repository")

    fname = datasets_utils._get_module_folder().parent / FIGSHARE_RESOURCES_FNAME
    data = datasets_utils.load_yaml(fname)
    for primary_key in data:
        for secondary_key in data[primary_key]:
            print(f"fetching: {primary_key} / {secondary_key}")

            params = data[primary_key][secondary_key]

            url = params["url"]
            dst_folder = params["dst_folder"]
            with tempfile.TemporaryDirectory() as tmpfolder:
                tmpfolder = pathlib.Path(tmpfolder)
                try:
                    path = datasets_utils.download_url(url, tmpfolder)
                except requests.exceptions.SSLError:
                    path = datasets_utils.download_url(url, tmpfolder, verify=False)

                untar_folder = tmpfolder / "__untar__"
                datasets_utils.untar(path, untar_folder)
                path.unlink()
                shutil.copytree(untar_folder, dst_folder, copy_function=_copy_file, dirs_exist_ok=True)
