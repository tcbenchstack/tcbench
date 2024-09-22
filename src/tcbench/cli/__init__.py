import pathlib
import sys
import logging

from rich.console import Console, RenderableType
from rich.logging import RichHandler
from rich.protocol import rich_cast
from rich.theme import Theme

def get_rich_console(
    fname: pathlib.Path = None,    
    log_time: bool = False,
    record: bool = False,
) -> Console:
    curr_module = sys.modules[__name__]
    folder_module = pathlib.Path(curr_module.__file__).parent

    file=fname
    if fname is not None:
        fname = pathlib.Path(fname)
        if not fname.parent.exists():
            fname.parent.mkdir(parents=True)
        file = open(fname, "w")

    return Console(
        theme=Theme.read(folder_module / "rich.theme"),
        log_time=log_time,
        log_path=False,
        file=file,
        record=record,
    )

class ConsoleLogger:
    def __init__(
        self,
        #name: str = "tcbench",
        fname: pathlib.Path = None,
    ):
        self.console = get_rich_console(record=True)
        self.console_file = None
        if fname:
            self.fname = pathlib.Path(fname)
            self.console_file = get_rich_console(fname, log_time=True)

    def log(self, obj: str | RenderableType, echo: bool = True) -> None:
        if echo:
            self.console.print(obj)
        if self.console_file:
            self.console_file.log(obj)

    def save_svg(self, save_as: pathlib.Path, title: str = "") -> None:
        self.console.save_svg(save_as, title=title)


logger = ConsoleLogger()
console = logger.console


def reset_logger(fname: pathlib.Path = None) -> ConsoleLogger:
    global logger
    global console
    logger = ConsoleLogger(fname)
    console = logger.console
