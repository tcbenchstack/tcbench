def get_rich_console():
    from rich.console import Console
    from rich.theme import Theme
    import sys
    import pathlib

    curr_module = sys.modules[__name__]
    folder_module = pathlib.Path(curr_module.__file__).parent
    return Console(theme=Theme.read(folder_module / "rich.theme"))


console = get_rich_console()
