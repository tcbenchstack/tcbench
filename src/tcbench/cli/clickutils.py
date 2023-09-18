import rich_click as click

from typing import List, Dict, Any

from tcbench import DATASETS
from tcbench.modeling import MODELING_METHOD_TYPE, MODELING_INPUT_REPR_TYPE


def _create_choice(enumeration):
    return click.Choice(list(map(lambda x: x.value, enumeration)), case_sensitive=False)


def _create_choice_callback(enumeration):
    return lambda c, p, v: enumeration.from_str(v)


CLICK_TYPE_DATASET_NAME = _create_choice(DATASETS)
CLICK_CALLBACK_DATASET_NAME = _create_choice_callback(DATASETS)

CLICK_TYPE_METHOD_NAME = _create_choice(MODELING_METHOD_TYPE)
CLICK_CALLBACK_METHOD_NAME = _create_choice_callback(MODELING_METHOD_TYPE)

CLICK_TYPE_INPUT_REPR = _create_choice(MODELING_INPUT_REPR_TYPE)
CLICK_CALLBACK_INPUT_REPR = _create_choice_callback(MODELING_INPUT_REPR_TYPE)

CLICK_CALLBACK_TOINT = lambda c, p, v: int(v)


def compose_help_string_from_list(items:List[str]) -> str:
    """Compose a string from a list"""
    return "\[" + f'{"|".join(items)}' + "]."


def convert_params_dict_to_list(params:Dict[str,Any], skip_params:List[str]=None) -> List[str]:
    """Convert a dictionary of parameters (name,value) pairs into a list of "--<param-name> <param-value>"""
    if skip_params is None:
        skip_params = set()

    l = []
    for par_name, par_value in params.items():
        if par_name in skip_params or par_value == False or par_value is None:
            continue
        par_name = par_name.replace("_", "-")
        if par_value == True:
            l.append(f"--{par_name}")
        else:
            l.append(f"--{par_name} {str(par_value)}")

    return l


def help_append_choices(help_string:str, values:List[str]) -> str:
    """Append to an help string a styled version of a list of values"""
    text = "|".join([f"[bold]{text}[/bold]" for text in values])
    return f"{help_string} [yellow]Choices: [{text}][/yellow]"
