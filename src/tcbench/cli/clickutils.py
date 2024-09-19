from __future__ import annotations
import rich_click as click

from typing import List, Dict, Any

import functools

from tcbench.core import StringEnum
from tcbench.libtcdatasets.constants import (
    DATASET_NAME,
)
from tcbench.modeling import (
    MODELING_METHOD_TYPE, 
    MODELING_INPUT_REPR_TYPE
)


def _create_choice(enumeration:StringEnum) -> click.Choice:
    return click.Choice(enumeration.values(), case_sensitive=False)

def _parse_enum_from_str(command: str, parameter:str, value:str, enumeration:StringEnum) -> StringEnum:
    return enumeration.from_str(value)

def _parse_str_to_int(command: str, parameter: str, value: str) -> int:
    return int(value)

CLICK_CHOICE_DATASET_NAME = _create_choice(DATASET_NAME)
CLICK_PARSE_DATASET_NAME = functools.partial(_parse_enum_from_str, enumeration=DATASET_NAME)

#CLICK_CHOICE_METHOD_NAME = _create_choice(MODELING_METHOD_TYPE)
#CLICK_PARSE_METHOD_NAME = functools.partial(_parse_enum_from_str, enumeration=MODELING_METHOD_TYPE)

#CLICK_CHOICE_INPUT_REPR = _create_choice(MODELING_INPUT_REPR_TYPE)
#CLICK_PARSE_INPUT_REPR = functools.partial(_parse_enum_from_str, enumeration=MODELING_INPUT_REPR_TYPE)

CLICK_PARSE_STRTOINT = _parse_str_to_int


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
