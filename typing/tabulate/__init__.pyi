import textwrap
from _typeshed import Incomplete
from typing import NamedTuple

__all__ = ['tabulate', 'tabulate_formats', 'simple_separated_format']

class Line(NamedTuple):
    begin: Incomplete
    hline: Incomplete
    sep: Incomplete
    end: Incomplete

class DataRow(NamedTuple):
    begin: Incomplete
    sep: Incomplete
    end: Incomplete

class TableFormat(NamedTuple):
    lineabove: Incomplete
    linebelowheader: Incomplete
    linebetweenrows: Incomplete
    linebelow: Incomplete
    headerrow: Incomplete
    datarow: Incomplete
    padding: Incomplete
    with_header_hide: Incomplete

tabulate_formats: Incomplete

def simple_separated_format(separator): ...
def tabulate(tabular_data, headers=(), tablefmt: str = 'simple', floatfmt=..., intfmt=..., numalign=..., stralign=..., missingval=..., showindex: str = 'default', disable_numparse: bool = False, colalign=None, maxcolwidths=None, rowalign=None, maxheadercolwidths=None): ...

class JupyterHTMLStr(str):
    @property
    def str(self): ...

class _CustomTextWrap(textwrap.TextWrapper):
    max_lines: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
