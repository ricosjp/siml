"""
SiML
"""

import toml
from pathlib import Path

from siml import datasets  # NOQA
from siml import inferer  # NOQA
from siml import networks  # NOQA
from siml import optimize  # NOQA
from siml import prepost  # NOQA
from siml import setting  # NOQA
from siml import study  # NOQA
from siml import trainer  # NOQA
from siml import util  # NOQA


def get_version():
    path = Path(__file__).resolve().parent.parent / 'pyproject.toml'
    pyproject = toml.loads(open(str(path)).read())
    return pyproject['tool']['poetry']['version']


__version__ = get_version()
