"""
SiML
"""

import os.path as _op
from glob import glob as _glob

__all__ = [_op.splitext(module_name)[0] for module_name in  _glob("*.py")]
