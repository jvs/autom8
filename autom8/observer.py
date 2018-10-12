import warnings
from .exceptions import Autom8Warning


class Observer:
    def warn(self, message):
        warnings.warn(Autom8Warning(message), stacklevel=3)
