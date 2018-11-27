import warnings


class Autom8Exception(Exception):
    """The class of all exceptions raised by autom8."""


class Autom8Warning(UserWarning):
    """The class of all warnings issued by autom8."""


def expected(expected, received):
    return Autom8Exception(f'Expected {expected}. Received: {received}')


def typename(obj):
    try:
        return type(obj).__name__
    except Exception:
        return 'unexpected object'


def warn(message):
    warnings.warn(Autom8Warning(message), stacklevel=4)
