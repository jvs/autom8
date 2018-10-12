class Autom8Exception(Exception):
    """Class of all exceptions raised by autom8."""


class Autom8Warning(UserWarning):
    """Class of all warnings issued by autom8."""


def expected(msg, received):
    return Autom8Exception(f'Expected {msg}. Received: {received}')
