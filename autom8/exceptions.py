
class AutoM8Exception(Exception):
    """Base class of all exceptions raised by autom8."""


def expected(msg, received):
    return AutoM8Exception(f'Expected {msg}. Received: {received}')
