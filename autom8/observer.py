from . import exceptions


class Observer:
    def warn(self, message):
        exceptions.warn(message)
