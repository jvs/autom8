from . import exceptions


class Receiver:
    def receive_report(self, report):
        pass

    def warn(self, message):
        exceptions.warn(message)


class Accumulator(Receiver):
    def __init__(self):
        self.warnings = []
        self.reports = []

    def receive_report(self, report):
        self.reports.append(report)

    def warn(self, message):
        self.warnings.append(message)
