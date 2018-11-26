from . import exceptions


class Receiver:
    def receive_context(self, context):
        pass

    def receive_report(self, report):
        pass

    def warn(self, message):
        exceptions.warn(message)


class Accumulator(Receiver):
    def __init__(self):
        self.test_indices = None
        self.warnings = []
        self.reports = []

    def receive_context(self, context):
        self.test_indices = context.test_indices

    def receive_report(self, report):
        self.reports.append(report)

    def warn(self, message):
        self.warnings.append(message)
