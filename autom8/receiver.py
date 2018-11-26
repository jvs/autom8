from . import exceptions


class Receiver:
    def receive_context(self, context):
        pass

    def receive_candidate(self, candidate):
        pass

    def warn(self, message):
        exceptions.warn(message)


class Accumulator(Receiver):
    def __init__(self):
        self.test_indices = None
        self.warnings = []
        self.candidates = []

    def receive_candidate(self, candidate):
        self.candidates.append(candidate)

    def receive_context(self, context):
        self.test_indices = context.test_indices

    def warn(self, message):
        self.warnings.append(message)
