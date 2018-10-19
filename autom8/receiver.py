from . import exceptions


class Receiver:
    def receive_pipeline(self, pipeline, report):
        pass

    def warn(self, message):
        exceptions.warn(message)


class Accumulator(Receiver):
    def __init__(self):
        self.warnings = []
        self.pipelines = []

    def receive_pipeline(self, pipeline, report):
        self.pipelines.append((pipeline, report))

    def warn(self, message):
        self.warnings.append(message)
