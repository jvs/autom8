from . import exceptions


class Receiver:
    def create_executor(self):
        return ImmediateExecutor()

    def receive_pipeline(self, pipeline, report):
        exceptions.warn(f'Unused pipeline: {pipeline}')

    def warn(self, message):
        exceptions.warn(message)


class ImmediateExecutor:
    def fit(self, context, estimator):
        self.submit(context.fit, estimator)

    def submit(self, func, *args, **kwargs):
        func(*args, **kwargs)

    def shutdown(self, wait=True):
        pass


class Accumulator(Receiver):
    def __init__(self):
        self.warnings = []
        self.pipelines = []

    def receive_pipeline(self, pipeline, report):
        self.pipelines.append((pipeline, report))

    def warn(self, message):
        self.warnings.append(message)
