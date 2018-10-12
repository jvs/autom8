from . import exceptions


class Observer:
    def create_executor(self):
        return ImmediateExecutor()

    def receive_pipeline(self, pipeline):
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
