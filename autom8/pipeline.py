from collections import namedtuple
import numpy as np

from .matrix import create_matrix, Matrix
from .observer import Observer
from .exceptions import expected, typename
from .preprocessors import playback


PredictionReport = namedtuple('PredictionReport', 'predictions, probabilities')


class Pipeline:
    def __init__(self, steps, estimator, label_encoder):
        self.steps = steps
        self.estimator = estimator
        self.label_encoder = label_encoder

    def run(self, features, observer=None):
        if not isinstance(features, (list, Matrix)):
            raise expected('list or Matrix', typename(features))

        if observer is None:
            observer = Observer()

        # TODO: Swizzle the input vectors to match the pipeline's schema.
        if isinstance(features, list):
            features = create_matrix(features, observer)

        ctx = PipelineContext(features, observer)
        playback(self.steps, ctx)

        X = ctx.matrix.stack_columns()

        if hasattr(self.estimator, 'predict_proba'):
            y = self.estimator.predict_proba(X)
            probabilities = [np.asscalar(p) for p in y]
            predictions = [np.argmax(p) for p in probabilities]
        else:
            probabilities = None
            predictions = self.estimator.predict(X)

        if self.label_encoder is not None:
            predictions = self.label_encoder.inverse_transform(predictions)

        return PredictionReport(predictions, probabilities)


class PipelineContext:
    def __init__(self, matrix, observer):
        self.matrix = matrix
        self.observer = observer

    @property
    def is_training(self):
        return False
