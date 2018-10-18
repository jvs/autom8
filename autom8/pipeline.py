from collections import namedtuple
import numpy as np
import scipy.sparse

from .matrix import create_matrix, Matrix
from .exceptions import expected, typename
from .preprocessors import playback
from .receiver import Receiver


PredictionReport = namedtuple('PredictionReport', 'predictions')
ProbabilityReport = namedtuple('ProbabilityReport',
    'predictions, probabilities, classes')


class Pipeline:
    def __init__(self, steps, estimator, label_encoder):
        self.steps = steps
        self.estimator = estimator
        self.label_encoder = label_encoder

    def run(self, features, receiver=None):
        if not isinstance(features, (list, Matrix)):
            raise expected('list or Matrix', typename(features))

        if receiver is None:
            receiver = Receiver()

        # TODO: Swizzle the input vectors to match the pipeline's schema.
        if isinstance(features, list):
            features = create_matrix(features, receiver)

        ctx = PipelineContext(features, receiver)
        playback(self.steps, ctx)

        X = ctx.matrix.stack_columns()
        return self._predict(X)

    def _predict(self, X):
        if X.dtype != float:
            X = X.astype(float)

        encoder = self.label_encoder

        if hasattr(self.estimator, 'predict_proba'):
            classes = [] if encoder is None else encoder.classes_
            report = ProbabilityReport([], [], classes)
        else:
            report = PredictionReport([])

        # TODO: Calculate an appropriate window size.
        for i in range(0, len(X), 1000):
            self._predict_window(report, X[i : i + 1000])

        if encoder is not None:
            decoded = encoder.inverse_transform(report.predictions)
            return report._replace(predictions=decoded)
        else:
            return report

    def _predict_window(self, report, X):
        X = scipy.sparse.csr_matrix(X)

        if hasattr(self.estimator, 'predict_proba'):
            y = self.estimator.predict_proba(X)
            report.predictions.extend(np.argmax(p) for p in y)
            report.probabilities.extend(p for p in y)
        else:
            report.predictions.extend(self.estimator.predict(X))


class PipelineContext:
    def __init__(self, matrix, receiver):
        self.matrix = matrix
        self.receiver = receiver

    @property
    def is_training(self):
        return False
