from collections import namedtuple
import numpy as np

from .context import PredictingContext, playback
from .matrix import create_matrix, Matrix
from .observer import Observer
from .exceptions import expected


PredictionReport = namedtuple('PredictionReport', 'predictions, probabilities')


class Pipeline:
    def __init__(self, preprocessors, estimator, label_encoder):
        self.preprocessors = preprocessors
        self.estimator = estimator
        self.label_encoder = label_encoder

    def run(self, features, observer=None):
        if not isinstance(features, (list, Matrix)):
            raise expected('list or Matrix', type(features).__name__)

        if observer is None:
            observer = Observer()

        if isinstance(features, list):
            features = create_matrix(features, observer)

        ctx = PredictingContext(features, observer)
        playback(self.preprocessors, ctx)

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
