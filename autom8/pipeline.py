from collections import namedtuple
import numpy as np
import scipy.sparse

from .matrix import create_matrix, Matrix
from .exceptions import Autom8Exception, expected, typename
from .preprocessors import playback
from .receiver import Receiver


PredictionReport = namedtuple('PredictionReport', 'predictions, probabilities')


class Pipeline:
    def __init__(self, input_columns, predicts_column, steps, estimator, label_encoder):
        self.input_columns = input_columns
        self.predicts_column = predicts_column
        self.steps = steps
        self.estimator = estimator
        self.label_encoder = label_encoder

    @property
    def predicts_classes(self):
        return self.label_encoder.classes_ if self.label_encoder else None

    def run(self, features, receiver=None):
        if not isinstance(features, (list, Matrix)):
            raise expected('list or Matrix', typename(features))

        if receiver is None:
            receiver = Receiver()

        if isinstance(features, list):
            features = create_matrix(features, receiver=receiver)

        # Rearrange the matrix so that the columns are in the right order.
        assert isinstance(features, Matrix)
        selected = self._select_columns(features, receiver)

        ctx = PlaybackContext(selected, receiver)
        playback(self.steps, ctx)
        return self._predict(ctx.matrix)

    def _predict(self, X):
        # TODO: Require a receiver, and notify it when the features need a lot
        # of coercion.
        X = X._float_array()

        has_proba = hasattr(self.estimator, 'predict_proba')
        probabilities = [] if has_proba else None
        predictions = []

        # TODO: Calculate an appropriate stride.
        stride = 1000
        for i in range(0, len(X), stride):
            window = scipy.sparse.csr_matrix(X[i : i + stride])

            if has_proba:
                y = self.estimator.predict_proba(window)
                rows = [self._format_probabilities(row) for row in y]
                predictions.extend(row[0][0] for row in rows)
                probabilities.extend(rows)
            else:
                y = self.estimator.predict(window)
                if self.label_encoder is not None:
                    y = self.label_encoder.inverse_transform(y)
                predictions.extend(y)

        return PredictionReport(
            _literalize(predictions),
            _literalize(probabilities),
        )

    def _format_probabilities(self, probs):
        classes = self.label_encoder.classes_
        pairs = [(c, p) for c, p in zip(classes, probs) if p > 0]
        return sorted(pairs, reverse=True, key=lambda pair: pair[1])[:3]

    def _select_columns(self, matrix, receiver):
        assert isinstance(matrix, Matrix)

        c1 = len(matrix.columns)
        c2 = len(self.input_columns)

        if c1 < c2:
            raise expected(f'matrix with at least {c2} columns', c1)

        try:
            return matrix.select_columns_by_name(self.input_columns)
        except Autom8Exception:
            pass

        receiver.warn(
            'Cannot match column names'
            f' from {[col.name for col in matrix.columns]}'
            f' to {self.input_columns}.'
        )

        if c1 == c2:
            return matrix
        else:
            return Matrix([col.copy() for col in matrix.columns[:c2]])


class PlaybackContext:
    def __init__(self, matrix, receiver):
        self.matrix = matrix
        self.receiver = receiver
        self.is_recording = False


def _literalize(obj):
    if hasattr(obj, 'tolist'):
        return obj.tolist()

    if isinstance(obj, list):
        return [_literalize(i) for i in obj]

    if isinstance(obj, tuple):
        return tuple(_literalize(i) for i in obj)

    return obj
