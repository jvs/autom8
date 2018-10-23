from collections import namedtuple
import numpy as np
import scipy.sparse

from .matrix import create_matrix, Matrix
from .exceptions import Autom8Exception, expected, typename
from .preprocessors import playback
from .receiver import Receiver


PredictionReport = namedtuple('PredictionReport', 'predictions, probabilities')


class Pipeline:
    def __init__(self, input_columns, steps, estimator, label_encoder):
        self.input_columns = input_columns
        self.steps = steps
        self.estimator = estimator
        self.label_encoder = label_encoder

    def run(self, features, receiver=None):
        if not isinstance(features, (list, Matrix)):
            raise expected('list or Matrix', typename(features))

        if receiver is None:
            receiver = Receiver()

        if isinstance(features, list):
            features = create_matrix(features, receiver=receiver)

        # Rearrange the matrix so that the columns are in the right order.
        assert isinstance(features, Matrix)
        swizzled = self._swizzle(features, receiver)

        ctx = PipelineContext(swizzled, receiver)
        playback(self.steps, ctx)

        X = ctx.matrix.stack_columns()
        return self._predict(X)

    def _predict(self, X):
        if X.dtype != float:
            X = X.astype(float)

        has_proba = hasattr(self.estimator, 'predict_proba')
        probabilities = [] if has_proba else None
        predictions = []

        # TODO: Calculate an appropriate stride.
        stride = 1000
        for i in range(0, len(X), stride):
            window = scipy.sparse.csr_matrix(X[i : i + stride])

            if has_proba:
                y = self.estimator.predict_proba(window)
                predictions.extend(np.argmax(p) for p in y)
                probabilities.extend(self._format_probabilities(p) for p in y)
            else:
                predictions.extend(self.estimator.predict(window))

        if self.label_encoder is not None:
            predictions = self.label_encoder.inverse_transform(predictions)

        return PredictionReport(predictions, probabilities)

    def _format_probabilities(self, probs):
        decode = lambda i: self.label_encoder.inverse_transform([i])[0]
        pairs = [(s, decode(i)) for i, s in enumerate(probs) if s > 0]
        top3 = sorted(pairs, reverse=True)[:3]

        # Return the top three pairs.
        return [(label, score) for score, label in top3]

    def _swizzle(self, matrix, receiver):
        assert isinstance(matrix, Matrix)

        c1 = len(matrix.columns)
        c2 = len(self.input_columns)

        if c1 < c2:
            raise expected(f'matrix with at least {c2} columns', c1)

        try:
            return matrix.swizzle(self.input_columns)
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


class PipelineContext:
    def __init__(self, matrix, receiver):
        self.matrix = matrix
        self.receiver = receiver
        self.is_fitting = False
