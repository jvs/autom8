from collections import namedtuple
import functools
import logging
import numpy as np
import sklearn.preprocessing
from .exceptions import expected


def create_training_context(feature_matrix, label_array, test_indices, problem_type):
    if problem_type not in ('regression', 'classification'):
        raise expected(
            'problem_type to be "regression" or "classification"', problem_type
        )

    if problem_type == 'regression':
        labels = LabelContext(label_array, label_array, None)
    else:
        assert problem_type == 'classification'
        encoder = sklearn.preprocessing.LabelEncoder()
        encoded = encoder.fit_transform(label_array)
        labels = LabelContext(label_array, encoded, encoder)

    return TrainingContext(feature_matrix, labels, test_indices, problem_type)


def create_predicting_context(feature_matrix):
    return PredictingContext(feature_matrix)


class _ContextMixin:
    def require_training_context(self):
        if not self.is_training:
            raise expected('TrainingContext', type(self).__name__)


class PredictingContext(_ContextMixin):
    def __init__(self, matrix):
        self.matrix = matrix

    @property
    def is_training(self):
        return False


class TrainingContext(_ContextMixin):
    def __init__(self, matrix, labels, test_indices, problem_type, preprocessors=None):
        self.matrix = matrix.copy()
        self.labels = labels
        self.test_indices = test_indices
        self.problem_type = problem_type
        self.preprocessors = list(preprocessors) if preprocessors else []

    def copy(self):
        return TrainingContext(
            matrix=self.matrix,
            labels=self.labels,
            test_indices=self.test_indices,
            problem_type=self.problem_type,
            preprocessors=self.preprocessors,
        )

    @property
    def is_training(self):
        return True

    @property
    def is_classification(self):
        return self.problem_type == 'classification'

    @property
    def is_regression(self):
        return self.problem_type == 'regression'

    def testing_data(self):
        feat = self.matrix.select_rows(self.test_indices)
        lab = self.labels.encoded[self.test_indices]
        return (feat.stack_columns(), lab)

    def training_data(self):
        feat = self.matrix.exclude_rows(self.test_indices)
        lab = np.delete(self.labels.encoded, self.test_indices)
        return (feat.stack_columns(), lab)


class LabelContext(namedtuple('LabelContext', 'original, encoded, encoder')):
    def decode(self, encoded_labels):
        if self.encoder is None:
            return encoded_labels
        else:
            return self.encoder.inverse_transform(encoded_labels)


def planner(f):
    @functools.wraps(f)
    def wrapper(ctx, *a, **k):
        ctx.require_training_context()
        try:
            f(ctx, *a, **k)
        except Exception:
            logging.exception('Planner "%s" failed', f.__name__)
    return wrapper


def preprocessor(f):
    @functools.wraps(f)
    def wrapper(ctx, *a, **k):
        f(ctx, *a, **k)
        ctx.preprocessors.append({'func': f, 'args': a, 'kwargs': k})
    return wrapper


def playback(preprocessors, ctx):
    for item in preprocessors:
        f, a, k = item['func'], item['args'], item['kwargs']
        try:
            f(ctx, *a, **k)
        except Exception:
            logging.exception('playback failed on step %r', f)
