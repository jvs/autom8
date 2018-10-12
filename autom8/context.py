from collections import namedtuple
import functools
import logging
import numpy as np
import sklearn.preprocessing

from .exceptions import expected
from .observer import Observer


def create_training_context(
    feature_matrix, label_array, test_indices, problem_type, observer=None
):
    if problem_type not in ('regression', 'classification'):
        raise expected(
            'problem_type to be "regression" or "classification"', problem_type
        )

    # Maybe create the matrix if the features are in a list.
    # Similarly, create the array if the labels are in a list.

    if observer is None:
        observer = Observer()

    if problem_type == 'regression':
        labels = LabelContext(label_array, label_array, None)
    else:
        assert problem_type == 'classification'
        encoder = sklearn.preprocessing.LabelEncoder()
        encoded = encoder.fit_transform(label_array)
        labels = LabelContext(label_array, encoded, encoder)

    return TrainingContext(
        feature_matrix, labels, test_indices, problem_type, observer
    )


class _ContextMixin:
    def require_training_context(self):
        if not self.is_training:
            raise expected('TrainingContext', type(self).__name__)


class PredictingContext(_ContextMixin):
    def __init__(self, matrix, observer):
        self.matrix = matrix
        self.observer = observer

    @property
    def is_training(self):
        return False


class TrainingContext(_ContextMixin):
    def __init__(
            self, matrix, labels, test_indices,
            problem_type, observer, preprocessors=None
        ):
        self.matrix = matrix.copy()
        self.labels = labels
        self.test_indices = test_indices
        self.problem_type = problem_type
        self.observer = observer
        self.preprocessors = list(preprocessors) if preprocessors else []

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
            msg = f'Planning step "{f.__name__}" failed'
            logging.exception(msg)
            ctx.observer.warn(msg)
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
            msg = f'Playback failed on step {f.__name__}'
            logging.exception(msg)
            ctx.observer.warn(msg)
