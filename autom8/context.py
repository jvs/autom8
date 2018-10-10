from collections import namedtuple
import functools
import logging
import numpy as np
from .exceptions import expected


class _AbstractContext:
    def __init__(self, matrix, preprocessors=None, reports=None):
        self.matrix = matrix.copy()
        self.preprocessors = list(preprocessors) if preprocessors else []
        self.reports = dict(reports) if reports else {}

    @property
    def is_predicting(self):
        return not self.is_training

    def require_training_context(self):
        if not self.is_training:
            raise expected('TrainingContext', type(self).__name__)


class PredictingContext(_AbstractContext):
    def copy(self):
        return PredictingContext(
            matrix=self.matrix,
            preprocessors=self.preprocessors,
            reports=self.reports,
        )

    @property
    def is_training(self):
        return False


class TrainingContext(_AbstractContext):
    def __init__(
        self,
        feature_matrix,
        label_array,
        test_indices,
        problem_type='regression',
        preprocessors=None,
        reports=None,
    ):
        _AbstractContext.__init__(self, feature_matrix, preprocessors, reports)
        self.labels = label_array
        self.test_indices = test_indices
        self.problem_type = problem_type

    def copy(self):
        return TrainingContext(
            feature_matrix=self.matrix,
            label_array=self.labels,
            test_indices=self.test_indices,
            problem_type=self.problem_type,
            preprocessors=self.preprocessors,
            reports=self.reports,
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
        lab = self.labels[self.test_indices]
        return (feat.stack_columns(), lab)

    def training_data(self):
        feat = self.matrix.exclude_rows(self.test_indices)
        lab = np.delete(self.labels, self.test_indices)
        return (feat.stack_columns(), lab)


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
