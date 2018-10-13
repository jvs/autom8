from collections import namedtuple
from contextlib import contextmanager
import logging
import numpy as np
import scipy.sparse
import sklearn.preprocessing

from .evaluate import evaluate_pipeline
from .exceptions import expected
from .observer import Observer
from .pipeline import Pipeline


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


class TrainingContext:
    def __init__(
            self, matrix, labels, test_indices,
            problem_type, observer, steps=None
        ):
        self.matrix = matrix.copy()
        self.labels = labels
        self.test_indices = test_indices
        self.problem_type = problem_type
        self.observer = observer
        self.steps = list(steps) if steps else []
        self.pool = None

    def copy(self):
        return TrainingContext(
            matrix=self.matrix,
            labels=self.labels,
            test_indices=self.test_indices,
            problem_type=self.problem_type,
            observer=self.observer,
            steps=self.steps,
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

    def __lshift__(self, estimator):
        if self.pool is None:
            self.fit(estimator)
        elif hasattr(self.pool, 'fit'):
            self.pool.fit(self, estimator)
        else:
            self.pool.submit(self.fit, estimator)

    def fit(self, estimator):
        X, y = self.training_data()
        try:
            X = scipy.sparse.csr_matrix(X.astype(float))
            estimator.fit(X, y)
        except Exception:
            logging.exception('Training failed')
            return

        pipeline = Pipeline(list(self.steps), estimator, self.labels.encoder)
        report = evaluate_pipeline(self, pipeline)
        self.observer.receive_pipeline(pipeline, report)

    def submit(self, func, *args, **kwargs):
        if self.pool is None:
            func(*args, **kwargs)
        else:
            self.pool.submit(f, *args, **kwargs)

    def testing_data(self):
        feat = self.matrix.select_rows(self.test_indices)
        lab = self.labels.encoded[self.test_indices]
        return (feat.stack_columns(), lab)

    def training_data(self):
        feat = self.matrix.exclude_rows(self.test_indices)
        lab = np.delete(self.labels.encoded, self.test_indices)
        return (feat.stack_columns(), lab)

    @contextmanager
    def sandbox(self):
        saved_matrix = self.matrix.copy()
        saved_steps = list(self.steps)
        yield
        self.matrix = saved_matrix
        self.steps = saved_steps

    @contextmanager
    def parallel(self):
        if self.pool is not None:
            yield
            return

        num_steps = len(self.steps)
        self.pool = self.observer.create_executor()
        yield

        try:
            self.pool.shutdown(wait=True)
        finally:
            self.pool = None

        if len(self.steps) != num_steps:
            self.observer.warn(
                'Potential race condition: The TrainingContext was updated'
                ' within a `parallel` context. To avoid any race conditions,'
                ' create a copy of the TrainingContext before applying any'
                ' preprocessing functions to it.'
            )


class LabelContext(namedtuple('LabelContext', 'original, encoded, encoder')):
    def decode(self, encoded_labels):
        if self.encoder is None:
            return encoded_labels
        else:
            return self.encoder.inverse_transform(encoded_labels)
