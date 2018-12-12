from collections import namedtuple
from contextlib import contextmanager
import logging
import random

import numpy as np
import scipy.sparse

from .candidate import create_candidate
from .categories import LabelEncoder
from .docstrings import render_docstring
from .exceptions import expected, typename
from .inference import _infer_role
from .matrix import create_matrix, Matrix
from .pipeline import Pipeline
from .receiver import Receiver


@render_docstring
def create_context(
    dataset,
    column_names=None,
    column_roles=None,
    target_column=None,
    problem_type=None,
    test_ratio=None,
    random_state=None,
    allow_multicore=True,
    executor_class=None,
    receiver=None,
):
    """Returns a new `autom8.RecordingContext` object, ready to create pipelines.

    Note:
        If you are not writing your own preprocessing or training logic, then
        you do not need to create your own context. You can use either
        `autom8.fit()` or `autom8.run()`, either of which will automatically
        create a context for you.

    Parameters:
        $all_context_parameters
    """

    # Create a default receiver if the user didn't provide one.
    if receiver is None:
        receiver = Receiver()

    # Small optimization: don't copy the matrix if we don't need to.
    if isinstance(dataset, Matrix) and column_names is None and column_roles is None:
        matrix = dataset
    else:
        matrix = create_matrix(
            dataset=dataset,
            column_names=column_names,
            column_roles=column_roles,
            receiver=receiver,
        )

    num_cols = len(matrix.columns)

    if num_cols == 0:
        raise expected('dataset with more than one column', repr(dataset))

    if num_cols == 1:
        raise expected('dataset with more than one column', num_cols)

    if target_column is None or target_column == -1:
        target_column = num_cols - 1

    if not isinstance(target_column, (int, str)):
        raise expected(f'target_column to be an int or a str',
            typename(target_column))

    if isinstance(target_column, str):
        target_column = target_column.strip()
        for index, col in enumerate(matrix.columns):
            if target_column == col.name.strip():
                target_column = index
                break

    if isinstance(target_column, str):
        raise expected(f'target_column to be one of: {matrix.column_names}',
            repr(target_column))

    if isinstance(target_column, int) and target_column >= len(matrix.columns):
        raise expected(
            f'target_column to be valid column number (less than {num_cols})',
            target_column,
        )

    labelcol = matrix.columns[target_column]
    label_name, label_values = labelcol.name, labelcol.values
    matrix.drop_columns_by_index(target_column)

    if problem_type is None:
        role = _infer_role(labelcol, receiver)
        problem_type = 'regression' if role == 'numerical' else 'classification'

    valid_problems = {'regression', 'classification'}
    if not isinstance(problem_type, str) or problem_type not in valid_problems:
        raise expected(f'problem_type in {valid_problems}', repr(problem_type))

    if problem_type == 'regression':
        labels = Labels(label_name, label_values, label_values, None)
    else:
        assert problem_type == 'classification'
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(label_values)
        labels = Labels(label_name, label_values, encoded, encoder)

    if test_ratio is None:
        test_ratio = 0.2

    if not isinstance(test_ratio, float) or not (0 < test_ratio < 1):
        raise expected('test_ratio between 0.0 and 1.0', test_ratio)

    count = len(matrix)
    test_indices = sorted(random.sample(range(count), int(count * test_ratio) or 1))

    if executor_class is None:
        executor_class = SynchronousExecutor

    return RecordingContext(
        matrix=matrix,
        labels=labels,
        test_indices=test_indices,
        problem_type=problem_type,
        random_state=random_state,
        allow_multicore=allow_multicore,
        executor_class=executor_class,
        receiver=receiver,
    )


class RecordingContext:
    """Keep track of autom8's internal state as it generates candidate
    pipelines.

    Notes:
        You can get a reference to this object in your receiver's
        `receive_context` method. If you keep your reference to the context
        around, keep in mind that autom8 mutates the context as it generates
        pipelines. Specically, the `matrix` and `steps` attributes frequently
        change while `autom8.run()` is running.

    Attributes:
        matrix (Matrix): The current feature matrix.
        labels (Labels): The labels that we're trying to predict.

            This is essentially the target column, but the values may be
            encoded, depending on whether or the data is categorical or
            numerical.

        test_indices (list[int]): A list of indices. Indicates which rows in
            should be used in the test dataset.
        problem_type (str): Either `classification` or `regression`.
        allow_multicore (bool): Indicates if estimators may use multiple cores.
        executor_class (class): The executor class that autom8 should use when
            it wants to run tasks in parallel.
        receiver (Receiver): An object that receives out-of-band data, like
            candidate pipelines and warnings.
        steps (list[Step]): A list of all the preprocessing steps that have
            been applied to the feature matrix.
        pool (Executor): The current executor, for executing tasks in parallel.
        is_recording (bool, always True): Indicates that this is a
            `RecordingContext` as opposed to a `PlaybackContext`.
    """

    def __init__(
            self, matrix, labels, test_indices, problem_type,
            random_state, allow_multicore, executor_class, receiver,
        ):
        self.input_columns = matrix.column_names
        self.matrix = matrix.copy()
        self.labels = labels
        self.test_indices = test_indices
        self.problem_type = problem_type
        self.random_state = random_state
        self.allow_multicore = allow_multicore
        self.executor_class = executor_class
        self.receiver = receiver
        self.steps = []
        self.pool = None
        self.is_recording = True

    @property
    def is_classification(self):
        return self.problem_type == 'classification'

    @property
    def is_regression(self):
        return self.problem_type == 'regression'

    @property
    def predicts_column(self):
        return self.labels.name

    @property
    def predicts_classes(self):
        return self.labels.classes

    @property
    def random_state_kw(self):
        return ({} if self.random_state is None
            else {'random_state': self.random_state})

    def __lshift__(self, estimator):
        """Provides convenient syntax for calling submit_fit.

        So instead of:

            ctx.submit_fit(xgboost.XGBRegressor(n_jobs=-1, random_state=1))

        You can write:

            ctx << xgboost.XGBRegressor(n_jobs=-1, random_state=1)
        """

        self.submit_fit(estimator)

    def submit_fit(self, estimator):
        """Submits `self.fit(estimator)` to the current executor.

        If the context does not currently have an executor, then this method
        simply calls `self.fit(estimator)`.
        """

        if self.pool is None:
            self.fit(estimator)
        elif hasattr(self.pool, 'fit'):
            self.pool.fit(self, estimator)
        else:
            self.pool.submit(self.fit, estimator)

    def fit(self, estimator):
        X, y = self.training_data()
        X = scipy.sparse.csr_matrix(X._float_array())

        try:
            estimator.fit(X, y)
        except Exception:
            logging.getLogger('autom8').exception('Failure in fit method')
            return

        pipeline = Pipeline(
            input_columns=self.input_columns,
            predicts_column=self.predicts_column,
            steps=list(self.steps),
            estimator=estimator,
            label_encoder=self.labels.encoder,
        )

        candidate = create_candidate(self, pipeline)
        self.receiver.receive_candidate(candidate)

    def submit(self, func, *args, **kwargs):
        if self.pool is None:
            func(*args, **kwargs)
        else:
            self.pool.submit(f, *args, **kwargs)

    def testing_data(self):
        feat = self.matrix.select_rows(self.test_indices)
        lab = self.labels.encoded[self.test_indices]
        return (feat, lab)

    def training_data(self):
        feat = self.matrix.exclude_rows(self.test_indices)
        lab = np.delete(self.labels.encoded, self.test_indices)
        return (feat, lab)

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
        self.pool = self.executor_class()
        yield

        try:
            self.pool.shutdown(wait=True)
        finally:
            self.pool = None

        if len(self.steps) != num_steps:
            self.receiver.warn(
                'Potential race condition: The RecordingContext was updated'
                ' within a `parallel` context. To avoid any race conditions,'
                ' use the `sandbox` method to create a temporary copy of the'
                ' RecordingContext. Then apply your preprocessing functions'
                ' to the sandboxed copy.'
            )


class Labels(namedtuple('Labels', 'name, original, encoded, encoder')):
    @property
    def classes(self):
        return self.encoder.classes_ if self.encoder else None



class SynchronousExecutor:
    def fit(self, context, estimator):
        self.submit(context.fit, estimator)

    def submit(self, func, *args, **kwargs):
        func(*args, **kwargs)

    def shutdown(self, wait=True):
        pass
