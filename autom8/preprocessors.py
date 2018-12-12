from collections import namedtuple
import functools
import itertools
import logging

import numpy as np
import sklearn.feature_extraction.text
import sklearn.preprocessing

from . import categories
from .exceptions import expected, typename


Step = namedtuple('Step', 'func, args, kwargs')


def planner(f):
    @functools.wraps(f)
    def wrapper(ctx, *a, **k):
        if not ctx.is_recording:
            raise expected('RecordingContext', typename(ctx))
        try:
            f(ctx, *a, **k)
        except Exception as exc:
            msg = f'Planning step "{f.__name__}" failed: {exc}'
            logging.getLogger('autom8').exception(msg)
            ctx.receiver.warn(msg)
    return wrapper


def preprocessor(f):
    @functools.wraps(f)
    def wrapper(ctx, *a, **k):
        f(ctx, *a, **k)
        if ctx.is_recording:
            ctx.steps.append(Step(wrapper, a, k))
    return wrapper


def playback(steps, ctx):
    for f, a, k in steps:
        try:
            f(ctx, *a, **k)
        except Exception:
            msg = f'Playback failed on step {f.__name__}'
            logging.getLogger('autom8').exception(msg)
            ctx.receiver.warn(msg)


@preprocessor
def add_column_of_ones(ctx):
    """Adds a column containing only the number 1, to help some estimators."""
    columns = ctx.matrix.columns
    num_rows = len(columns[0]) if columns else 0
    ctx.matrix.append_column(
        values=np.full(num_rows, 1),
        formula=['constant(1)'],
        role='numerical',
    )


@planner
def binarize_fractions(ctx):
    indices = _original_numerical_indices(ctx)
    if indices:
        _binarize_fractions(ctx, indices)


@preprocessor
def _binarize_fractions(ctx, indices):
    """Indicates when a column contains a fractional value."""
    found = ctx.matrix.select_columns(indices)
    found.coerce(float)
    for col in found.columns:
        ctx.matrix.append_column(
            values=np.absolute(col.values) < 1,
            formula=['is-fraction', col],
            role='encoded',
        )


@planner
def binarize_signs(ctx):
    indices = _original_numerical_indices(ctx)
    if indices:
        _binarize_signs(ctx, indices)


@preprocessor
def _binarize_signs(ctx, indices):
    """Indicates when a column contains a positive value."""
    found = ctx.matrix.select_columns(indices)
    found.coerce(float)
    for col in found.columns:
        ctx.matrix.append_column(
            values=col.values > 0,
            formula=['is-positive', col],
            role='encoded',
        )


@planner
def coerce_columns(ctx):
    dtypes = [col.dtype for col in ctx.matrix.columns]
    _coerce_columns(ctx, dtypes)


@preprocessor
def _coerce_columns(ctx, dtypes):
    """Forces each column to have the expected type."""

    for col, dtype in zip(ctx.matrix.columns, dtypes):
        if col.dtype == dtype or dtype == object:
            continue
        try:
            col.values = col.values.astype(dtype)
        except Exception:
            ctx.receiver.warn(
                f'Failed to convert column "{col.name}" to {dtype}'
            )


@planner
def divide_columns(ctx):
    numerators = _original_numerical_indices(ctx)
    # Find the indices of the original numerical columns that don't contain
    # any 0 values.
    denominators = _original_numerical_indices(ctx, lambda c: np.all(c.values != 0))

    if numerators and denominators:
        _divide_columns(ctx, numerators, denominators)


@preprocessor
def _divide_columns(ctx, numerators, denominators):
    """
    Divides pairs of numerical columns, as long as the divisor column
    does not contain any 0 values.
    """
    xs = ctx.matrix.select_columns(numerators)
    ys = ctx.matrix.select_columns(denominators)
    xs.coerce(float)
    ys.coerce(float)
    for col_x, col_y in itertools.product(xs.columns, ys.columns):
        x, y = col_x.values, col_y.values
        if not np.array_equal(x, y):
            # Use 0 if we see a denominator that equals 0.
            ctx.matrix.append_column(
                values=np.divide(x, y, out=np.zeros_like(x), where=y != 0),
                formula=['divide', col_x, col_y],
                role='numerical',
            )


@planner
def drop_duplicate_columns(ctx):
    duplicates = []
    visited = set()
    for index, col in enumerate(ctx.matrix.columns):
        signature = col.values.tobytes()
        if signature in visited:
            duplicates.append(index)
        else:
            visited.add(signature)
    if duplicates:
        _drop_duplicate_columns(ctx, duplicates)


@preprocessor
def _drop_duplicate_columns(ctx, duplicates):
    """Removes duplicate columns from your dataset."""
    ctx.matrix.drop_columns_by_index(duplicates)


@planner
def drop_weak_columns(ctx, feature_selector=None):
    import sklearn.feature_selection as fs

    if feature_selector is None and ctx.is_regression:
        feature_selector = fs.SelectFwe(fs.f_regression)

    if feature_selector is None and ctx.is_classification:
        feature_selector = fs.SelectFwe(fs.f_classif)

    X, y = ctx.training_data()
    X = X._float_array()

    feature_selector.fit(X, y)
    weak_cols = np.where(np.invert(feature_selector.get_support()))[0]

    # For now, ignore the selector if it wants to drop every column.
    if 0 < len(weak_cols) < len(ctx.matrix.columns):
        _drop_weak_columns(ctx, weak_cols.tolist())


@preprocessor
def _drop_weak_columns(ctx, indices):
    """Selects the best columns in your dataset."""
    ctx.matrix.drop_columns_by_index(indices)


@planner
def encode_categories(ctx, method='ordinal', only_strings=False):
    if method not in {'one-hot', 'ordinal'}:
        raise expected('one-hot or ordinal', method)

    indices = categories.select_indices(ctx, only_strings=only_strings)

    if indices:
        encoder = categories.create_encoder(ctx, method, indices)
        _encode_categories(ctx, encoder, indices)


@preprocessor
def _encode_categories(ctx, encoder, indices):
    """Turns categorical features into columns of numbers."""
    categories.encode(ctx, encoder, indices)


@planner
def encode_text(ctx):
    indices = ctx.matrix.column_indices_where(lambda col: col.role == 'textual')
    if indices:
        encoder = sklearn.feature_extraction.text.TfidfVectorizer(
            stop_words='english',
            min_df=5,
            max_features=2500,
            sublinear_tf=True
        )
        _encode_text(ctx, encoder, indices)


@preprocessor
def _encode_text(ctx, encoder, indices):
    """Turns textual features into vectors of numbers."""
    found = ctx.matrix.select_columns(indices)
    ctx.matrix.drop_columns_by_index(indices)

    # Combine textual columns into single column.
    found.coerce(str)
    combined = [' '.join(text) for text in found.stack_columns()]

    if ctx.is_recording:
        result = encoder.fit_transform(combined).toarray()
    else:
        # TODO: Append the appropriate number of 0-columns when this fails.
        result = encoder.transform(combined).toarray()

    for i in range(result.shape[1]):
        ctx.matrix.append_column(
            values=result[:, i],
            formula=['encode-text'] + found.columns,
            role='encoded',
        )


@planner
def logarithm_columns(ctx):
    # Find the indices of the original numerical columns that only contain
    # positive values.
    indices = _original_numerical_indices(ctx, lambda c: np.all(c.values > 0))
    if indices:
        _logarithm_columns(ctx, indices)


@preprocessor
def _logarithm_columns(ctx, indices):
    """
    Computes the natural log each numerical column, as long as the column
    only contains positive numbers.
    """
    found = ctx.matrix.select_columns(indices)
    found.coerce(float)
    for col in found.columns:
        v = col.values
        ctx.matrix.append_column(
            values=np.log(v, out=np.zeros_like(v), where=v > 0),
            formula=['log', col],
            role='numerical',
        )


@planner
def multiply_columns(ctx):
    indices = _original_numerical_indices(ctx)
    if indices:
        _multiply_columns(ctx, indices)


@preprocessor
def _multiply_columns(ctx, indices):
    """Multiplies pairs of numerical columns."""
    found = ctx.matrix.select_columns(indices)
    found.coerce(float)
    for x, y in itertools.combinations(found.columns, 2):
        ctx.matrix.append_column(
            values=x.values * y.values,
            formula=['multiply', x, y],
            role='numerical',
        )


@planner
def scale_columns(ctx, scaler=None):
    indices = ctx.matrix.column_indices_where(lambda col: col.is_numerical)
    if indices and scaler is None:
        scaler = sklearn.preprocessing.RobustScaler(quantile_range=(5, 95))

    if indices:
        _scale_columns(ctx, scaler, indices)


@preprocessor
def _scale_columns(ctx, scaler, indices):
    """Standardizes numerical columns using different scaling strategies."""
    found = ctx.matrix.select_columns(indices)
    found.coerce(float)

    array = found.stack_columns()
    ctx.matrix.drop_columns_by_index(indices)

    if ctx.is_recording:
        result = scaler.fit_transform(array)
    else:
        try:
            result = scaler.transform(array)
        except Exception:
            result = np.zeros_like(array)

    for i in range(result.shape[1]):
        ctx.matrix.append_column(
            values=result[:, i],
            formula=['scale', found.columns[i]],
            role='numerical',
        )


@planner
def square_columns(ctx):
    indices = _original_numerical_indices(ctx)
    if indices:
        _square_columns(ctx, indices)


@preprocessor
def _square_columns(ctx, indices):
    """Multiplies each numerical column by itself."""
    found = ctx.matrix.select_columns(indices)
    found.coerce(float)
    for col in found.columns:
        ctx.matrix.append_column(
            values=col.values * col.values,
            formula=['square', col],
            role='numerical',
        )


@planner
def sqrt_columns(ctx):
    # Find the indices of the original numerical columns that only contain
    # values greater than or equal to zero.
    indices = _original_numerical_indices(ctx, lambda c: np.all(c.values >= 0))
    if indices:
        _sqrt_columns(ctx, indices)


@preprocessor
def _sqrt_columns(ctx, indices):
    """
    Computes the square root of each numerical column, as long as the column
    does not contain any negative numbers.
    """
    found = ctx.matrix.select_columns(indices)
    found.coerce(float)
    for col in found.columns:
        v = col.values
        ctx.matrix.append_column(
            values=np.sqrt(v, out=np.zeros_like(v), where=v >= 0),
            formula=['square-root', col],
            role='numerical',
        )


def _original_numerical_indices(ctx, where=None):
    if where is None:
        where = lambda col: True
    return ctx.matrix.column_indices_where(lambda col:
        col.is_numerical and col.is_original and where(col))
