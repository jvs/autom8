from collections import namedtuple
import numpy as np
import sklearn.metrics
from .exceptions import expected


Candidate = namedtuple('Candidate', 'pipeline, formulas, train, test')
MetricsReport = namedtuple('MetricsReport', 'predictions, probabilities, metrics')


def create_candidate(ctx, pipeline):
    """Creates a Candidate object from the provided context and pipeline.

    Parameters:
        ctx (RecordingContext): The current context.
        pipeline (Pipeline): The newly created pipeline.

    Returns:
        Candidate: The candidate object, containing the pipeline and its metrics.
    """

    return Candidate(
        pipeline=pipeline,
        formulas=ctx.matrix.formulas,
        train=_evaluate_predictions(ctx, pipeline, *ctx.training_data()),
        test=_evaluate_predictions(ctx, pipeline, *ctx.testing_data()),
    )


def _evaluate_predictions(ctx, pipeline, X, y):
    outputs = pipeline._predict(X)

    if ctx.is_regression:
        metrics = _evaluate_regressor(ctx, y, outputs.predictions)
        metrics.update(_extra_regressor_metrics(ctx, pipeline, metrics, len(y)))
    else:
        encoder = pipeline.label_encoder
        # MAY: Eventually use the probabilities, too.
        predicted = encoder.transform(outputs.predictions)
        metrics = _evaluate_classifier(ctx, y, predicted, encoder)

    return MetricsReport(outputs.predictions, outputs.probabilities, metrics)


def _evaluate_regressor(ctx, actual, predicted):
    funcs = [
        sklearn.metrics.explained_variance_score,
        sklearn.metrics.mean_absolute_error,
        sklearn.metrics.mean_squared_error,
        sklearn.metrics.mean_squared_log_error,
        sklearn.metrics.median_absolute_error,
        sklearn.metrics.r2_score,
    ]
    return _apply_metrics(funcs, actual, predicted)


def _extra_regressor_metrics(ctx, pipeline, initial_metrics, num_rows):
    funcs = [adjusted_r2_score]
    return _apply_metrics(funcs, ctx, pipeline, initial_metrics, num_rows)


def _evaluate_classifier(ctx, actual, predicted, encoder):
    funcs1 = [
        sklearn.metrics.f1_score,
        sklearn.metrics.precision_score,
        sklearn.metrics.recall_score,
    ]
    funcs2 = [
        sklearn.metrics.accuracy_score,
    ]
    funcs3 = [
        normalized_confusion_matrix,
        precision_recall_fscore_support,
    ]
    return {
        **_apply_metrics(funcs1, actual, predicted, average='weighted'),
        **_apply_metrics(funcs2, actual, predicted),
        **_apply_metrics(funcs3, actual, predicted, encoder),
    }


def _tolist(x):
    return x.tolist() if hasattr(x, 'tolist') else x


def _apply_metrics(funcs, *a, **k):
    result = {}
    for f in funcs:
        try:
            result[f.__name__] = _tolist(f(*a, **k))
        except Exception:
            pass
    return result


def normalized_confusion_matrix(actual, predicted, encoder):
    mat = sklearn.metrics.confusion_matrix(
        y_true=encoder.inverse_transform(actual),
        y_pred=encoder.inverse_transform(predicted),
        labels=encoder.classes_,
    )
    return mat.astype('float') / mat.sum(axis=1)[:, np.newaxis]


def precision_recall_fscore_support(actual, predicted, encoder):
    arrays = sklearn.metrics.precision_recall_fscore_support(
        y_true=encoder.inverse_transform(actual),
        y_pred=encoder.inverse_transform(predicted),
        labels=encoder.classes_,
    )
    result = []
    for cls, precision, recall, f1, support in zip(encoder.classes_, *arrays):
        result.append({
            'class': cls,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
        })
    return result


def adjusted_r2_score(ctx, pipeline, initial_metrics, num_rows):
    """Calculates the Adjusted R2 Score.

    See: https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2

    Parameters:
        ctx (RecordingContext): The current context.
        pipeline (Pipeline): The pipeline being evaluated.
        initial_metrics (dict): A dict of the metrics we've collected so far.
        num_rows (int): The number of rows in this segment of the dataset.

    Returns:
        float: The Adjusted R2 Score.
    """

    # Only count root columns that end up in a formula with a nonzero weight.
    # (A "root column" in a column that appeared in the initial matrix, before
    # any preprocessing, and was used to derive other columns.)
    roots = _nonzero_root_columns(ctx, pipeline)
    num_cols = len(roots)
    ratio = (num_rows - 1) / (num_rows - num_cols - 1)

    if ratio > 0:
        r2 = initial_metrics['r2_score']
        return 1 - (1 - r2) * ratio

    raise expected('more rows than columns',
        f'{num_rows} rows and {num_cols} columns.')


def _nonzero_root_columns(ctx, pipeline):
    est = pipeline.estimator
    weights = getattr(est, 'coef_', getattr(est, 'feature_importances_', None))
    columns = ctx.matrix.columns
    nonzeros = columns if weights is None else [
        c for c, w in zip(columns, weights) if w != 0
    ]
    return {i for c in nonzeros for i in c.root_columns()}
