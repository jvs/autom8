from collections import namedtuple

import numpy as np
import pandas as pd
import sklearn.metrics
import scipy

from .exceptions import expected


Evaluation = namedtuple('Evaluation',
    'problem_type, test_indices, derived_columns, train, test, stats')

EvaluationSection = namedtuple('EvaluationSection',
    'predictions, probabilities, metrics')


def evaluate_pipeline(ctx, pipeline):
    return Evaluation(
        problem_type=ctx.problem_type,
        test_indices=ctx.test_indices,
        derived_columns=ctx.matrix.formulas,
        train=_evaluate_predictions(ctx, pipeline, *ctx.training_data()),
        test=_evaluate_predictions(ctx, pipeline, *ctx.testing_data()),
        stats=_calculate_stats(ctx, pipeline),
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

    return EvaluationSection(outputs.predictions, outputs.probabilities, metrics)


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


def _apply_metrics(funcs, *a, **k):
    result = {}
    for f in funcs:
        try:
            result[f.__name__] = f(*a, **k)
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
    """
    Calculates the Adjusted R2 Score.
    See: https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2
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


def _calculate_stats(ctx, pipeline):
    if not ctx.is_regression:
        return None

    result = {}
    try:
        # Get our predictions for the whole dataset.
        X = ctx.matrix.stack_columns(float)
        y = ctx.labels.encoded
        y_hat = pipeline._predict(X).predictions

        # Calculate standard squared error.
        sse = np.sum((y_hat - y) ** 2)
        result['sse'] = sse

        # Get the mean squared error.
        mse = sklearn.metrics.mean_squared_error(y, y_hat)
        result['mse'] = mse

        # Keep n handy. (Used by f_stat, coef_se, coef_pval.)
        n = X.shape[0]

        # Calculate summary F-statistic for beta coefficients.
        p = X.shape[1]
        r2 = sklearn.metrics.r2_score(y, y_hat)
        f_stat = (r2 / p) / ((1 - r2) / (n - p - 1))
        result['f_stat'] = f_stat

        # Calculate standard error for beta coefficients.
        X1 = np.hstack((np.ones((n, 1)), np.matrix(X)))
        inv = np.linalg.inv(X1.T * X1)
        matrix = scipy.linalg.sqrtm(mse * inv)
        coef_se = np.diagonal(matrix)

        # If we got a bunch of complex numbers, then just stop here.
        num_complex = sum(1 for i in coef_se if isinstance(i, complex))
        too_many = len(coef_se) / 4
        if num_complex > too_many:
            return result

        result['coef_se'] = coef_se

        # Calculate t-statistic for beta coefficients.
        est = pipeline.estimator
        a = np.array(est.intercept_ / coef_se[0])
        b = np.array(est.coef_ / coef_se[1:])
        coef_tval = np.append(a, b)
        result['coef_tval'] = coef_tval

        # Calculate p-values for beta coefficients.
        coef_pval = 2 * (1 - scipy.stats.t.cdf(abs(coef_tval), n - 1))
        result['coef_pval'] = coef_pval
    except Exception:
        # For now, just silently ignore it. It usually means that the estimator
        # doesn't have the required "coef_" and "intercept_" attributes.
        pass
    finally:
        return result or None
