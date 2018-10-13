from collections import namedtuple
import numpy as np
import sklearn.metrics
import scipy.sparse


Evaluation = namedtuple('Evaluation', 'general, train, test')
EvaluationSection = namedtuple('EvaluationSection', 'predictions, metrics')


def evaluate_pipeline(ctx, pipeline):
    estimator = pipeline.estimator
    general = _general_stats(ctx, pipeline, estimator)
    train = _evaluate(ctx, estimator, *ctx.training_data())
    test = _evaluate(ctx, estimator, *ctx.testing_data())
    return Evaluation(general, train, test)


def _general_stats(ctx, pipeline, estimator):
    result = {
        'num_steps': len(pipeline.steps),
        'num_columns': len(ctx.matrix.columns),
    }
    for attr in dir(estimator):
        if attr.endswith('_') and not attr.endswith('__'):
            result[attr] = getattr(estimator, attr)
    return result


def _evaluate(ctx, estimator, X, y):
    X = X.astype(float)
    predictions = []

    # TODO: Calculate an appropriate window size.
    for i in range(0, len(X), 1000):
        window = X[i : i + 1000]
        matrix = scipy.sparse.csr_matrix(window)
        encoded = estimator.predict(matrix)
        decoded = ctx.labels.decode(encoded)
        predictions.extend(decoded)

    # Convert the predictions list to a numpy array.
    predictions = np.array(predictions)

    if ctx.is_regression:
        metrics = _evaluate_regressor(ctx, y, predictions)
    else:
        metrics = _evaluate_classifier(ctx, y, predictions)

    return EvaluationSection(predictions, metrics)


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


def _evaluate_classifier(ctx, actual, predicted):
    funcs = [
        sklearn.metrics.precision_score,
        sklearn.metrics.recall_score,
        sklearn.metrics.f1_score,
    ]
    return _apply_metrics(funcs, actual, predicted, average='weighted')


def _apply_metrics(funcs, *a, **k):
    result = {}
    for f in funcs:
        try:
            result[f.__name__] = f(*a, **k)
        except Exception:
            pass
    return result
