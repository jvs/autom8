from collections import namedtuple
import numpy as np
import sklearn.metrics
import scipy.sparse


Evaluation = namedtuple('Evaluation', 'train, test')
EvaluationSection = namedtuple('EvaluationSection', 'predictions, metrics')


def evaluate_pipeline(ctx, pipeline):
    estimator = pipeline.estimator
    train = _evaluate_section(ctx, estimator, *ctx.training_data())
    test = _evaluate_section(ctx, estimator, *ctx.testing_data())
    return Evaluation(train, test)


def _evaluate_section(ctx, estimator, X, y):
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


def _evaluate_regressor(ctx, actual_labels, predicted_labels):
    return {f.__name__: f(actual_labels, predicted_labels) for f in [
        sklearn.metrics.explained_variance_score,
        sklearn.metrics.mean_absolute_error,
        sklearn.metrics.mean_squared_error,
        sklearn.metrics.mean_squared_log_error,
        sklearn.metrics.median_absolute_error,
        sklearn.metrics.r2_score,
    ]}


def _evaluate_classifier(ctx, actual_labels, predicted_labels):
    return {
        f.__name__: f(actual_labels, predicted_labels, average='weighted')
        for f in [
            sklearn.metrics.precision_score,
            sklearn.metrics.recall_score,
            sklearn.metrics.f1_score,
        ]
    }
