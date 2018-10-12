from collections import namedtuple
import numpy as np
import sklearn.metrics
import scipy.sparse

from .pipeline import Pipeline


TrainingReport = namedtuple('TrainingReport', 'pipeline, train, test')


def train(ctx, estimator):
    ctx.require_training_context()
    X, y = ctx.training_data()
    X = scipy.sparse.csr_matrix(X.astype(float))
    estimator.fit(X, y)
    train = _evaluate(ctx, estimator, is_train=True)
    test = _evaluate(ctx, estimator, is_train=False)
    pipeline = Pipeline(list(ctx.preprocessors), estimator, ctx.labels.encoder)
    return TrainingReport(pipeline, train, test)


def _evaluate(ctx, estimator, is_train):
    if is_train:
        X, y = ctx.training_data()
    else:
        X, y = ctx.testing_data()

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
        result = _evaluate_regressor(ctx, y, predictions)
    else:
        result = _evaluate_classifier(ctx, y, predictions)

    result['predictions'] = predictions
    return result


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
    # TODO: Add the confusion matrix and the classification report.
    return {
        f.__name__: f(actual_labels, predicted_labels, average='weighted')
        for f in [
            sklearn.metrics.precision_score,
            sklearn.metrics.recall_score,
            sklearn.metrics.f1_score,
        ]
    }
