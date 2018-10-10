from collections import namedtuple
import numpy as np
import sklearn.metrics
import scipy.sparse


TrainingReport = namedtuple('TrainingReport', 'estimator, train, test')


def train(ctx, estimator):
    ctx.require_training_context()
    X, y = ctx.training_data()
    X = scipy.sparse.csr_matrix(X.astype(float))
    estimator.fit(X, y)
    train = _evaluate(ctx, estimator, is_train=True)
    test = _evaluate(ctx, estimator, is_train=False)
    return TrainingReport(estimator, train, test)


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
        return _evaluate_regressor(ctx, y, predictions)
    else:
        return _evaluate_classifier(ctx, y, predictions)


def _evaluate_regressor(ctx, actual_labels, predicted_labels):
    result = {'predictions': predicted_labels}
    funcs = [
        sklearn.metrics.explained_variance_score,
        sklearn.metrics.mean_absolute_error,
        sklearn.metrics.mean_squared_error,
        sklearn.metrics.mean_squared_log_error,
        sklearn.metrics.median_absolute_error,
        sklearn.metrics.r2_score,
    ]
    for f in funcs:
        result[f.__name__] = f(actual_labels, predicted_labels)
    return result


def _evaluate_classifier(ctx, actual_labels, predicted_labels):
    result = {'predictions': predicted_labels}
    funcs = [
        sklearn.metrics.precision_score,
        sklearn.metrics.recall_score,
        sklearn.metrics.f1_score,
    ]
    for f in funcs:
        result[f.__name__] = f(actual_labels, predicted_labels, average='weighted')

    # TODO: Add the confusion matrix and the classification report.
    return result
