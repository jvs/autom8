from collections import namedtuple
import sklearn.metrics


Evaluation = namedtuple('Evaluation', 'preprocessing, train, test')

PredictionSection = namedtuple('PredictionSection',
    'predictions, probabilities, metrics')


def evaluate_pipeline(ctx, pipeline):
    return Evaluation(
        preprocessing = {
            'input': ctx.initial_formulas,
            'output': ctx.matrix.formulas,
            'steps': [s.func.__name__ for s in pipeline.steps],
        },
        train=_evaluate(ctx, pipeline, *ctx.training_data()),
        test=_evaluate(ctx, pipeline, *ctx.testing_data()),
    )


def _evaluate(ctx, pipeline, X, y):
    outputs = pipeline._predict(X)

    if ctx.is_regression:
        metrics = _evaluate_regressor(ctx, y, outputs.predictions)
    else:
        # MAY: Eventually use the probabilities, too.
        predicted = pipeline.label_encoder.transform(outputs.predictions)
        metrics = _evaluate_classifier(ctx, y, predicted)

    return PredictionSection(outputs.predictions, outputs.probabilities, metrics)


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
    funcs1 = [
        sklearn.metrics.f1_score,
        sklearn.metrics.precision_score,
        sklearn.metrics.recall_score,
    ]
    funcs2 = [
        sklearn.metrics.accuracy_score,
    ]
    d1 = _apply_metrics(funcs1, actual, predicted, average='weighted')
    d2 = _apply_metrics(funcs2, actual, predicted)
    d1.update(d2)
    return d1


def _apply_metrics(funcs, *a, **k):
    result = {}
    for f in funcs:
        try:
            result[f.__name__] = f(*a, **k)
        except Exception:
            pass
    return result
