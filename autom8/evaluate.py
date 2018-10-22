from collections import namedtuple
import numpy as np
import sklearn.metrics


Evaluation = namedtuple('Evaluation', 'context, preprocessing, train, test')
ContextSection = namedtuple('Section', 'problem_type, test_indices, classes')
PreprocessingSection = namedtuple('PreprocessingSection', 'input, output, steps')

EvaluationSection = namedtuple('EvaluationSection',
    'predictions, probabilities, metrics')


def evaluate_pipeline(ctx, pipeline):
    classes = pipeline.label_encoder.classes_ if ctx.is_classification else []
    return Evaluation(
        context=ContextSection(ctx.problem_type, ctx.test_indices, classes),
        preprocessing=PreprocessingSection(
            input=ctx.initial_formulas,
            output=ctx.matrix.formulas,
            steps=[s.func.__name__ for s in pipeline.steps],
        ),
        train=_evaluate_predictions(ctx, pipeline, *ctx.training_data()),
        test=_evaluate_predictions(ctx, pipeline, *ctx.testing_data()),
    )


def _evaluate_predictions(ctx, pipeline, X, y):
    outputs = pipeline._predict(X)

    if ctx.is_regression:
        metrics = _evaluate_regressor(ctx, y, outputs.predictions)
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
