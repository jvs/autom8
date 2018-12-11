import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.neural_network
import sklearn.svm
import sklearn.tree

try:
    import lightgbm
except ImportError:
    lightgbm = None

try:
    import xgboost
except ImportError:
    xgboost = None

from .cleaning import clean_dataset
from .docstrings import render_docstring
from .inference import infer_roles
from . import preprocessors as then
from .context import create_context
from .selector import create_selector


@render_docstring
def fit(*args, **kwargs):
    """Runs autom8 with the provided settings and returns the best pipeline.

    Parameters:
        $common_context_parameters
        $selector_parameters

    Returns:
        autom8.Pipeline: The best pipeline, as indicated by the `selector`
        parameter.
    """

    if 'receiver' in kwargs:
        raise TypeError("fit() got an unexpected keyword argument 'receiver'")

    selector = create_selector(kwargs.pop('selector', None))
    run(*args, **kwargs, receiver=selector)
    return None if selector.best is None else selector.best.pipeline


@render_docstring
def run(*args, **kwargs):
    """Runs autom8 with the provided settings.

    After creating the `RecordingContext` object, autom8 passes the context to
    the receiver's `receive_context` method. The allows the receive to copy any
    information that it may need later, like the context's `test_indices`
    attribute, for example.

    As autom8 creates candidate pipelines, it passes each candidate to the
    receiver's `receive_candidate` method. At that point, the receiver is free
    to do whatever it wants with the candidate.

    Parameters:
        $all_context_parameters
    """

    ctx = create_context(*args, **kwargs)
    ctx.receiver.receive_context(ctx)

    clean_dataset(ctx)
    infer_roles(ctx)
    then.coerce_columns(ctx)
    then.encode_text(ctx)

    with ctx.sandbox():
        then.encode_categories(ctx, method='ordinal', only_strings=True)
        then_fit_estimators(ctx, boosted_trees=True, classical=False)

        then_finish_up_preprocessing(ctx)
        then_fit_estimators(ctx, boosted_trees=True, classical=True)

    with ctx.sandbox():
        then.encode_categories(ctx, method='ordinal', only_strings=True)
        then.multiply_columns(ctx)
        then.add_column_of_ones(ctx)
        then_fit_estimators(ctx, boosted_trees=True, classical=False)

    with ctx.sandbox():
        # This time around, use one-hot encoding and skip the boosted trees.
        then.encode_categories(ctx, method='one-hot', only_strings=False)

        # Try not to engineer more columns than the number of rows in the
        # test dataset.
        max_num_cols = len(ctx.matrix.columns) + len(ctx.test_indices)
        has_room = lambda: len(ctx.matrix.columns) < max_num_cols

        if has_room(): then.multiply_columns(ctx)
        if has_room(): then.square_columns(ctx)
        if has_room(): then.divide_columns(ctx)
        if has_room(): then.binarize_signs(ctx)
        if has_room(): then.logarithm_columns(ctx)
        if has_room(): then.sqrt_columns(ctx)
        if has_room(): then.binarize_fractions(ctx)

        then_finish_up_preprocessing(ctx)
        then_fit_estimators(ctx, boosted_trees=False, classical=True)


def then_finish_up_preprocessing(ctx):
    # TODO: Search for a good scaler.
    then.scale_columns(ctx)
    then.drop_weak_columns(ctx)
    then.add_column_of_ones(ctx)
    then.drop_duplicate_columns(ctx)


def then_fit_estimators(ctx, **flags):
    with ctx.parallel():
        if ctx.is_regression:
            fit_regressors(ctx, **flags)

        if ctx.is_classification:
            fit_classifiers(ctx, **flags)


def fit_regressors(ctx, boosted_trees=True, classical=True):
    random_state = ctx.random_state_kw

    if boosted_trees and lightgbm is not None:
        ctx << lightgbm.LGBMRegressor(n_jobs=n_jobs(ctx), **random_state)

    if boosted_trees and xgboost is not None:
        ctx << xgboost.XGBRegressor(n_jobs=n_jobs(ctx), **random_state)

    # Always fit a linear regression.
    ctx << sklearn.linear_model.LinearRegression()

    # Only run SGD when the dataset is big (and when `classical` is True).
    if classical and is_big(ctx):
        ctx << sklearn.linear_model.SGDRegressor(
            learning_rate='constant',
            eta0=0.00001,
            max_iter=1000,
            tol=1e-3,
            **random_state,
        )

    # Non-SGD estimators struggle to converge when there are too many rows.
    if classical and not is_big(ctx):
        ctx << sklearn.tree.DecisionTreeRegressor(**random_state)

        ctx << sklearn.linear_model.ElasticNetCV(
            l1_ratio=0.0,
            alphas=[0.1, 1.0, 10.0],
            **random_state,
        )

        ctx << sklearn.linear_model.LassoCV(**random_state)
        ctx << sklearn.linear_model.ElasticNetCV(**random_state)
        ctx << sklearn.svm.LinearSVR(**random_state)

        # TODO: Consider only running this on appropriately-sized datasets.
        ctx << sklearn.neural_network.MLPRegressor(
            hidden_layer_sizes=(10, 10),
            **random_state,
        )


def fit_classifiers(ctx, boosted_trees=True, classical=True):
    random_state = ctx.random_state_kw

    if boosted_trees and lightgbm is not None:
        ctx << lightgbm.LGBMClassifier(
            class_weight='balanced', n_jobs=n_jobs(ctx), **random_state,
        )

    if boosted_trees and xgboost is not None:
        ctx << xgboost.XGBClassifier(
            class_weight='balanced', n_jobs=n_jobs(ctx), **random_state,
        )

    # Skip the decision tree when the dataset is large.
    if classical and not is_big(ctx):
        ctx << sklearn.tree.DecisionTreeClassifier(**random_state)

    if classical:
        ctx << sklearn.neighbors.KNeighborsClassifier()
        ctx << sklearn.naive_bayes.BernoulliNB(alpha=0.01)

        ctx << sklearn.linear_model.SGDClassifier(
            class_weight='balanced', max_iter=1000, tol=1e-3, **random_state,
        )

        ctx << sklearn.neural_network.MLPClassifier(
            hidden_layer_sizes=(10, 10), **random_state,
        )


def is_big(ctx):
    # TODO: Implement a smarter heuristic. 90k is a rough heuristic that does
    # not take into account the quality or quantity of features.
    return len(ctx.matrix) >= 90000


def n_jobs(ctx):
    return -1 if ctx.allow_multicore else 1
