import pandas as pd
import statsmodels.api as sm

from .cleaning import clean_dataset
from .inference import infer_roles
from . import preprocessors as then
from .context import create_context
from .exceptions import expected


def summarize(*args, **kwargs):
    if 'problem_type' not in kwargs:
        kwargs['problem_type'] = 'regression'

    if kwargs['problem_type'] != 'regression':
        raise expected('problem_type to be \'regression\'', problem_type)

    # TODO: Factor out some of the common logic with the fit function.
    ctx = create_context(*args, **kwargs)
    clean_dataset(ctx)
    infer_roles(ctx)
    then.encode_text(ctx)
    then.encode_categories(ctx, method='ordinal', only_strings=True)

    X1 = _get_X(ctx)
    then.add_column_of_ones(ctx)
    X2 = _get_X(ctx)
    y = ctx.labels.encoded

    classes = [sm.OLS, sm.WLS, sm.GLS, sm.RecursiveLS]
    res1 = [_summarize(c, y, X1) for c in classes]
    res2 = [_summarize(c, y, X2) for c in classes]
    return res1 + res2


def _summarize(cls, y, X):
    model = cls(y, X)
    results = model.fit()
    return results.summary()


def _get_X(ctx):
    return pd.DataFrame(
        data=ctx.matrix.stack_columns(as_type=float),
        columns=[c.name for c in ctx.matrix.columns],
    )
