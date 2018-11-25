import numpy as np
import pandas as pd
from category_encoders.one_hot import OneHotEncoder


def select_indices(ctx, only_strings=False):
    def should_be_encoded(col):
        if col.role != 'categorical':
            return False
        elif only_strings:
            return any(isinstance(i, str) for i in col.values)
        else:
            return True

    return ctx.matrix.column_indices_where(should_be_encoded)


def create_encoder(ctx, method, indices):
    if method == 'one-hot':
        cols = list(range(len(indices)))
        return OneHotEncoder(cols=cols, use_cat_names=True)
    else:
        return OrdinalEncoder()


def encode(ctx, encoder, indices):
    found = ctx.matrix.select_columns(indices)
    array = found.stack_columns()
    ctx.matrix.drop_columns_by_index(indices)

    if ctx.is_fitting:
        df = encoder.fit_transform(array)
    else:
        try:
            df = encoder.transform(array)
        except Exception:
            df = _create_failed_encoding(found, encoder)
            ctx.receiver.warn('Failed to encode categorical data.')

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    if isinstance(encoder, OneHotEncoder):
        formulas = _one_hot_encoded_formulas(df.columns, found)
    else:
        formulas = [['encode', col] for col in found.columns]

    for i in range(df.shape[1]):
        ctx.matrix.append_column(
            values=df.iloc[:, i].values,
            formula=formulas[i],
            role='encoded',
        )


def _one_hot_encoded_formulas(df_columns, found):
    result = []
    for i, dfcol in enumerate(df_columns):
        try:
            index, val = dfcol.split('_')
            col = found.columns[int(index)]
            result.append([f'equals[{val}]', col])
        except Exception:
            result.append(['one-hot-encode'] + found.columns)
    return result


def _create_failed_encoding(matrix, encoder):
    if isinstance(encoder, OneHotEncoder):
        cols = _create_one_hot_column_names(encoder)
    else:
        cols = matrix.column_names

    return pd.DataFrame(0, index=np.arange(len(matrix)), columns=cols)


def _create_one_hot_column_names(encoder):
    columns = []
    for m in encoder.category_mapping:
        prefix = m['col']
        for oldval, newval in m['mapping']:
            columns.append(f'{prefix}_{oldval}')
        columns.append(f'{prefix}_-1')
    return columns


class OrdinalEncoder:
    def __init__(self):
        self.mapping = None

    def __repr__(self):
        return 'OrdinalEncoder'

    def fit_transform(self, X):
        X = pd.DataFrame(X)
        self.mapping = [_ordinally_map_series(X[i]) for i in X.columns]
        return self.transform(X)

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        for i, series in enumerate(self.mapping):
            X[i] = X[i].map(series)
            X[i].fillna(0, inplace=True)
        return X


def _ordinally_map_series(series):
    index = [x for x in pd.unique(series) if x is not None]
    data = list(range(1, len(index) + 1))
    return pd.Series(data=data, index=index)
