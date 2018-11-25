import numpy as np
import pandas as pd


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
    return OneHotEncoder() if method == 'one-hot' else OrdinalEncoder()


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
            df = _create_failed_encoding(encoder, found)
            ctx.receiver.warn('Failed to encode categorical data.')

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    if isinstance(encoder, OneHotEncoder):
        formulas = _one_hot_encoded_formulas(encoder, found)
    else:
        formulas = [['encode', col] for col in found.columns]

    for i in range(df.shape[1]):
        ctx.matrix.append_column(
            values=df.iloc[:, i].values,
            formula=formulas[i],
            role='encoded',
        )


def _one_hot_encoded_formulas(encoder, found):
    result = []
    for series, column in zip(encoder.mapping, found.columns):
        for value in series.index:
            result.append([f'equals[{value}]', column])
    return result


def _create_failed_encoding(encoder, found):
    num_rows = len(found)
    if isinstance(encoder, OneHotEncoder):
        num_cols = sum(len(s) for s in encoder.mapping)
    else:
        num_cols = len(found.columns)
    return pd.DataFrame(0, index=range(num_rows), columns=range(num_cols))


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


class OneHotEncoder:
    def __init__(self):
        self._encoder = None

    def __repr__(self):
        return 'OneHotEncoder'

    @property
    def mapping(self):
        return self._encoder.mapping

    def fit_transform(self, X):
        X = pd.DataFrame(X)
        self._encoder = OrdinalEncoder()
        X = self._encoder.fit_transform(X)
        return self._elaborate(X)

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = self._encoder.transform(X)
        return self._elaborate(X)

    def _elaborate(self, X):
        cols = []
        for i, series in enumerate(self.mapping):
            for value in series:
                cols.append((X[i] == value).rename(len(cols)).astype(int))
        return pd.concat(cols, axis=1)
