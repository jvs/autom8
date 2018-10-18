import numpy as np
import pandas as pd
from category_encoders import OrdinalEncoder
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
    cols = list(range(len(indices)))
    if method == 'one-hot':
        return OneHotEncoder(cols=cols, use_cat_names=True)
    else:
        return OrdinalEncoder(cols=cols)


def encode(ctx, encoder, indices):
    found = ctx.matrix.select_columns(indices)
    array = found.stack_columns()
    ctx.matrix.drop_columns_by_index(indices)

    if ctx.is_training:
        df = encoder.fit_transform(array)
    else:
        try:
            df = encoder.transform(array)
        except Exception:
            df = _create_failed_encoding(found, encoder)
            ctx.receiver.warn('Failed to encode categorical data.')

    if isinstance(encoder, OneHotEncoder):
        column_names = _rename_one_hot_encoded_columns(df.columns, found)
    else:
        column_names = [f'ENCODED {col.name}' for col in found.columns]

    for i in range(df.shape[1]):
        ctx.matrix.append_column(
            values=df.iloc[:, i].values,
            name=column_names[i],
            role='encoded',
        )


def _rename_one_hot_encoded_columns(df_columns, found):
    # TODO: Replace the last "= -1" suffix with "IS UNKNOWN" or something.
    result = []
    visited = set()
    for i, col in enumerate(df_columns):
        try:
            orig, val = col.split('_')
            orig = found.columns[int(orig)].name
        except Exception:
            orig, val = 'X', i

        # Make sure we don't have any duplicate names.
        # (It can happen when -1 is one of the values.)
        name = f'{orig} = {val}'
        count = 1
        while name in visited:
            count += 1
            name = f'{orig} = {val} [{count}]'

        visited.add(name)
        result.append(name)
    return result


def _create_failed_encoding(matrix, encoder):
    if isinstance(encoder, OneHotEncoder):
        cols = _create_one_hot_column_names(encoder)
    else:
        cols = [i.name for i in matrix.columns]

    return pd.DataFrame(0, index=np.arange(len(matrix)), columns=cols)


def _create_one_hot_column_names(encoder):
    columns = []
    for m in encoder.category_mapping:
        prefix = m['col']
        for oldval, newval in m['mapping']:
            columns.append(f'{prefix}_{oldval}')
        columns.append(f'{prefix}_-1')
    return columns
