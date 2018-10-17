import re
import numpy as np
from .exceptions import typename
from .parsing import parse_number
from .preprocessors import planner, preprocessor


@planner
def clean_dataset(ctx):
    for index, col in enumerate(ctx.matrix.columns):
        if col.dtype == object:
            _clean_column(ctx, index, col)


def _clean_column(ctx, index, col):
    values = col.values
    num_values = len(values)

    num_bools = sum(1 for i in values if isinstance(i, bool))
    num_floats = sum(1 for i in values if isinstance(i, float))
    num_ints = sum(1 for i in values if isinstance(i, int))
    num_none = sum(1 for i in values if i is None)
    num_strings = sum(1 for i in values if isinstance(i, str))
    computed_total = num_bools + num_floats + num_ints + num_none + num_strings

    if computed_total != num_values:
        try:
            found = {typename(i) for i in values if i is not None
                and not isinstance(i, (bool, int, float, str))}
        except Exception:
            found = 'unexpected values'
        msg1 = 'columns to only contain booleans, numbers, and strings'
        msg2 = f'{col.name}: {found}'
        raise expected(msg1, msg2)

    # If we somehow got an array of primitives with dtype == object, then just
    # coerce it to the appropriate type.
    casts = {bool: num_bools, float: num_floats, int: num_ints}
    zeros = {bool: False, float: 0.0, int: 0}
    for typ, num in casts.items():
        if num == num_values:
            _coerce_column(ctx, index, typ, zeros[typ])
            return

    # If we have all None values, then drop this column.
    if num_none == num_values:
        observer.warn(f'Found column of all None values: {col.name}')
        _drop_weak_columns(ctx, [index])
        return

    # If we have some strings, try converting them all to numbers.
    if num_strings > 0:
        try:
            _coerce_strings_to_numbers(ctx, index, coerce_all=True)
        except Exception:
            # Well, that didn't work. Just keep trucking along, then.
            pass
        else:
            # We were able to convert the strings to numbers, so let's recur.
            _clean_column(ctx, index, col)
            return

    # If we have all strings, then we're clean. Just leave this column alone.
    if num_strings == num_values:
        assert all(isinstance(i, str) for i in values)
        return

    # If we have all strings, but some are None, then use the empty string.
    # (And don't warn the user about the None values in this case.)
    if num_strings + num_none == num_values:
        _replace_nones(ctx, index, '')
        return

    # If we have some ints and some floats, then coerce the ints to floats
    # and recur.
    if num_ints > 0 and num_floats > 0:
        _coerce_ints_to_floats(ctx, index)
        _clean_column(ctx, index, col)
        return

    # If all of the strings are the empty string, then replace the empty
    # strings with None and recur.
    if num_strings > 0 and all(i == '' for i in values if isinstance(i, str)):
        _replace_empty_strings(ctx, index, None)
        _clean_column(ctx, index, col)
        return

    # If we have all primitive values, but some are None, then replace None
    # values with the appropriate zero value. Add a boolean column that records
    # which values were missing.
    for typ, num in casts.items():
        if num + num_none == num_values:
            observer.warn(
                f'Found column with {num_none} missing value'
                f'{"" if num_none == 1 else "s"}: {column.name}'
            )
            _flag_missing_values(ctx, index, zeros[typ])
            return

    # Now we know we have some strings, some numbers, and maybe some Nones.
    # We also know that the non-empty strings cannot all be coerced into
    # numbers (but maybe some can be coerced).

    # First, coerce as many of the strings as we can into numbers. Then split
    # this column into two columns: one for numbers and another for strings.
    _coerce_strings_to_numbers(ctx, index, coerce_all=False)


@preprocessor
def _coerce_column(ctx, index, typ, replacement):
    assert typ != str
    col = ctx.matrix.columns[index]
    def conv(x):
        if isinstance(x, typ):
            return x
        try:
            return typ(x)
        except Exception:
            return replacement
    new_values = [conv(x) for x in col.values]
    col.values = np.array(new_values, dtype=typ)


@preprocessor
def _coerce_strings_to_numbers(ctx, index, coerce_all=True):
    col = ctx.matrix.columns[index]
    new_values = [_coerce_string_to_number(i, is_required=coerce_all)
        if isinstance(i, str) else i
        for i in col.values]
    col.values = _create_array(new_values)


def _coerce_string_to_number(obj, is_required=True):
    assert isinstance(obj, str)

    # Remove leading and trailing whitespace.
    obj = obj.strip()

    # Replace the empty string with None.
    if obj == '':
        return None

    try:
        return parse_number(obj)
    except Exception:
        pass

    # Pull out the number part and try parsing it.
    m = re.match(r'^\$*(\-?[0-9\.]+)\%*$', obj)
    if m:
        return parse_number(m.group(1))
    elif is_required:
        raise Exception(f'Failed to coerce string to number: {obj}')
    else:
        return obj


@preprocessor
def _replace_nones(ctx, index, replacement):
    col = ctx.matrix.columns[index]
    new_values = [replacement if i is None else i for i in col.values]
    col.values = _create_array(new_values)


@preprocessor
def _coerce_ints_to_floats(ctx, index):
    col = ctx.matrix.columns[index]
    new_values = [float(i) if isinstance(i, int) else i for i in col.values]
    col.values = _create_array(new_values)


@preprocessor
def _replace_empty_strings(ctx, index, replacement):
    col = ctx.matrix.columns[index]
    new_values = [
        replacement if isinstance(i, str) and i == '' else i
        for i in col.values
    ]
    col.values = _create_array(new_values)


@preprocessor
def _flag_missing_values(ctx, index, replacement):
    col = ctx.matrix.columns[index]
    new_values = [replacement if i is None else i for i in col.values]
    ctx.matrix.drop_columns_by_index([index])
    ctx.matrix.append_column(col.copy_with(_create_array(new_values)))
    ctx.matrix.append_column(
        values=col.values != None,
        name=f'PRESENT ({col.name})',
        role='encoded',
        is_original=True,
    )


@preprocessor
def _bipartition_strings(ctx, index):
    # For None values, just put a 0 in the number-column and an empty string
    # in the string-column.
    col = ctx.matrix.columns[index]
    ctx.matrix.drop_columns_by_index([index])

    numbers = [i if isinstance(i, (int, float)) else 0 for i in col.values]
    strings = [i if isinstance(i, str) else '' for i in col.values]

    ctx.matrix.append_column(
        values=_create_array(numbers),
        name=f'NUMBERS ({col.name})',
        role=None,
        is_original=True,
    )

    ctx.matrix.append_column(
        values=_create_array(strings),
        name=f'STRINGS ({col.name})',
        role=None,
        is_original=True,
    )


def _create_array(values):
    has_str = any(isinstance(i, str) for i in values)
    return np.array(values, dtype=object if has_str else None)
