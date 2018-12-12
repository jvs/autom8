import re
import numpy as np
from .exceptions import expected, typename
from .formats import parse_number
from .matrix import create_array
from .preprocessors import planner, preprocessor, _drop_weak_columns


int_type = (int, np.int64)


@planner
def clean_dataset(ctx):
    """Cleans the context's dataset, and records the steps for later playback.

    This function peforms a handful of steps:

    - Drops columns that contain unexpected values. A value is unexpected if is
      is not a boolean, a number, a string, or None.
    - Drops columns that only contain None values.
    - Coerces columns to have numeric types whenever possible.
    - In columns of strings, replaces None values with the empty string.
    - In a column of numbers, if any numbers are missing, replace the missing
      number with zero, and add a second column of booleans that indicates which
      values were missing.
    - Replace a column of mixed strings and numbers with two columns: one for
      the string values and another for the number values.

    After running these steps, each column in the dataset should have a uniform
    type for all the values in that column. For example, if a column has one
    boolean value, then all the values in the column will be booleans.

    Parameters:
        ctx (RecordingContext): The current context.
    """

    for col in ctx.matrix.columns:
        _clean_column(ctx, col)


def _clean_column(ctx, col):
    # If numpy inferred a real dtype for this column, then it's clean enough.
    # TODO: Consider bipartitioning columns with nan values.
    if col.dtype != object:
        return

    index = ctx.matrix.columns.index(col)
    values = col.values
    num_values = len(values)

    num_bools = sum(1 for i in values if isinstance(i, bool))
    num_floats = sum(1 for i in values if isinstance(i, float))

    num_ints = sum(1 for i in values
        if isinstance(i, int_type)
        and not isinstance(i, bool))

    num_none = sum(1 for i in values if i is None)
    num_strings = sum(1 for i in values if isinstance(i, str))
    computed_total = num_bools + num_floats + num_ints + num_none + num_strings

    if computed_total != num_values:
        try:
            found = {typename(i) for i in values if i is not None
                and not isinstance(i, (bool, int, float, str, np.int64))}
        except Exception:
            found = 'unexpected values'
        ctx.receiver.warn(f'Dropping column "{col.name}". A column must only'
            f' contain booleans, numbers, and strings. Received: {found}.')
        _drop_weak_columns(ctx, [index])
        return

    # Record the number of values for each type.
    counts = {bool: num_bools, float: num_floats, int: num_ints}

    # If we somehow got an array of primitives with dtype == object, then just
    # coerce it to the appropriate type.
    for typ, num in counts.items():
        if num == num_values:
            _coerce_column(ctx, index, typ)
            return

    # If we have all None values, then drop this column.
    if num_none == num_values:
        ctx.receiver.warn(f'Dropping column of all None values: {col.name}')
        _drop_weak_columns(ctx, [index])
        return

    # If we have some strings, see if we can convert them all to numbers.
    if num_strings > 0 and _can_coerce_all_strings_to_numbers(values):
        _coerce_strings_to_numbers(ctx, index)
        _clean_column(ctx, col)
        return

    # If we have all strings, then we're clean. Just leave this column alone.
    if num_strings == num_values:
        assert all(isinstance(i, str) for i in values)
        return

    # If we have all strings, but some are None, then use the empty string.
    # (And don't warn the user about the None values in this case.)
    if num_strings + num_none == num_values:
        _replace_none_values(ctx, index, '')
        return

    # If we have some ints and some floats, then coerce the ints to floats
    # and recur.
    if num_ints > 0 and num_floats > 0:
        _coerce_ints_to_floats(ctx, index)
        _clean_column(ctx, col)
        return

    # If any strings are blank, then replace them with the empty string and
    # recur.
    if any(i.isspace() for i in values if isinstance(i, str)):
        _replace_blank_strings(ctx, index)
        _clean_column(ctx, col)
        return

    # If all of the strings are the empty string, then replace the empty
    # strings with None and recur.
    if num_strings > 0 and all(i == '' for i in values if isinstance(i, str)):
        _replace_empty_strings(ctx, index, None)
        _clean_column(ctx, col)
        return

    # If we have all primitive values, but some are None, then replace None
    # values with the appropriate zero value. Add a boolean column that records
    # which values were missing.
    for typ, num in counts.items():
        if num + num_none == num_values:
            ctx.receiver.warn(
                f'Column {repr(col.name)} has {num_none} missing'
                f' value{"" if num_none == 1 else "s"}.'
            )
            _flag_missing_values(ctx, index, typ(0))
            return

    # Now we know we have some strings, some numbers, and maybe some Nones.
    # We also know that the nonempty strings cannot all be coerced into
    # numbers (but maybe some can be coerced).

    # First, coerce as many of the strings as we can into numbers. Then split
    # this column into two columns: one for numbers and another for strings.
    _coerce_strings_to_numbers(ctx, index)
    _bipartition_strings(ctx, index)


@preprocessor
def _coerce_column(ctx, index, to_type):
    col = ctx.matrix.columns[index]
    col.coerce(to_type)


_string_to_number_regex = re.compile(r'^\$*(\-?[0-9,\.]+)\%*$')


def _can_coerce_all_strings_to_numbers(values):
    return all(_string_to_number_regex.match(i)
        for i in values if isinstance(i, str))


@preprocessor
def _coerce_strings_to_numbers(ctx, index):
    col = ctx.matrix.columns[index]
    new_values = [_coerce_string_to_number(i)
        if isinstance(i, str) else i
        for i in col.values]
    col.values = create_array(new_values)


def _coerce_string_to_number(obj):
    assert isinstance(obj, str)

    # Remove leading and trailing whitespace.
    obj = obj.strip()

    # Replace the empty string with None.
    if obj == '':
        return None

    try:
        return parse_number(obj)
    except Exception:
        # Pull out the number part and try parsing it.
        m = _string_to_number_regex.match(obj)
        return parse_number(m.group(1)) if m else obj


@preprocessor
def _replace_none_values(ctx, index, replacement):
    col = ctx.matrix.columns[index]
    new_values = [replacement if i is None else i for i in col.values]
    col.values = create_array(new_values)


@preprocessor
def _coerce_ints_to_floats(ctx, index):
    col = ctx.matrix.columns[index]
    new_values = [float(i) if isinstance(i, int_type) else i for i in col.values]
    col.values = create_array(new_values)


@preprocessor
def _replace_blank_strings(ctx, index):
    col = ctx.matrix.columns[index]
    new_values = [
        '' if isinstance(i, str) and i.isspace() else i for i in col.values
    ]
    col.values = create_array(new_values)


@preprocessor
def _replace_empty_strings(ctx, index, replacement):
    col = ctx.matrix.columns[index]
    new_values = [
        replacement if isinstance(i, str) and i == '' else i
        for i in col.values
    ]
    col.values = create_array(new_values)


@preprocessor
def _flag_missing_values(ctx, index, replacement):
    col = ctx.matrix.columns[index]
    new_values = [replacement if i is None else i for i in col.values]
    ctx.matrix.drop_columns_by_index([index])
    ctx.matrix.columns.append(col.copy_with(create_array(new_values)))
    ctx.matrix.append_column(
        values=col.values != None,
        formula=['is-defined', col],
        role='encoded',
        is_original=True,
    )


@preprocessor
def _bipartition_strings(ctx, index):
    # For None values, just put a 0 in the number-column and an empty string
    # in the string-column.
    col = ctx.matrix.columns[index]
    ctx.matrix.drop_columns_by_index([index])

    is_num = lambda x: x is not None and not isinstance(x, str)
    numbers = [i if is_num(i) else 0 for i in col.values]
    strings = [i if isinstance(i, str) else '' for i in col.values]

    ctx.matrix.append_column(
        values=create_array(numbers),
        formula=['number', col],
        role=None,
        is_original=True,
    )

    ctx.matrix.append_column(
        values=create_array(strings),
        formula=['string', col],
        role=None,
        is_original=True,
    )
