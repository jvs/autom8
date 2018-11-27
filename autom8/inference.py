from .preprocessors import planner, preprocessor


@planner
def infer_roles(ctx):
    """Infers the role of each column.

    autom8 uses a handful of roles: `categorical`, `encoded`, `numerical`, and
    `textual`.

    autom8 infers that a column contains categorical data when the column
    contains fewer than 50 unique values, and when less than 25% of the values
    are unique.

    autom8 infers that a column contains encoded data when the column only
    contains boolean values.

    Otherwise, if a column contains strings, then autom8 infers that it
    contains textual data. And if it contains numbers, then autom8 infers that
    it contains numerical data.
    """

    roles = [_infer_role(col, ctx.receiver) for col in ctx.matrix.columns]
    _set_roles(ctx, roles)


@preprocessor
def _set_roles(ctx, roles):
    """Records the type of each column."""
    for col, role in zip(ctx.matrix.columns, roles):
        col.role = role


def _infer_role(col, receiver):
    inferred = _infer_role_from_values(col.values)
    return _merge_roles(col, inferred, receiver)


def _infer_role_from_values(values):
    # If it's a column of booleans, then just say that it's already encoded.
    if values.dtype == bool:
        return 'encoded'

    # And if it's a column of floats, then say that it's numerical.
    if values.dtype == float:
        return 'numerical'

    num_unique = len(set(values))
    ratio = num_unique / len(values)
    is_categorical = ratio <= 0.25 and num_unique < 50
    is_all_strings = all(isinstance(e, str) for e in values)

    if is_categorical:
        return 'categorical'
    elif is_all_strings:
        return 'textual'
    else:
        return 'numerical'


def _merge_roles(col, new_role, receiver):
    # Let's use short names in here.
    old, new = col.role, new_role

    # Complain if one is "numerical" and the other is "textual".
    if (old, new) in {('numerical', 'textual'), ('textual', 'numerical')}:
        receiver.warn(
            f'Found a {new} column declared as a {old} column: {col.name}'
        )
        return new

    # If either one is None, return the other one.
    if new is None or old is None:
        return new or old

    # If the inferencer thinks its encoded, and the user says its categorical,
    # then we can stick with "encoded".
    if new == 'encoded' and old == 'categorical':
        return 'encoded'

    # In any other situation, just trust whatever the old value says.
    return old
