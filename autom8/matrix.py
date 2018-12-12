import re
import numpy as np

from .docstrings import render_docstring
from .exceptions import expected, typename
from .formats import (
    drop_empty_rows,
    excel_column_index,
    excel_column_name,
    parse_number,
)
from .receiver import Receiver


@render_docstring
def create_matrix(dataset, column_names=None, column_roles=None, receiver=None):
    """Returns a new Matrix object from the provided dataset.

    Parameters:
        $dataset_parameters
        $receiver_parameter
    """

    if receiver is None:
        receiver = Receiver()

    matrix = _create_matrix(dataset, column_names, column_roles, receiver)

    # Warn the user if the column names are not unique.
    names = matrix.column_names
    if len(names) != len(set(names)):
        receiver.warn(f'Column names are not unique in {repr(names)}')

    return matrix


def _create_matrix(dataset, names, roles, receiver):
    if not isinstance(dataset, (list, tuple, np.ndarray, Matrix)):
        raise expected('list, tuple, numpy array, or Matrix', typename(dataset))

    if isinstance(dataset, Matrix):
        return _copy_and_update_matrix(dataset, names, roles)

    # Drop empty rows.
    rows = drop_empty_rows(dataset)
    num_dropped = len(dataset) - len(rows)

    # Warn the users if we dropped some rows.
    if num_dropped:
        suffix = '' if num_dropped == 1 else 's'
        receiver.warn(f'Dropped {num_dropped} empty row{suffix} from dataset.')

    # Figure out how many columns we need.
    mincols = min(len(row) for row in rows) if rows else 0
    maxcols = max(len(row) for row in rows) if rows else 0

    # Warn the user if we're dropping any extra columns.
    if mincols < maxcols:
        num_extra = maxcols - mincols
        suffix1 = '' if num_extra == 1 else 's'
        suffix2 = '' if mincols == 1 else 's'
        receiver.warn(
            f'Dropped {num_extra} extra column{suffix1} from dataset.'
            f' Keeping first {mincols} column{suffix2}.'
            ' To avoid this behavior, ensure that each row in the dataset has'
            ' the same number of columns.'
        )

    def make(index):
        values = create_array([row[index] for row in rows])
        formula = excel_column_name(index)
        return Column(values=values, formula=formula, role=None, is_original=True)

    matrix = Matrix([make(i) for i in range(mincols)])
    _name_columns(matrix, names)
    _update_roles(matrix, roles)
    return matrix


class Matrix:
    """Represents a matrix of features.

    Attributes:
        columns (list[Column]): A list of the matrix's columns.
    """

    def __init__(self, columns):
        assert isinstance(columns, list)
        assert all(isinstance(i, Column) for i in columns)
        self.columns = columns

    @property
    def column_names(self):
        return [col.name for col in self.columns]

    @property
    def formulas(self):
        return [col.formula for col in self.columns]

    def __len__(self):
        return len(self.columns[0]) if self.columns else 0

    def copy(self):
        return Matrix([i.copy() for i in self.columns])

    def tolist(self):
        cols = [c.values.astype(object) for c in self.columns]
        return np.column_stack(cols).tolist()

    def stack_columns(self):
        return np.column_stack([col.values for col in self.columns])

    def to_array(self):
        if len(self.columns) == 1:
            return self.columns[0].values
        else:
            raise expected('matrix with only one column', len(self.columns))

    def append_column(self, values, formula, role, is_original=False):
        if not hasattr(values, 'shape'):
            raise expected('numpy array', type(values))

        if len(values.shape) != 1:
            raise expected('array of values', values.shape)

        col = Column(values, formula=formula, role=role, is_original=is_original)
        self.columns.append(col)

    def drop_columns_by_index(self, indices):
        if isinstance(indices, int):
            indices = [indices]

        self.columns = [col for i, col in enumerate(self.columns) if i not in indices]

    def select_rows(self, indices):
        return Matrix([col.select_rows(indices) for col in self.columns])

    def exclude_rows(self, indices):
        return Matrix([col.exclude_rows(indices) for col in self.columns])

    def select_columns(self, indices):
        return Matrix([self.columns[i].copy() for i in indices])

    def exclude_columns(self, indices):
        return Matrix([col.copy()
            for i, col in enumerate(self.columns)
                if i not in indices])

    def select_columns_by_name(self, names):
        # If the names match exactly, then return the matrix as-is.
        if self.column_names == names:
            return self

        # If this matrix has the requested columns, then make a new matrix with them.
        if set(self.column_names).issuperset(names):
            lookup = {col.name: col for col in self.columns}
            return Matrix([lookup[n].copy() for n in names])

        raise expected(f'column names to be in {self.column_names}', names)

    def column_indices_where(self, predicate):
        return [i for i, col in enumerate(self.columns) if predicate(col)]

    def coerce(self, to_type):
        for col in self.columns:
            col.coerce(to_type)

    def _float_array(self):
        """Warning: This method mutates the matrix object."""
        self.coerce(float)
        return self.stack_columns()


class Column:
    def __init__(self, values, formula, role, is_original):
        assert len(values.shape) == 1
        assert isinstance(formula, (str, list))
        assert role is None or isinstance(role, str)
        assert isinstance(is_original, bool)
        self.values = values

        if isinstance(formula, str):
            self.formula = formula
        else:
            self.formula = [getattr(col, 'formula', col) for col in formula]

        self._role = None
        self.role = role
        self.is_original = is_original

    def __len__(self):
        return len(self.values)

    @property
    def name(self):
        if isinstance(self.formula, str):
            return self.formula
        else:
            # TODO: Pretty-print the formula. Maybe put "=" in front.
            return repr(self.formula)

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def is_numerical(self):
        return self.role == 'numerical'

    @property
    def role(self):
        return self._role

    @role.setter
    def role(self, role):
        valid_roles = {None, 'categorical', 'encoded', 'numerical', 'textual'}

        if role not in valid_roles:
            raise expected(f'role in {valid_roles}', repr(role))

        self._role = role

    def copy(self):
        values = np.copy(self.values)
        return self.copy_with(values)

    def copy_with(self, values, role='__copy__'):
        assert len(values.shape) == 1

        # Use a magic sentinel instead of None, since None is a valid role.
        if role == '__copy__':
            role = self.role

        return Column(
            values=values,
            formula=self.formula,
            role=role,
            is_original=self.is_original,
        )

    def coerce(self, to_type):
        expected = (bool, float, int, object, str)
        if to_type not in expected:
            raise TypeError(f'coerce() argument must be one of: {expected}')

        if to_type == bool:
            self._coerce(bool, False, bool)

        elif to_type in [float, int]:
            self.coerce_values_to_numbers(to_type)

        elif to_type == object:
            self.values = self.values.astype(object, copy=False)

        elif to_type == str:
            self.coerce_values_to_strings()

        else:
            assert False

    def coerce_values_to_numbers(self, to_type=float):
        def conv(x):
            try:
                return to_type(parse_number(x))
            except Exception:
                return to_type(x)

        self._coerce(to_type, to_type(0), conv)

    def coerce_values_to_strings(self):
        def conv(x):
            try:
                return str(x)
            except Exception:
                return ''

        if not all(isinstance(x, str) for x in self.values):
            self.values = create_array([conv(x) for x in self.values])

    def _coerce(self, to_type, default, coerce_func):
        try:
            self.values = self.values.astype(to_type, copy=False)
            return
        except Exception:
            pass

        def conv(x):
            if isinstance(x, to_type):
                return x
            try:
                return coerce_func(x)
            except Exception:
                return default

        self.values = np.array([conv(x) for x in self.values], dtype=to_type)

    def select_rows(self, indices):
        return self.copy_with(self.values[indices])

    def exclude_rows(self, indices):
        return self.copy_with(np.delete(self.values, indices))

    def root_columns(self):
        roots = set()
        stack = [self.formula]
        while stack:
            f = stack.pop()
            if isinstance(f, str):
                roots.add(f)
            else:
                assert isinstance(f, (list, tuple))
                assert isinstance(f[0], str)
                stack.extend(f[1:])
        return roots


def _copy_and_update_matrix(matrix, names, roles):
    matrix = matrix.copy()

    if names is not None:
        _name_columns(matrix, names)

    if roles is not None:
        _update_roles(matrix, roles)

    return matrix


def _name_columns(matrix, names):
    # If the column names are included in the dataset, then remove the first
    # value from each column and use that at its name.
    if isinstance(names, str) and names == 'included':
        _extract_column_names(matrix)
        return

    # If the column names are missing from the dataset, then assign a new,
    # unique name to each column.
    if isinstance(names, str) and names == 'missing':
        _generate_column_names(matrix)
        return

    # If the column names are provided separately, then assign a name to each
    # column. Raise an exception if we don't have the right number of names.
    if isinstance(names, (list, tuple)):
        _update_column_names(matrix, names)
        return

    # Make sure that names is either 'unknown' or None.
    if names not in {'unknown', None}:
        valid_codes = {'included', 'missing', 'unknown', None}
        raise expected(
            f'column names to be a list of strings, or one of {valid_codes}',
            repr(names)
        )

    # Figure out whether the column names are included or missing, and then recur.
    are_included = _includes_column_names(matrix)
    _name_columns(matrix, 'included' if are_included else 'missing')


def _includes_column_names(matrix):
    # If the matrix is empty, then the column names must be missing.
    if len(matrix) == 0:
        return False

    # If the first row contains all strings, then assume those are the
    # column names.
    if all(isinstance(col.values[0], str) for col in matrix.columns):
        return True

    # If any column looks like it starts with a name, then assume that this
    # matrix must include the column names.
    for col in matrix.columns:
        vals = col.values
        has_name = isinstance(vals[0], str)
        has_rest = len(vals) > 1
        has_nums = any(i is not None and not isinstance(i, str) for i in vals[1:])
        no_strs = all(not isinstance(i, str) or i == '' for i in vals[1:])
        if has_name and has_rest and has_nums and no_strs:
            return True

    # Well, it doesn't look like this matrix contains the column names.
    return False


def _extract_column_names(matrix):
    if len(matrix) == 0:
        raise expected('nonempty dataset', 'empty dataset')

    for col in matrix.columns:
        # Use the first value, even if it's not a string.
        col.formula = str(col.values[0])

        # Let numpy infer a (potentially) new dtype.
        col.values = create_array(col.values[1:].tolist())


def _generate_column_names(matrix):
    for i, col in enumerate(matrix.columns):
        col.formula = excel_column_name(i)


def _update_column_names(matrix, names):
    assert isinstance(names, (list, tuple))

    if len(matrix.columns) != len(names):
        raise expected(f'{len(matrix.columns)} column names', len(names))

    if not all(isinstance(i, str) for i in names):
        raise expected('column names to be a list of strings', repr(names))

    for col, name in zip(matrix.columns, names):
        col.formula = name


def _update_roles(matrix, roles):
    if roles is None:
        return

    if isinstance(roles, (list, tuple)):
        _assign_roles(matrix, roles)
        return

    if not isinstance(roles, dict):
        raise expected('column roles to be a list or a dict', repr(roles))

    colmap = {col.name: col for col in matrix.columns}

    def lookup(key):
        if isinstance(key, str) and key not in colmap:
            names = [col.name for col in matrix.columns]
            raise expected(f'column to be one of {names}', repr(key))

        if isinstance(key, str):
            return colmap[key]

        try:
            return matrix.columns[key]
        except Exception:
            pass

        raise expected('valid column key', repr(key))

    for key, role in roles.items():
        col = lookup(key)
        col.role = role


def _assign_roles(matrix, roles):
    if len(matrix.columns) != len(roles):
        raise expected(f'{len(matrix.columns)} column roles', len(roles))

    for col, role in zip(matrix.columns, roles):
        col.role = role


def create_array(values):
    has_str = any(isinstance(i, str) for i in values)
    return np.array(values, dtype=object if has_str else None)
