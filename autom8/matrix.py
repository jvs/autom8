from collections import namedtuple
import functools
import numpy as np

from .observer import Observer
from .exceptions import expected


def create_matrix(data, observer=None):
    if not isinstance(data, (dict, list, tuple)):
        raise expected('dict, list, or tuple', type(data))

    if observer is None:
        observer = Observer()

    if isinstance(data, dict):
        return _create_matrix_from_dict(data, observer)
    else:
        return _create_matrix_from_iterable(data, observer)


class Matrix:
    def __init__(self, columns):
        assert isinstance(columns, list)
        assert all(isinstance(i, Column) for i in columns)
        self.columns = columns

    @property
    def schema(self):
        return [{'name': col.name, 'role': col.role} for col in self.columns]

    def __len__(self):
        return len(self.columns[0]) if self.columns else 0

    def __repr__(self):
        return f'Matrix({repr(self.tolist())})'

    def copy(self):
        return Matrix([i.copy() for i in self.columns])

    def tolist(self):
        result = [[c.name for c in self.columns]]
        result.extend(self.stack_columns().tolist())
        return result

    def stack_columns(self):
        return np.column_stack([col.values.astype(object) for col in self.columns])

    def to_array(self):
        if len(self.columns) == 1:
            return self.columns[0].values
        else:
            raise expected('matrix with only one column', len(self.columns))

    def append_column(self, values, name, role):
        if not hasattr(values, 'shape'):
            raise expected('numpy array', type(values))

        if len(values.shape) != 1:
            raise expected('array of values', values.shape)

        col = Column(values=values, name=name, role=role, is_original=False)
        self.columns.append(col)

    def drop_columns_by_index(self, indices):
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

    def column_indices_where(self, predicate):
        return [i for i, col in enumerate(self.columns) if predicate(col)]

    def coerce_values_to_strings(self):
        for col in self.columns:
            if not all(isinstance(i, str) for i in col.values):
                col.values = np.array([str(i) for i in col.values], dtype=object)

    def coerce_values_to_numbers(self, default=0, as_type=None):
        def conv(obj):
            if isinstance(obj, (int, float)):
                return obj
            try:
                return float(obj)
            except Exception:
                pass
            try:
                return int(obj, 16)
            except Exception:
                pass
            return default

        for col in self.columns:
            if col.dtype not in (int, float):
                col.values = np.array([conv(i) for i in col.values], dtype=float)

        if as_type is not None:
            for col in self.columns:
                if col.dtype != as_type:
                    col.values = col.values.astype(as_type)

class Column:
    def __init__(self, values, name, role, is_original):
        assert len(values.shape) == 1
        assert isinstance(name, str)
        assert role is None or isinstance(role, str)
        assert isinstance(is_original, bool)
        self.values = values
        self.name = name
        self._role = None
        self.role = role
        self.is_original = is_original

    def __len__(self):
        return len(self.values)

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
            raise expected(f'role in {valid_roles}', role)

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
            name=self.name,
            role=role,
            is_original=self.is_original,
        )

    def select_rows(self, indices):
        return self.copy_with(self.values[indices])

    def exclude_rows(self, indices):
        return self.copy_with(np.delete(self.values, indices))


def _create_matrix_from_dict(data, observer):
    assert isinstance(data, dict)

    if 'rows' not in data:
        raise expected('dict with "rows" element', list(data.keys()))

    if 'schema' not in data:
        raise expected('dict with "schema" element', list(data.keys()))

    # Unpack the two entries that we care about.
    rows, schema = data['rows'], data['schema']

    if not isinstance(schema, (list, tuple)):
        raise expected('"schema" to be a list of dict objects.', type(schema))

    if not all(isinstance(i, dict) for i in schema):
        raise expected('"schema" to be a list of dict objects.', type(schema[0]))

    if not all('name' in i and 'role' in i for i in schema):
        raise expected('schema items to contain "name" and "role" entries.', schema[0])

    # Create the matrix and then update the columns.
    matrix = create_matrix(rows, observer)
    for col, details in zip(matrix.columns, schema):
        col.name = details['name']
        col.role = details['role']
    return matrix


def _create_matrix_from_iterable(data, observer):
    if len(data) > 0 and isinstance(data[0], Column):
        # In this case, require each element to be a Column object.
        if not all(isinstance(i, Column) for i in data):
            raise expected('list of Column objects', [type(i) for i in data])

        # Make a copy of the columns and return the matrix.
        return Matrix([i.copy() for i in data])

    # Drop empty rows.
    rows = [row for row in data if row]
    num_dropped = len(data) - len(rows)

    # Warn the users if we dropped some rows.
    if num_dropped:
        suffix = '' if num_dropped == 1 else 's'
        observer.warn(f'Dropped {num_dropped} empty row{suffix} from dataset.')

    # If we don't have any rows, then just return an empty matrix.
    if not rows:
        return Matrix([])

    # Figure out how many columns we need.
    mincols = min(len(row) for row in rows)
    maxcols = max(len(row) for row in rows)

    # Warn the users if we're dropping any extra columns.
    if mincols < maxcols:
        num_extra = maxcols - mincols
        suffix1 = '' if num_extra == 1 else 's'
        suffix2 = '' if mincols == 1 else 's'
        observer.warn(
            f'Dropped {num_extra} extra column{suffix1} from dataset.'
            f' Keeping first {mincols} column{suffix2}.'
            ' To avoid this behavior, ensure that each row in the dataset has'
            ' the same number of columns.'
        )

    columns = [_make_column(rows, i) for i in range(mincols)]
    return Matrix(columns)


def _make_column(rows, index):
    raw = [row[index] for row in rows]
    has_text = any(isinstance(i, str) for i in raw)
    cooked = np.array(raw, dtype=object if has_text else None)
    name = f'Column-{index + 1}'
    return Column(values=cooked, name=name, role=None, is_original=True)
