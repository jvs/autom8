import re
import numpy as np
from .observer import Observer
from .exceptions import expected, typename


def create_matrix(data, observer=None, infer_names=True):
    if observer is None:
        observer = Observer()

    if not isinstance(data, (dict, list, tuple, np.ndarray, Matrix)):
        raise expected('dict, list, tuple, numpy array, or Matrix', typename(data))

    if isinstance(data, Matrix):
        return data

    if isinstance(data, dict):
        return _create_matrix_from_dict(data, observer)
    else:
        return _create_matrix_from_iterable(data, observer, infer_names)


class Matrix:
    def __init__(self, columns):
        assert isinstance(columns, list)
        assert all(isinstance(i, Column) for i in columns)
        self.columns = columns

    @property
    def schema(self):
        return [col.schema for col in self.columns]

    def __len__(self):
        return len(self.columns[0]) if self.columns else 0

    def __repr__(self):
        if len(self) == 0:
            return 'Matrix([])'
        try:
            return f'Matrix({repr(self.tolist())})'
        except Exception:
            return '<Matrix>'

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

    def append_column(self, values, name, role, is_original=False):
        if not hasattr(values, 'shape'):
            raise expected('numpy array', type(values))

        if len(values.shape) != 1:
            raise expected('array of values', values.shape)

        col = Column(values=values, name=name, role=role, is_original=is_original)
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
    def schema(self):
        return {'name': self.name, 'role': self.role, 'dtype': self.dtype.name}

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

    has_schema = 'schema' in data
    matrix = create_matrix(data['rows'], observer, infer_names=not has_schema)

    if not has_schema:
        return matrix

    schema = data['schema']

    if not isinstance(schema, (list, tuple)):
        raise expected('"schema" to be a list of dict objects.', type(schema))

    if not all(isinstance(i, dict) for i in schema):
        raise expected('"schema" to be a list of dict objects.', type(schema[0]))

    if not all('name' in i and 'role' in i for i in schema):
        raise expected('schema items to contain "name" and "role" entries.', schema[0])

    num_cols = len(matrix.columns)
    if num_cols != len(schema):
        raise expected(f'schema to have one item for each column ({num_cols})',
            len(schema))

    for col, details in zip(matrix.columns, schema):
        col.name = _merge_names(col.name, details['name'], observer)
        col.role = details['role']

        if 'dtype' in details:
            _coerce_values(col, details['dtype'], observer)

    return matrix


def _merge_names(inferred, provided, observer):
    n1, n2 = inferred.strip(), provided.strip()
    is_anonymous = re.match(r'Column-\d+', n1)
    if not is_anonymous and n1.lower() != n2.lower():
        observer.warn(f'Found column with two names: {inferred} and {provided}.')
    return provided.strip()


def _create_matrix_from_iterable(data, observer, infer_names=False):
    if len(data) > 0 and isinstance(data[0], Column):
        # In this case, require each element to be a Column object.
        if not all(isinstance(i, Column) for i in data):
            raise expected('list of Column objects', [typename(i) for i in data])

        # Make a copy of the columns and return the matrix.
        return Matrix([i.copy() for i in data])

    # Drop empty rows.
    rows = _drop_empty_rows(data)
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

    anon = Matrix([_make_column(rows, i) for i in range(mincols)])
    return _infer_column_names(anon) if infer_names else anon


def _drop_empty_rows(rows):
    # This is phrased a bit awkwardly to also filter out empty rows.
    return [row for row in rows if any(not _is_blank(i) for i in row)]


def _is_blank(obj):
    if isinstance(obj, str):
        return obj == '' or obj.isspace()
    else:
        return obj is None


def _make_column(rows, index):
    raw = [row[index] for row in rows]
    has_text = any(isinstance(i, str) for i in raw)
    values = np.array(raw, dtype=object if has_text else None)
    name = f'Column-{index + 1}'
    return Column(values=values, name=name, role=None, is_original=True)


def _infer_column_names(matrix):
    # If the matrix is empty, then just stop here.
    if len(matrix) == 0:
        return matrix

    # Get the first value from each column.
    row = [col.values[0] for col in matrix.columns]

    # If the first row is not all text, then stop.
    all_text = all(isinstance(i, str) for i in row)
    if not all_text:
        return matrix

    # The first row contains the column names.
    columns = []
    for col, name in zip(matrix.columns, row):
        # Let numpy infer a (potentially) new dtype.
        rest = col.values[1:].tolist()
        columns.append(Column(np.array(rest), name, col.role, col.is_original))

    return Matrix(columns)


def _coerce_values(col, dtype, observer):
    try:
        new_values = col.values.astype(dtype)
    except Exception:
        observer.warn(f'Failed to convert column {repr(col.name)} to type {dtype}')
