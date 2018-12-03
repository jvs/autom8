import pytest
import numpy as np
import autom8


def test_invalid_arguments():
    with pytest.raises(autom8.Autom8Exception) as excinfo:
        autom8.create_matrix(0)
    excinfo.match('Expected.*list')

    with pytest.raises(autom8.Autom8Exception) as excinfo:
        autom8.create_matrix('')
    excinfo.match('Expected.*list')

    with pytest.raises(autom8.Autom8Exception) as excinfo:
        autom8.create_matrix({})
    excinfo.match('Expected.*list')


def test_empty_datasets():
    for data in [[], (), np.array([]), autom8.Matrix([])]:
        acc = autom8.Accumulator()
        matrix = autom8.create_matrix(data, receiver=acc)
        assert len(matrix.columns) == 0
        assert len(acc.warnings) == 0


def test_empty_dataset_with_empty_rows():
    # Assert that we see one warning when we have three empty rows.
    acc = autom8.Accumulator()
    matrix = autom8.create_matrix([[], [], []], receiver=acc)
    assert len(matrix.columns) == 0
    assert len(acc.warnings) == 1


def test_empty_dataset_warning_message():
    a1 = autom8.Accumulator()
    a2 = autom8.Accumulator()
    a3 = autom8.Accumulator()
    autom8.create_matrix([[]], receiver=a1)
    autom8.create_matrix([[], []], receiver=a2)
    autom8.create_matrix([[], [], []], receiver=a3)
    assert a1.warnings == ['Dropped 1 empty row from dataset.']
    assert a2.warnings == ['Dropped 2 empty rows from dataset.']
    assert a3.warnings == ['Dropped 3 empty rows from dataset.']


def test_extra_columns_warning_message():
    a1 = autom8.Accumulator()
    a2 = autom8.Accumulator()
    m1 = autom8.create_matrix([[1, 2], [1, 2, 3]], receiver=a1)
    m2 = autom8.create_matrix(
        [[1], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4]], receiver=a2
    )

    assert len(m1.columns), 2
    assert a1.warnings == [
        'Dropped 1 extra column from dataset.'
        ' Keeping first 2 columns.'
        ' To avoid this behavior, ensure that each row in the dataset has'
        ' the same number of columns.'
    ]

    assert len(m2.columns), 1
    assert a2.warnings == [
        'Dropped 3 extra columns from dataset.'
        ' Keeping first 1 column.'
        ' To avoid this behavior, ensure that each row in the dataset has'
        ' the same number of columns.'
    ]


def test_creating_simple_matrix_with_names_and_roles():
    acc = autom8.Accumulator()
    matrix = autom8.create_matrix(
        dataset=[['hi', True], ['bye', False]],
        column_names=['msg', 'flag'],
        column_roles=['textual', 'encoded'],
        receiver=acc,
    )

    c1, c2 = matrix.columns
    e1 = np.array(['hi', 'bye'], dtype=object)
    e2 = np.array([True, False], dtype=None)

    assert np.array_equal(c1.values, e1)
    assert np.array_equal(c2.values, e2)

    assert c1.name == 'msg'
    assert c2.name == 'flag'

    assert c1.role == 'textual'
    assert c2.role == 'encoded'

    assert c1.is_original
    assert c2.is_original

    assert len(acc.warnings) == 0


def test_creating_simple_matrix_from_list():
    acc = autom8.Accumulator()
    matrix = autom8.create_matrix(
        [['hi', 1, True], ['bye', 2, False]],
        receiver=acc,
    )

    c1, c2, c3 = matrix.columns
    e1 = np.array(['hi', 'bye'], dtype=object)
    e2 = np.array([1, 2], dtype=None)
    e3 = np.array([True, False], dtype=None)

    assert np.array_equal(c1.values, e1)
    assert np.array_equal(c2.values, e2)
    assert np.array_equal(c3.values, e3)

    assert c1.name == 'A'
    assert c2.name == 'B'
    assert c3.name == 'C'

    assert c1.role is None
    assert c2.role is None
    assert c3.role is None

    assert c1.is_original
    assert c2.is_original
    assert c3.is_original

    assert len(acc.warnings) == 0


def test_creating_simple_matrix_from_numpy_array():
    acc = autom8.Accumulator()
    matrix = autom8.create_matrix(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
        receiver=acc,
    )

    c1, c2, c3 = matrix.columns
    e1 = np.array([1, 4, 7, 10], dtype=object)
    e2 = np.array([2, 5, 8, 11], dtype=None)
    e3 = np.array([3, 6, 9, 12], dtype=None)

    assert np.array_equal(c1.values, e1)
    assert np.array_equal(c2.values, e2)
    assert np.array_equal(c3.values, e3)


def test_len_method():
    m1 = autom8.create_matrix([
        ['hi', 1, True],
        ['so', 2, True],
        ['bye', 3, False],
    ])
    m2 = autom8.create_matrix([[1], [2], [3], [4], [5], [6], [7]])
    assert len(m1) == 3
    assert len(m2) == 7


def test_copy_method():
    m1 = autom8.create_matrix([
        ['hi', 1.1, True],
        ['so', 2.2, True],
        ['bye', 3.3, False],
    ])
    m2 = autom8.create_matrix([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    n1, n2 = m1.copy(), m2.copy()

    assert m1 is not n1
    assert m2 is not n2

    assert len(m1.columns) == len(n1.columns)
    assert len(m2.columns) == len(n2.columns)

    for a, b in zip(m1.columns + m2.columns, n1.columns + n2.columns):
        assert a is not b
        assert a.values is not b.values
        assert a.name == b.name
        assert a.role == b.role
        assert a.is_original == b.is_original
        assert np.array_equal(a.values, b.values)


def test_tolist_method():
    m1 = autom8.create_matrix(
        dataset=[['hi', True], ['bye', False]],
        column_names=['msg', 'flag'],
        column_roles=['textual', 'encoded'],
    )
    m2 = autom8.create_matrix([[1, 2.0], [3, 4.0], [5, 6.0]])
    assert m1.tolist() == [['hi', True], ['bye', False]]
    assert m2.tolist() == [[1, 2.0], [3, 4.0], [5, 6.0]]


def test_to_array_method():
    m1 = autom8.create_matrix([[1], [2], [3], [4]])
    m2 = autom8.create_matrix([[1, 2], [3, 4], [5, 6]])
    assert np.array_equal(m1.to_array(), np.array([1, 2, 3, 4]))
    with pytest.raises(autom8.Autom8Exception) as excinfo:
        m2.to_array()
    excinfo.match('Expected.*one column')


def test_append_column():
    matrix = autom8.create_matrix([[1], [2], [3], [4]])
    matrix.append_column(np.array([2, 4, 6, 8]), 'foo', 'encoded')
    c1, c2 = matrix.columns
    assert c2.name == 'foo'
    assert c2.role == 'encoded'
    assert not c2.is_original
    assert np.array_equal(c2.values, np.array([2, 4, 6, 8]))
    assert not np.array_equal(c2.values, np.array([1, 2, 3, 4]))


def test_drop_columns_by_index():
    m1 = autom8.create_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    m2 = autom8.create_matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    m1.drop_columns_by_index([0, 2])
    m2.drop_columns_by_index([1, 2])
    assert len(m1.columns) == 1
    assert len(m2.columns) == 2
    assert np.array_equal(m1.columns[0].values, np.array([2, 5, 8]))
    assert m1.tolist() == [[2], [5], [8]]
    assert m2.tolist() == [[1, 4], [5, 8], [9, 12]]


def test_select_rows():
    mat = autom8.create_matrix([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    m1 = mat.select_rows([0, 2, 4])
    m2 = mat.select_rows([1, 3])
    assert m1.tolist() == [[1, 2], [5, 6], [9, 10]]
    assert m2.tolist() == [[3, 4], [7, 8]]


def test_exclude_rows():
    mat = autom8.create_matrix([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    m1 = mat.exclude_rows([0, 2, 4])
    m2 = mat.exclude_rows([1, 3])
    assert m1.tolist() == [[3, 4], [7, 8]]
    assert m2.tolist() == [[1, 2], [5, 6], [9, 10]]


def test_select_columns():
    mat = autom8.create_matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    m1 = mat.select_columns([0, 2, 3])
    m2 = mat.select_columns([1])
    assert m1.tolist() == [[1, 3, 4], [5, 7, 8], [9, 11, 12]]
    assert m2.tolist() == [[2], [6], [10]]


def test_exclude_columns():
    mat = autom8.create_matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    m1 = mat.exclude_columns([0, 2, 3])
    m2 = mat.exclude_columns([1])
    assert m1.tolist() == [[2], [6], [10]]
    assert m2.tolist() == [[1, 3, 4], [5, 7, 8], [9, 11, 12]]


def test_column_indices_where():
    matrix = autom8.create_matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    pred = lambda x: x.name == 'B' or x.name == 'D'
    indices = matrix.column_indices_where(pred)
    assert indices == [1, 3]


def test_column_len_method():
    matrix = autom8.create_matrix([[1, 2], [3, 4], [5, 6], [7, 8]])
    c1, c2 = matrix.columns
    assert len(c1) == 4
    assert len(c2) == 4


def test_column_dtype_property():
    matrix = autom8.create_matrix([
        ['hi', 10, 1.1, True, None],
        ['so', 20, 2.2, True, None],
        ['bye', 30, 3.3, False, None],
    ])
    c1, c2, c3, c4, c5 = matrix.columns
    assert c1.dtype == np.dtype('O')
    assert c2.dtype == np.dtype('int64')
    assert c3.dtype == np.dtype('float64')
    assert c4.dtype == np.dtype('bool')
    assert c5.dtype == np.dtype('O')


def test_properties_of_valid_column_roles():
    matrix = autom8.create_matrix([
        [1, 2, 3, 4, 'a'],
        [2, 3, 4, 5, 'b'],
        [3, 4, 5, 6, 'c'],
    ])
    c1, c2, c3, c4, c5 = matrix.columns
    c1.role = None
    c2.role = 'categorical'
    c3.role = 'encoded'
    c4.role = 'numerical'
    c5.role = 'textual'

    assert not c1.is_numerical
    assert not c2.is_numerical
    assert not c3.is_numerical
    assert c4.is_numerical
    assert not c5.is_numerical

    assert c1.role is None
    assert c2.role == 'categorical'
    assert c3.role == 'encoded'
    assert c4.role == 'numerical'
    assert c5.role == 'textual'


def test_setting_an_invalid_column_role():
    matrix = autom8.create_matrix([[1], [2], [3]])
    col = matrix.columns[0]
    with pytest.raises(autom8.Autom8Exception) as excinfo:
        col.role = 'foo'
    excinfo.match('Expected.*role in')


def test_duplicate_column_names():
    acc = autom8.Accumulator()
    matrix = autom8.create_matrix(
        dataset=[[1, 2, 3]],
        column_names=['A', 'B', 'A'],
        receiver=acc,
    )
    assert len(acc.warnings) == 1
    assert 'Column names are not unique' in acc.warnings[0]


def test_select_columns_by_name_with_superset():
    dataset = [
        ['hi', 1, True, 10.5],
        ['so', 2, True, 15.5],
        ['bye', 3, False, 20.5],
    ]
    matrix = autom8.create_matrix(dataset, column_names=['A', 'B', 'C', 'D'])
    received = matrix.select_columns_by_name(['C', 'B'])
    assert received.tolist() == [[True, 1], [True, 2], [False, 3]]


def test_select_columns_by_name_with_invalid_arguments():
    dataset = [
        ['hi', 1, True, 10.5],
        ['so', 2, True, 15.5],
        ['bye', 3, False, 20.5],
    ]
    matrix = autom8.create_matrix(dataset, column_names=['A', 'B', 'C', 'D'])
    with pytest.raises(autom8.Autom8Exception) as excinfo:
        matrix.select_columns_by_name(['C', 'Z'])
    excinfo.match('Expected column names')


def test_creating_a_matrix_with_list_of_roles():
    dataset = [
        ['hi', 1, True, 10.5],
        ['so', 2, True, 15.5],
        ['bye', 3, False, 20.5],
    ]
    matrix = autom8.create_matrix(dataset,
        column_roles=['textual', 'categorical', 'encoded', 'numerical']
    )
    assert matrix.columns[0].role == 'textual'
    assert matrix.columns[1].role == 'categorical'
    assert matrix.columns[2].role == 'encoded'
    assert matrix.columns[3].role == 'numerical'


def test_creating_a_matrix_with_map_of_roles():
    dataset = [
        ['hi', 1, True, 10.5],
        ['so', 2, True, 15.5],
        ['bye', 3, False, 20.5],
    ]
    matrix = autom8.create_matrix(dataset,
        column_names=['A', 'B', 'C', 'D'],
        column_roles={
            'D': 'numerical',
            'C': 'encoded',
            1: 'categorical',
            'A': 'textual',
        },
    )
    assert matrix.columns[0].role == 'textual'
    assert matrix.columns[1].role == 'categorical'
    assert matrix.columns[2].role == 'encoded'
    assert matrix.columns[3].role == 'numerical'

    with pytest.raises(autom8.Autom8Exception) as excinfo:
        autom8.create_matrix(dataset,
            column_names=['A', 'B', 'C', 'D'],
            column_roles={'Z': 'numerical'},
        )
    excinfo.match('Expected column')

    with pytest.raises(autom8.Autom8Exception) as excinfo:
        autom8.create_matrix(dataset,
            column_names=['A', 'B', 'C', 'D'],
            column_roles={object(): 'numerical'},
        )
    excinfo.match('Expected valid column')


def test_coerce_values_to_numbers_with_weird_values():
    matrix = autom8.create_matrix([[1], [2.0], ['3'], ['hi'], [object()], [4]])
    matrix.coerce_values_to_numbers(default=100, as_type=int)
    assert matrix.tolist() == [[1], [2], [3], [100], [100], [4]]
