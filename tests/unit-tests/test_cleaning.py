import logging

import autom8
from autom8.pipeline import PlaybackContext
from autom8.preprocessors import playback


def setup_module(*a, **k):
    logging.disable(logging.CRITICAL)


def teardown_module(*a, **k):
    logging.disable(logging.NOTSET)


def test_matrix_with_unexpected_value():
    dataset = [
        ['A', 'B', 'C'],
        [1, 2, ()],
        [3, 4, {}],
        [5, 6, object()],
    ]
    acc = autom8.Accumulator()
    matrix = autom8.create_matrix(_add_labels(dataset), receiver=acc)
    ctx = autom8.create_context(matrix, receiver=acc)

    autom8.clean_dataset(ctx)
    assert len(acc.warnings) == 1
    assert 'Dropping column' in acc.warnings[0]
    assert 'contain booleans, numbers' in acc.warnings[0]
    assert ctx.matrix.tolist() == [[1, 2], [3, 4], [5, 6]]

    vectors = [['A', 'B', 'C'], [1, 2, 'foo'], [3, 4, 'bar']]
    matrix = autom8.create_matrix(vectors, receiver=acc)
    out = PlaybackContext(matrix, receiver=acc)
    playback(ctx.steps, out)
    assert out.matrix.tolist() == [[1, 2], [3, 4]]


def test_primitives_with_object_dtype():
    dataset = [
        ['A', 'B', 'C'],
        [True, 1.1, 2],
        [False, 3.1, 4],
        [True, 5.1, 6],
    ]

    acc = autom8.Accumulator()
    matrix = autom8.create_matrix(_add_labels(dataset), receiver=acc)
    for col in matrix.columns:
        col.values = col.values.astype(object)

    ctx = autom8.create_context(matrix, receiver=acc)
    autom8.clean_dataset(ctx)

    dtypes = [c.dtype for c in ctx.matrix.columns]
    assert dtypes[0] == bool
    assert dtypes[1] == float
    assert dtypes[2] == int

    vectors = [['A', 'B', 'C'], [1, 2, 3.0], [0, 4, 5.0], [1, False, 6.9]]
    matrix = autom8.create_matrix(vectors, receiver=acc)
    out = PlaybackContext(matrix, receiver=acc)
    playback(ctx.steps, out)
    assert out.matrix.tolist() == [
        [True, 2.0, 3], [False, 4.0, 5], [True, 0.0, 6]
    ]

    dtypes = [c.dtype for c in out.matrix.columns]
    assert dtypes[0] == bool
    assert dtypes[1] == float
    assert dtypes[2] == int

    vectors = [['A', 'B', 'C'], ['1', '2', None], ['', None, ()]]
    matrix = autom8.create_matrix(vectors, receiver=acc)
    out = PlaybackContext(matrix, receiver=acc)
    playback(ctx.steps, out)

    # Just use repr to avoid having to fart around with nan.
    assert repr(out.matrix.tolist()) == (
        "[[True, 2.0, 0], [False, nan, 0]]"
    )


def test_column_with_all_none():
    dataset = [
        ['A', 'B', 'C'],
        [True, None, 2],
        [False, None, 4],
        [True, None, 6],
    ]

    acc = autom8.Accumulator()
    matrix = autom8.create_matrix(_add_labels(dataset), receiver=acc)
    ctx = autom8.create_context(matrix, receiver=acc)

    autom8.clean_dataset(ctx)
    assert len(acc.warnings) == 1
    assert 'Dropping column' in acc.warnings[0]
    assert ctx.matrix.tolist() == [[True, 2], [False, 4], [True, 6]]

    vectors = [['A', 'B', 'C'], [1, 2, 'foo'], [3, 4, 'bar']]
    matrix = autom8.create_matrix(vectors, receiver=acc)
    out = PlaybackContext(matrix, receiver=acc)
    playback(ctx.steps, out)
    assert out.matrix.tolist() == [[1, 'foo'], [3, 'bar']]


def test_columns_with_numbers_as_strings():
    dataset = [
        ['A', 'B', 'C'],
        ['1.1', '$4', 7],
        ['2.2', '$5', 8],
        ['3.3', '6%', 9],
    ]

    acc = autom8.Accumulator()
    matrix = autom8.create_matrix(_add_labels(dataset), receiver=acc)
    ctx = autom8.create_context(matrix, receiver=acc)

    autom8.clean_dataset(ctx)
    assert len(acc.warnings) == 0
    assert len(ctx.steps) == 2

    assert ctx.matrix.tolist() == [[1.1, 4, 7], [2.2, 5, 8], [3.3, 6, 9]]

    vectors = [['A', 'B', 'C'], [1, '2%', 'foo'], ['3', 4.0, 'bar']]
    matrix = autom8.create_matrix(vectors, receiver=acc)
    out = PlaybackContext(matrix, receiver=acc)
    playback(ctx.steps, out)
    assert out.matrix.tolist() == [[1, 2, 'foo'], [3, 4, 'bar']]
    assert out.matrix.columns[0].dtype == int
    assert out.matrix.columns[1].dtype == float


def test_columns_with_numbers_with_commas():
    dataset = [['A'], ['1,100.0'], ['2,200'], ['3,300'], ['50']]
    acc = autom8.Accumulator()
    matrix = autom8.create_matrix(_add_labels(dataset), receiver=acc)
    ctx = autom8.create_context(matrix, receiver=acc)
    autom8.clean_dataset(ctx)
    assert len(acc.warnings) == 0
    assert len(ctx.steps) == 1
    assert ctx.matrix.tolist() == [[1100], [2200], [3300], [50]]


def test_column_of_all_strings():
    dataset = [
        ['A', 'B'],
        ['1', 2],
        ['3', 4],
        ['n', 0],
    ]

    acc = autom8.Accumulator()
    matrix = autom8.create_matrix(_add_labels(dataset), receiver=acc)
    ctx = autom8.create_context(matrix, receiver=acc)

    autom8.clean_dataset(ctx)
    assert len(acc.warnings) == 0
    assert len(ctx.steps) == 0
    assert ctx.matrix.tolist() == [['1', 2], ['3', 4], ['n', 0]]


def test_column_of_all_strings_and_none_values():
    dataset = [
        ['A', 'B'],
        ['1', 2],
        ['foo', 4],
        [None, 0],
    ]

    acc = autom8.Accumulator()
    matrix = autom8.create_matrix(_add_labels(dataset), receiver=acc)
    ctx = autom8.create_context(matrix, receiver=acc)

    autom8.clean_dataset(ctx)
    assert len(acc.warnings) == 0
    assert len(ctx.steps) == 1
    assert ctx.matrix.tolist() == [['1', 2], ['foo', 4], ['', 0]]

    vectors = [['A', 'B'], [None, 'bar'], ['baz', None]]
    matrix = autom8.create_matrix(vectors, receiver=acc)
    out = PlaybackContext(matrix, receiver=acc)
    playback(ctx.steps, out)
    assert out.matrix.tolist() == [['', 'bar'], ['baz', None]]


def test_column_of_ints_and_floats():
    dataset = [
        ['A', 'B'],
        [1, 3.3],
        [2.2, 4],
        [None, None],
    ]

    acc = autom8.Accumulator()
    matrix = autom8.create_matrix(_add_labels(dataset), receiver=acc)
    ctx = autom8.create_context(matrix, receiver=acc)

    autom8.clean_dataset(ctx)

    assert len(ctx.steps) == 4
    assert len(acc.warnings) == 2
    assert ctx.matrix.tolist() == [
        [1.0, True, 3.3, True],
        [2.2, True, 4.0, True],
        [0.0, False, 0.0, False],
    ]

    vectors = [['A', 'B'], [None, 10], [20.0, None], [30, 40]]
    matrix = autom8.create_matrix(vectors, receiver=acc)
    out = PlaybackContext(matrix, receiver=acc)
    playback(ctx.steps, out)
    assert out.matrix.tolist() == [
        [0.0, False, 10.0, True],
        [20.0, True, 0.0, False],
        [30.0, True, 40.0, True],
    ]

    assert out.matrix.columns[0].dtype == float
    assert out.matrix.columns[2].dtype == float


def test_columns_with_some_empty_strings():
    dataset = [
        ['A', 'B', 'C'],
        [True, 1.1, 20],
        ['', 2.2, 30],
        [False, '', 40],
        [False, 3.3, ''],
        ['', 4.4, ''],
    ]

    acc = autom8.Accumulator()
    matrix = autom8.create_matrix(_add_labels(dataset), receiver=acc)
    ctx = autom8.create_context(matrix, receiver=acc)

    autom8.clean_dataset(ctx)

    assert len(ctx.steps) == 6
    assert len(acc.warnings) == 3
    assert ctx.matrix.tolist() == [
        [True, True, 1.1, True, 20, True],
        [False, False, 2.2, True, 30, True],
        [False, True, 0.0, False, 40, True],
        [False, True, 3.3, True, 0, False],
        [False, False, 4.4, True, 0, False],
    ]
    assert ctx.matrix.formulas == [
        'A', ['is-defined', 'A'],
        'B', ['is-defined', 'B'],
        'C', ['is-defined', 'C'],
    ]

    vectors = [['A', 'B', 'C'], ['', 5.5, ''], [True, '', 50]]
    matrix = autom8.create_matrix(vectors, receiver=acc)
    out = PlaybackContext(matrix, receiver=acc)
    playback(ctx.steps, out)
    assert out.matrix.tolist() == [
        [False, False, 5.5, True, 0, False],
        [True, True, 0.0, False, 50, True],
    ]
    assert out.matrix.formulas == [
        'A', ['is-defined', 'A'],
        'B', ['is-defined', 'B'],
        'C', ['is-defined', 'C'],
    ]


def test_column_with_some_blank_strings():
    # Repeat the previous test, only replace most of the empty strings with
    # blank strings.
    dataset = [
        ['A', 'B', 'C'],
        [True, 1.1, 20],
        [' ', 2.2, 30],
        [False, '\t', 40],
        [False, 3.3, ' \t \r \n\t'],
        ['', 4.4, '    '],
    ]

    acc = autom8.Accumulator()
    matrix = autom8.create_matrix(_add_labels(dataset), receiver=acc)
    ctx = autom8.create_context(matrix, receiver=acc)

    autom8.clean_dataset(ctx)

    assert ctx.matrix.tolist() == [
        [True, True, 1.1, True, 20, True],
        [False, False, 2.2, True, 30, True],
        [False, True, 0.0, False, 40, True],
        [False, True, 3.3, True, 0, False],
        [False, False, 4.4, True, 0, False],
    ]
    assert ctx.matrix.formulas == [
        'A', ['is-defined', 'A'],
        'B', ['is-defined', 'B'],
        'C', ['is-defined', 'C'],
    ]


def test_mixed_up_columns_with_strings_and_numbers():
    dataset = [
        ['A', 'B'],
        [True, 'foo'],
        [1.1, 30],
        [20, 4.4],
        ['bar', False],
        ['', 'baz'],
        [50, 'fiz'],
        [None, True],
    ]

    acc = autom8.Accumulator()
    matrix = autom8.create_matrix(_add_labels(dataset), receiver=acc)
    ctx = autom8.create_context(matrix, receiver=acc)

    autom8.clean_dataset(ctx)

    assert len(ctx.steps) == 6
    assert len(acc.warnings) == 0
    assert ctx.matrix.tolist() == [
        [1.0, '', 0.0, 'foo'],
        [1.1, '', 30.0, ''],
        [20.0, '', 4.4, ''],
        [0.0, 'bar', 0.0, ''],
        [0.0, '', 0.0, 'baz'],
        [50.0, '', 0.0, 'fiz'],
        [0.0, '', 1.0, ''],
    ]
    assert ctx.matrix.formulas == [
        ['number', 'A'], ['string', 'A'],
        ['number', 'B'], ['string', 'B'],
    ]

    vectors = [['A', 'B'], [False, 'buz'], ['zim', 10], [2, None]]
    matrix = autom8.create_matrix(vectors, receiver=acc)
    out = PlaybackContext(matrix, receiver=acc)
    playback(ctx.steps, out)
    assert out.matrix.tolist() == [
        [0.0, '', 0.0, 'buz'],
        [0.0, 'zim', 10.0, ''],
        [2.0, '', 0.0, ''],
    ]
    assert out.matrix.formulas == [
        ['number', 'A'], ['string', 'A'],
        ['number', 'B'], ['string', 'B'],
    ]


def test_clean_numeric_labels():
    dataset = [
        ['A', 'B', 'C'],
        [1, 2, '3'],
        [3, 4, '4'],
        [5, 6, 5],
        [7, 8, None],
        [9, 9, ''],
    ]
    acc = autom8.Accumulator()
    ctx = autom8.create_context(dataset, receiver=acc)

    assert len(acc.warnings) == 1
    assert ctx.labels.original.tolist() == [3, 4, 5, 0, 0]


def _add_labels(dataset):
    return [i + ['<label>'] for i in dataset]
