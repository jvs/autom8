import math
import numpy as np

import autom8
from autom8.pipeline import PlaybackContext
from autom8.preprocessors import playback


def test_add_column_of_ones():
    features = [['a', 'b'], ['c', 'd']]
    roles = ['categorical'] * 2

    ctx = _create_context(features, roles)
    autom8.add_column_of_ones(ctx)

    assert len(ctx.steps) == 1
    assert ctx.matrix.formulas == ['A', 'B', ['constant(1)']]
    assert ctx.matrix.tolist() == [['a', 'b', 1], ['c', 'd', 1]]

    # Try replaying it on the original data.
    assert _playback(ctx, roles, features) == ctx.matrix.tolist()

    m1 = _playback(ctx, roles, [['e', 'f'], ['g', 'h']])
    assert m1 == [['e', 'f', 1], ['g', 'h', 1]]


def test_binarize_fractions():
    features = [[1, 0.2], [0.3, 4]]
    roles = ['numerical'] * 2

    ctx = _create_context(features, roles)
    autom8.binarize_fractions(ctx)

    assert len(ctx.steps) == 1
    assert ctx.matrix.formulas == [
        'A', 'B', ['is-fraction', 'A'], ['is-fraction', 'B']
    ]
    assert ctx.matrix.tolist() == [
        [1, 0.2, False, True],
        [0.3, 4, True, False],
    ]

    # Try replaying it on the original data.
    assert _playback(ctx, roles, features) == ctx.matrix.tolist()

    m1 = _playback(ctx, roles, [[0.99, 1.1], [1.0, -0.1], [0.0, -1.2]])
    assert m1 == [
        [0.99, 1.1, True, False],
        [1.0, -0.1, False, True],
        [0.0, -1.2, True, False],
    ]


def test_binarize_signs():
    features = [[1, -2], [-3, 4], [0, 5]]
    roles = ['numerical'] * 2

    ctx = _create_context(features, roles)
    autom8.binarize_signs(ctx)

    assert len(ctx.steps) == 1
    assert ctx.matrix.formulas == [
        'A', 'B', ['is-positive', 'A'], ['is-positive', 'B']
    ]
    assert ctx.matrix.tolist() == [
        [1, -2, True, False],
        [-3, 4, False, True],
        [0, 5, False, True],
    ]

    # Try replaying it on the original data.
    assert _playback(ctx, roles, features) == ctx.matrix.tolist()

    m1 = _playback(ctx, roles, [[10, -0.1], [0.1, -10]])
    assert m1 == [
        [10, -0.1, True, False],
        [0.1, -10, True, False],
    ]


def test_divide_columns():
    features = [
        [1, 4, 'a', 7.0],
        [2, 5, 'b', 8.0],
        [3, 6, 'c', 9.0],
    ]

    roles = ['numerical', 'numerical', 'categorical', 'numerical']

    ctx = _create_context(features, roles)
    autom8.divide_columns(ctx)
    assert ctx.matrix.formulas == [
        'A', 'B', 'C', 'D',
        ['divide', 'A', 'B'], ['divide', 'A', 'D'],
        ['divide', 'B', 'A'], ['divide', 'B', 'D'],
        ['divide', 'D', 'A'], ['divide', 'D', 'B'],
    ]

    assert len(ctx.steps) == 1
    assert ctx.matrix.tolist() == [
        [1, 4, 'a', 7.0, 1 / 4, 1 / 7, 4 / 1, 4 / 7, 7 / 1, 7 / 4],
        [2, 5, 'b', 8.0, 2 / 5, 2 / 8, 5 / 2, 5 / 8, 8 / 2, 8 / 5],
        [3, 6, 'c', 9.0, 3 / 6, 3 / 9, 6 / 3, 6 / 9, 9 / 3, 9 / 6],
    ]

    # Try replaying it on the original data.
    assert _playback(ctx, roles, features) == ctx.matrix.tolist()

    # Try playing it back on some new data.
    m2 = _playback(ctx, roles, [
        [1, 2, 'x', 3.0],
        [4, 5, 'y', 6.0],
        [7, 8, 'z', 9.0],
    ])
    assert m2 == [
        [1, 2, 'x', 3.0, 1 / 2, 1 / 3, 2 / 1, 2 / 3, 3 / 1, 3 / 2],
        [4, 5, 'y', 6.0, 4 / 5, 4 / 6, 5 / 4, 5 / 6, 6 / 4, 6 / 5],
        [7, 8, 'z', 9.0, 7 / 8, 7 / 9, 8 / 7, 8 / 9, 9 / 7, 9 / 8],
    ]

    # Try playing it back when a zero appears as a denominator.
    m3 = _playback(ctx, roles, [
        [1, 3, 'x', 0.0],
        [2, 4, 'y', 5.0],
    ])
    assert m3 == [
        [1, 3, 'x', 0.0, 1 / 3, 0, 3 / 1, 0, 0, 0],
        [2, 4, 'y', 5.0, 2 / 4, 2 / 5, 4 / 2, 4 / 5, 5 / 2, 5 / 4],
    ]


def test_drop_duplicate_columns():
    features = [
        [np.nan, np.nan, 1, 'a', np.nan, 'a'],
        [np.inf, np.inf, 2, 'b', np.inf, 'b'],
        [0, 0, 3, 'c', 0, 'c'],
        [1, 1, 4, 'd', 1, 'd'],
    ]

    roles = [
        'numerical', 'numerical', 'numerical',
        'categorical', 'numerical', 'categorical',
    ]

    ctx = _create_context(features, roles)
    autom8.drop_duplicate_columns(ctx)

    assert len(ctx.steps) == 1

    # Just use repr for now, instead of wrestling with those NaN values.
    assert repr(ctx.matrix.tolist()) == repr([
        [np.nan, 1, 'a'],
        [np.inf, 2, 'b'],
        [0.0, 3, 'c'],
        [1.0, 4, 'd'],
    ])

    # Try replaying it on the original data.
    assert repr(_playback(ctx, roles, features)) == repr(ctx.matrix.tolist())


def test_ordinal_encode_categories():
    features = [
        [1.1, 10, 'foo', 'bar', True, 5.0],
        [2.2, 20, 'bar', 'foo', False, 5.0],
        [3.3, 30, 'foo', 'foo', True, 5.0],
    ]

    roles = [
        'numerical', 'categorical', 'categorical',
        'categorical', 'encoded', 'categorical',
    ]

    ctx = _create_context(features, roles)
    autom8.encode_categories(ctx, method='ordinal', only_strings=False)

    assert len(ctx.steps) == 1
    assert ctx.matrix.formulas == [
        'A', 'E', ['encode', 'B'], ['encode', 'C'], ['encode', 'D'], ['encode', 'F'],
    ]
    assert ctx.matrix.tolist() == [
        [1.1, True, 1, 1, 1, 1],
        [2.2, False, 2, 2, 2, 1],
        [3.3, True, 3, 1, 2, 1],
    ]

    # Try replaying it on the original data.
    assert _playback(ctx, roles, features) == ctx.matrix.tolist()

    # Try a case where all the values are expected.
    m2 = _playback(ctx, roles, [
        [1.11, 30, 'bar', 'bar', False, 5.0],
        [2.22, 20, 'foo', 'foo', False, 5.0],
        [3.33, 10, 'bar', 'foo', True, 5.0],
    ])
    assert m2 == [
        [1.11, False, 3, 2, 1, 1],
        [2.22, False, 2, 1, 2, 1],
        [3.33, True, 1, 2, 2, 1],
    ]

    # Now try a case where some of the values are unexpected.
    m3 = _playback(ctx, roles, [
        [1.12, 33, 'baz', 'bar', False, 5.0],
        [2.23, 22, 'foo', 'boo', False, 5.0],
        [3.34, 11, 'baz', 'boo', True, 6.0],
    ])
    assert m3 == [
        [1.12, False, 0, 0, 1, 1],
        [2.23, False, 0, 1, 0, 1],
        [3.34, True, 0, 0, 0, 0],
    ]

    # Now try a case where one of the columns has the wrong type.
    m4 = _playback(ctx, roles, [
        [1.17, 20, 3, 'foo', False, 5.0],
        [2.26, 10, 4, 'bar', True, 5.0],
        [3.35, 30, 5, 'baz', True, 5.0],
    ])
    assert m4 == [
        [1.17, False, 2, 0, 2, 1],
        [2.26, True, 1, 0, 1, 1],
        [3.35, True, 3, 0, 0, 1],
    ]

    # Now try just encoding the string columns.
    ctx = _create_context(features, roles)
    autom8.encode_categories(ctx, method='ordinal', only_strings=True)
    assert ctx.matrix.formulas == [
        'A', 'B', 'E', 'F', ['encode', 'C'], ['encode', 'D']
    ]

    # Once again, try playing it back on the original data.
    assert _playback(ctx, roles, features) == ctx.matrix.tolist()

    # Try a case where all the values are expected.
    m6 = _playback(ctx, roles, [
        [1.11, 30, 'bar', 'bar', False, 5.0],
        [2.22, 20, 'foo', 'foo', False, 5.0],
        [3.33, 10, 'bar', 'foo', True, 5.0],
    ])
    assert m6 == [
        [1.11, 30, False, 5.0, 2, 1],
        [2.22, 20, False, 5.0, 1, 2],
        [3.33, 10, True, 5.0, 2, 2],
    ]


def test_one_hot_encode_categories():
    features = [
        [1, 10, 'foo', 'bar', -1.0],
        [2, 20, 'bar', 'foo', -1.0],
        [3, 30, 'foo', 'foo', -1.0],
    ]

    roles = ['numerical'] + ['categorical'] * 4

    ctx = _create_context(features, roles)
    autom8.encode_categories(ctx, method='one-hot', only_strings=False)

    assert len(ctx.steps) == 1
    assert ctx.matrix.formulas == [
        'A',
        ['equals[10]', 'B'], ['equals[20]', 'B'], ['equals[30]', 'B'],
        ['equals[foo]', 'C'], ['equals[bar]', 'C'],
        ['equals[bar]', 'D'], ['equals[foo]', 'D'],
        ['equals[-1.0]', 'E'],
    ]

    assert ctx.matrix.tolist() == [
        [1, 1, 0, 0, 1, 0, 1, 0, 1],
        [2, 0, 1, 0, 0, 1, 0, 1, 1],
        [3, 0, 0, 1, 1, 0, 0, 1, 1],
    ]

    # Try replaying it on the original data.
    assert _playback(ctx, roles, features) == ctx.matrix.tolist()

    # Try a case where all the values are expected.
    m2 = _playback(ctx, roles, [
        [4, 30, 'bar', 'bar', -1.0],
        [5, 10, 'foo', 'foo', -1.0],
        [6, 20, 'bar', 'bar', -1.0],
    ])
    assert m2 == [
        [4, 0, 0, 1, 0, 1, 1, 0, 1],
        [5, 1, 0, 0, 1, 0, 0, 1, 1],
        [6, 0, 1, 0, 0, 1, 1, 0, 1],
    ]

    # Now try a case where some of the values are unexpected.
    m3 = _playback(ctx, roles, [
        [4, 33, 'foo', 'bar', -1.0],
        [5, 20, 'baz', 'foo', -1.0],
        [6, 10, 'bar', 'zim', -1.1],
    ])
    assert m3 == [
        [4, 0, 0, 0, 1, 0, 1, 0, 1],
        [5, 0, 1, 0, 0, 0, 0, 1, 1],
        [6, 1, 0, 0, 0, 1, 0, 0, 0],
    ]

    # Now try a case where some of the columns have the wrong type.
    m4 = _playback(ctx, roles, [
        [4, True, 'foo', 7, 'A'],
        [5, True, 'baz', 8, 'B'],
        [6, False, 'bar', 9, 'C'],
    ])
    assert m4 == [
        [4, 0, 0, 0, 1, 0, 0, 0, 0],
        [5, 0, 0, 0, 0, 0, 0, 0, 0],
        [6, 0, 0, 0, 0, 1, 0, 0, 0],
    ]


def test_ordinal_encode_categories_when_something_goes_wrong():
    import autom8.categories

    features = [
        [1, 10, 'foo', 'bar', -1.0],
        [2, 20, 'bar', 'foo', -1.0],
        [3, -1, 'foo', 'foo', -1.0],
    ]

    roles = ['numerical'] + ['categorical'] * 4

    ctx = _create_context(features, roles)
    autom8.encode_categories(ctx, method='ordinal', only_strings=False)
    encoder = ctx.steps[0].args[0]

    matrix = _create_matrix(features, roles)
    acc = autom8.Accumulator()
    plc = PlaybackContext(matrix, receiver=acc)

    # For now, just monkey-patch in a "steps" list. (This is pretty terrible.)
    ctx.steps = []

    # Break the encoder so that our function will raise an exception.
    encoder.transform = None

    autom8.categories.encode(plc, encoder, [1, 2, 3, 4])
    assert ctx.matrix.formulas == plc.matrix.formulas
    assert plc.matrix.tolist() == [
        [1, 0, 0, 0, 0],
        [2, 0, 0, 0, 0],
        [3, 0, 0, 0, 0],
    ]
    assert len(acc.warnings) == 1


def test_one_hot_encode_categories_when_something_goes_wrong():
    import autom8.categories

    features = [
        [1, 10, 'foo', 'bar', -1.0],
        [2, 20, 'bar', 'foo', -1.0],
        [3, -1, 'foo', 'foo', -1.0],
    ]

    roles = ['numerical'] + ['categorical'] * 4

    matrix = _create_matrix(features, roles)
    ctx = _create_context(features, roles)

    autom8.encode_categories(ctx, method='one-hot', only_strings=False)
    encoder = ctx.steps[0].args[0]

    acc = autom8.Accumulator()
    plc = PlaybackContext(matrix, receiver=acc)

    # As in the previous test, just monkey-patch in a "steps" list.
    # (Again, this is pretty terrible.)
    ctx.steps = []

    # Break the encoder so that our function will raise an exception.
    encoder.transform = None

    autom8.categories.encode(plc, encoder, [1, 2, 3, 4])
    assert ctx.matrix.formulas == plc.matrix.formulas
    assert plc.matrix.tolist() == [
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    assert len(acc.warnings) == 1


def test_encode_text():
    features = [
        ['foo and bar or baz', 'foo'],
        ['bar or foo and baz', 'bar'],
        ['baz or bar and foo', 'baz'],
        ['foo and baz or bar', 'foo bar'],
        ['foo or baz and bar', 'foo baz'],
    ]

    roles = ['textual'] * 2

    ctx = _create_context(features, roles)
    autom8.encode_text(ctx)

    assert len(ctx.steps) == 1
    for col in ctx.matrix.columns:
        assert col.formula[0].startswith('frequency[')
        assert col.formula[1:] == ['A', 'B']
        assert col.role == 'encoded'

    # First, try playing it back on the original data.
    assert _playback(ctx, roles, features) == ctx.matrix.tolist()

    def assert_expected_shape(matrix):
        assert len(matrix[0]) == len(ctx.matrix.columns)
        for row in matrix:
            assert all(isinstance(i, float) for i in row)

    # Try playing it back on some new data, but with the same set of words.
    assert_expected_shape(_playback(ctx, roles, [
        ['baz or foo or bar', 'foo'],
        ['bar or baz and foo', 'bar'],
    ]))

    # Try playing it back on some data that has old and new words.
    assert_expected_shape(_playback(ctx, roles, [
        ['foo bar baz zim zam', 'foo'],
        ['bar or baz and foo', 'bar'],
    ]))

    # Try playing it back on all new words.
    mx = _playback(ctx, roles, [['zim zam', 'zug'], ['bim bam', 'blam']])
    assert mx == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    # Try playing it back when the first column doesn't contain text.
    my1 = _playback(ctx, roles, [[True, 'foo bar'], [False, 'foo baz']])

    # Try playing it back when the second column doesn't contain text.
    my2 = _playback(ctx, roles, [['foo bar', 10], ['foo baz', 20]])

    assert_expected_shape(my1)
    assert_expected_shape(my2)
    assert my1 == my2

    # Try playing it back on columns that don't contain text.
    mz1 = _playback(ctx, roles, [[1.0, None], [2.0, None]])
    assert_expected_shape(mz1)
    assert mz1 == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    # Try playing it back on empty strings.
    mz2 = _playback(ctx, roles, [['x', ''], ['', 'y']])
    assert_expected_shape(mz2)
    assert mz2 == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_logarithm_columns():
    features = [[1, 2], [3, 4]]
    roles = ['numerical'] * 2

    ctx = _create_context(features, roles)
    autom8.logarithm_columns(ctx)

    assert ctx.matrix.formulas == ['A', 'B', ['log', 'A'], ['log', 'B']]
    assert ctx.matrix.tolist() == [
        [1, 2, math.log(1), math.log(2)],
        [3, 4, math.log(3), math.log(4)],
    ]

    # Try playing it back on the original data.
    assert _playback(ctx, roles, features) == ctx.matrix.tolist()

    # Try playing it back on some new data.
    m2 = _playback(ctx, roles, [[5, 6], [7, 8]])
    assert m2 == [
        [5, 6, math.log(5), math.log(6)],
        [7, 8, math.log(7), math.log(8)],
    ]

    # Try playing it back on some invalid data.
    m3 = _playback(ctx, roles, [[-55, 66], [77, 0]])
    assert m3 == [
        [-55, 66, 0, math.log(66)],
        [77, 0, math.log(77), 0],
    ]


def test_multiply_colums():
    features = [
        [3, 10, 'foo', 'bar', True, 5.0],
        [2, 20, 'bar', 'foo', False, 6.0],
        [1, 30, 'foo', 'foo', True, 7.0],
    ]

    roles = [
        'numerical', 'numerical', 'categorical',
        'categorical', 'encoded', 'numerical',
    ]

    ctx = _create_context(features, roles)
    autom8.multiply_columns(ctx)

    assert len(ctx.steps) == 1
    assert ctx.matrix.formulas == [
        'A', 'B', 'C', 'D', 'E', 'F',
        ['multiply', 'A', 'B'],
        ['multiply', 'A', 'F'],
        ['multiply', 'B', 'F'],
    ]
    assert ctx.matrix.tolist() == [
        [3, 10, 'foo', 'bar', True, 5.0, 3*10, 3*5.0, 10*5.0],
        [2, 20, 'bar', 'foo', False, 6.0, 2*20, 2*6.0, 20*6.0],
        [1, 30, 'foo', 'foo', True, 7.0, 1*30, 1*7.0, 30*7.0],
    ]

    # Try playing it back on the original data.
    assert _playback(ctx, roles, features) == ctx.matrix.tolist()

    # Try playing it back on some new data.
    m2 = _playback(ctx, roles, [
        [4, 9, 'foo', 'bar', True, 2.0],
        [3, 8, 'bar', 'foo', False, 3.0],
        [2, 7, 'foo', 'foo', True, 4.0],
    ])
    assert m2 == [
        [4, 9, 'foo', 'bar', True, 2.0, 36, 8.0, 18.0],
        [3, 8, 'bar', 'foo', False, 3.0, 24, 9.0, 24.0],
        [2, 7, 'foo', 'foo', True, 4.0, 14, 8.0, 28.0],
    ]

    # Try playing it back on some valid data and some invalid data.
    m3 = _playback(ctx, roles, [
        [4, '9', 'foo', 'bar', True, 2.0],
        [3, '8', 'bar', 'foo', False, 3.0],
        [2, '7', 'foo', 'foo', True, 4.0],
    ])
    assert m3 == [
        [4, '9', 'foo', 'bar', True, 2.0, 36.0, 8.0, 18.0],
        [3, '8', 'bar', 'foo', False, 3.0, 24.0, 9.0, 24.0],
        [2, '7', 'foo', 'foo', True, 4.0, 14.0, 8.0, 28.0],
    ]


def test_scale_columns():
    features = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
    roles = ['numerical'] * 2

    ctx = _create_context(features, roles)
    autom8.scale_columns(ctx)

    assert len(ctx.steps) == 1
    assert ctx.matrix.formulas == [['scale', 'A'], ['scale', 'B']]
    assert ctx.matrix.tolist() == [
        [-5/9, -5/9],
        [-3/9, -3/9],
        [-1/9, -1/9],
        [1/9, 1/9],
        [3/9, 3/9],
        [5/9, 5/9],
    ]

    # Try playing it back on the original data.
    assert _playback(ctx, roles, features) == ctx.matrix.tolist()

    m2 = _playback(ctx, roles, [[-100, 200], [100, 200]])
    assert m2 == [[-106/9, 193/9], [94/9, 193/9]]


def test_square_columns():
    features = [
        [1, 2, 'x', -1, 3],
        [4, 5, 'y', -2, 6],
        [7, 8, 'z', -3, 9],
    ]

    roles = ['numerical', 'numerical', 'categorical', 'encoded', 'numerical']

    ctx = _create_context(features, roles)
    autom8.square_columns(ctx)

    assert len(ctx.steps) == 1
    assert ctx.matrix.formulas == [
        'A', 'B', 'C', 'D', 'E', ['square', 'A'], ['square', 'B'], ['square', 'E'],
    ]
    assert ctx.matrix.tolist() == [
        [1, 2, 'x', -1, 3, 1*1, 2*2, 3*3],
        [4, 5, 'y', -2, 6, 4*4, 5*5, 6*6],
        [7, 8, 'z', -3, 9, 7*7, 8*8, 9*9],
    ]

    # Try playing it back on the original data.
    assert _playback(ctx, roles, features) == ctx.matrix.tolist()


def test_sqrt_columns():
    features = [[1, 2], [3, 4]]
    roles = ['numerical'] * 2

    ctx = _create_context(features, roles)
    autom8.sqrt_columns(ctx)

    assert ctx.matrix.formulas == [
        'A', 'B', ['square-root', 'A'], ['square-root', 'B'],
    ]
    assert ctx.matrix.tolist() == [
        [1, 2, math.sqrt(1), math.sqrt(2)],
        [3, 4, math.sqrt(3), math.sqrt(4)],
    ]

    # Try playing it back on the original data.
    assert _playback(ctx, roles, features) == ctx.matrix.tolist()

    # Try playing it back on some new data.
    m2 = _playback(ctx, roles, [[5, 6], [7, 8]])
    assert m2 == [
        [5, 6, math.sqrt(5), math.sqrt(6)],
        [7, 8, math.sqrt(7), math.sqrt(8)],
    ]

    # Try playing it back on some invalid data.
    m3 = _playback(ctx, roles, [[-55, 66], [77, -88]])
    assert m3 == [
        [-55, 66, 0, math.sqrt(66)],
        [77, -88, math.sqrt(77), 0],
    ]


def _column_names(dataset):
    num_cols = len(dataset[0])
    return [chr(65 + i) for i in range(num_cols)]


def _create_matrix(features, roles):
    return autom8.create_matrix(
        dataset=features,
        column_names=_column_names(features),
        column_roles=roles,
    )


def _create_context(features, roles):
    # Add a column of labels. It's required by create_context.
    dataset = [row + [0] for row in features]
    num_cols = len(dataset[0])

    return autom8.create_context(
        dataset=dataset,
        column_names=_column_names(dataset),
        column_roles=roles + ['numerical'],
    )


def _playback(fitted, roles, features, receiver=None):
    if receiver is None:
        receiver = autom8.Accumulator()

    matrix = _create_matrix(features, roles)
    ctx = PlaybackContext(matrix, receiver=receiver)
    playback(fitted.steps, ctx)
    assert fitted.matrix.formulas == ctx.matrix.formulas
    return ctx.matrix.tolist()
