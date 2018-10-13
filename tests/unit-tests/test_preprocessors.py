import unittest
import math
import numpy as np

import autom8
from autom8.pipeline import PipelineContext
from autom8.preprocessors import playback


class TestPreprocessors(unittest.TestCase):
    def test_add_column_of_ones(self):
        training = [['a', 'b'], ['c', 'd']]
        schema = [
            {'name': 'A', 'role': 'categorical'},
            {'name': 'B', 'role': 'categorical'},
        ]
        new_headers = ['A', 'B', 'CONSTANT']
        matrix = autom8.create_matrix({'rows': training, 'schema': schema})
        ctx = autom8.create_training_context(matrix, [], [], 'regression')
        autom8.add_column_of_ones(ctx)

        self.assertEqual(len(ctx.steps), 1)
        self.assertEqual(ctx.matrix.tolist(), [
            new_headers, ['a', 'b', 1], ['c', 'd', 1],
        ])

        self.assertEqual(_playback(ctx, schema, training), ctx.matrix.tolist())

        m1 = _playback(ctx, schema, [['e', 'f'], ['g', 'h']])
        self.assertEqual(m1, [
            new_headers,
            ['e', 'f', 1],
            ['g', 'h', 1],
        ])

    def test_binarize_fractions(self):
        training = [[1, 0.2], [0.3, 4]]
        schema = [
            {'name': 'A', 'role': 'numerical'},
            {'name': 'B', 'role': 'numerical'},
        ]
        new_headers = [
            'A', 'B', 'A IS A FRACTION', 'B IS A FRACTION',
        ]
        matrix = autom8.create_matrix({'rows': training, 'schema': schema})
        ctx = autom8.create_training_context(matrix, [], [], 'regression')
        autom8.binarize_fractions(ctx)

        self.assertEqual(len(ctx.steps), 1)
        self.assertEqual(ctx.matrix.tolist(), [
            new_headers,
            [1, 0.2, False, True],
            [0.3, 4, True, False],
        ])

        self.assertEqual(_playback(ctx, schema, training), ctx.matrix.tolist())

        m1 = _playback(ctx, schema, [[0.99, 1.1], [1.0, -0.1], [0.0, -1.2]])
        self.assertEqual(m1, [
            new_headers,
            [0.99, 1.1, True, False],
            [1.0, -0.1, False, True],
            [0.0, -1.2, True, False],
        ])

    def test_binarize_signs(self):
        training = [[1, -2], [-3, 4], [0, 5]]
        schema = [
            {'name': 'A', 'role': 'numerical'},
            {'name': 'B', 'role': 'numerical'},
        ]
        new_headers = [
            'A', 'B', 'A IS POSITIVE', 'B IS POSITIVE',
        ]
        matrix = autom8.create_matrix({'rows': training, 'schema': schema})
        ctx = autom8.create_training_context(matrix, [], [], 'regression')
        autom8.binarize_signs(ctx)

        self.assertEqual(len(ctx.steps), 1)
        self.assertEqual(ctx.matrix.tolist(), [
            new_headers,
            [1, -2, True, False],
            [-3, 4, False, True],
            [0, 5, False, True],
        ])

        self.assertEqual(_playback(ctx, schema, training), ctx.matrix.tolist())

        m1 = _playback(ctx, schema, [[10, -0.1], [0.1, -10]])
        self.assertEqual(m1, [
            new_headers,
            [10, -0.1, True, False],
            [0.1, -10, True, False],
        ])

    def test_divide_columns(self):
        training = [
            [1, 4, 'a', 7.0],
            [2, 5, 'b', 8.0],
            [3, 6, 'c', 9.0],
        ]
        schema = [
            {'name': 'A', 'role': 'numerical'},
            {'name': 'B', 'role': 'numerical'},
            {'name': 'C', 'role': 'categorical'},
            {'name': 'D', 'role': 'numerical'},
        ]
        new_headers = [
            'A', 'B', 'C', 'D',
            'A DIVIDED BY B', 'A DIVIDED BY D', 'B DIVIDED BY A',
            'B DIVIDED BY D', 'D DIVIDED BY A', 'D DIVIDED BY B',
        ]
        matrix = autom8.create_matrix({'rows': training, 'schema': schema})
        ctx = autom8.create_training_context(matrix, [], [], 'regression')
        autom8.divide_columns(ctx)

        self.assertEqual(len(ctx.steps), 1)
        self.assertEqual(ctx.matrix.tolist(), [
            new_headers,
            [1, 4, 'a', 7.0, 1 / 4, 1 / 7, 4 / 1, 4 / 7, 7 / 1, 7 / 4],
            [2, 5, 'b', 8.0, 2 / 5, 2 / 8, 5 / 2, 5 / 8, 8 / 2, 8 / 5],
            [3, 6, 'c', 9.0, 3 / 6, 3 / 9, 6 / 3, 6 / 9, 9 / 3, 9 / 6],
        ])

        # Try playing it back on the original data.
        m1 = _playback(ctx, schema, training)
        self.assertEqual(m1, ctx.matrix.tolist())

        # Try playing it back on some new data.
        m2 = _playback(ctx, schema, [
            [1, 2, 'x', 3.0],
            [4, 5, 'y', 6.0],
            [7, 8, 'z', 9.0],
        ])
        self.assertEqual(m2, [
            new_headers,
            [1, 2, 'x', 3.0, 1 / 2, 1 / 3, 2 / 1, 2 / 3, 3 / 1, 3 / 2],
            [4, 5, 'y', 6.0, 4 / 5, 4 / 6, 5 / 4, 5 / 6, 6 / 4, 6 / 5],
            [7, 8, 'z', 9.0, 7 / 8, 7 / 9, 8 / 7, 8 / 9, 9 / 7, 9 / 8],
        ])

        # Try playing it back when a zero appears as a denominator.
        m3 = _playback(ctx, schema, [
            [1, 3, 'x', 0.0],
            [2, 4, 'y', 5.0],
        ])
        self.assertEqual(m3, [
            new_headers,
            [1, 3, 'x', 0.0, 1 / 3, 0, 3 / 1, 0, 0, 0],
            [2, 4, 'y', 5.0, 2 / 4, 2 / 5, 4 / 2, 4 / 5, 5 / 2, 5 / 4],
        ])

    def test_drop_duplicate_columns(self):
        training = [
            [np.nan, np.nan, 1, 'a', np.nan, 'a'],
            [np.inf, np.inf, 2, 'b', np.inf, 'b'],
            [0, 0, 3, 'c', 0, 'c'],
            [1, 1, 4, 'd', 1, 'd'],
        ]
        schema = [
            {'name': 'A', 'role': 'numerical'},
            {'name': 'B', 'role': 'numerical'},
            {'name': 'C', 'role': 'numerical'},
            {'name': 'D', 'role': 'categorical'},
            {'name': 'E', 'role': 'numerical'},
            {'name': 'F', 'role': 'categorical'},
        ]
        new_headers = ['A', 'C', 'D']
        matrix = autom8.create_matrix({'rows': training, 'schema': schema})
        ctx = autom8.create_training_context(matrix, [], [], 'regression')
        autom8.drop_duplicate_columns(ctx)

        def _nan(x):
            return 'nan' if isinstance(x, float) and np.isnan(x) else x

        def _clean(a):
            return [[_nan(i) for i in row] for row in a]

        self.assertEqual(len(ctx.steps), 1)
        self.assertEqual(_clean(ctx.matrix.tolist()), _clean([
            new_headers,
            [np.nan, 1, 'a'],
            [np.inf, 2, 'b'],
            [0.0, 3, 'c'],
            [1.0, 4, 'd'],
        ]))

        m1 = _playback(ctx, schema, training)
        self.assertEqual(_clean(m1), _clean(ctx.matrix.tolist()))

    def test_ordinal_encode_categories(self):
        training = [
            [1.1, 10, 'foo', 'bar', True, 5.0],
            [2.2, 20, 'bar', 'foo', False, 5.0],
            [3.3, 30, 'foo', 'foo', True, 5.0],
        ]
        schema = [
            {'name': 'A', 'role': 'numerical'},
            {'name': 'B', 'role': 'categorical'},
            {'name': 'C', 'role': 'categorical'},
            {'name': 'D', 'role': 'categorical'},
            {'name': 'E', 'role': 'encoded'},
            {'name': 'F', 'role': 'categorical'},
        ]
        new_headers = [
            'A',
            'E',
            'ENCODED B',
            'ENCODED C',
            'ENCODED D',
            'ENCODED F',
        ]

        matrix = autom8.create_matrix({'rows': training, 'schema': schema})
        ctx = autom8.create_training_context(matrix, [], [], 'regression')

        restore_ctx = ctx.copy()
        autom8.encode_categories(ctx, method='ordinal', only_strings=False)

        self.assertEqual(len(ctx.steps), 1)
        self.assertEqual(ctx.matrix.tolist(), [
            new_headers,
            [1.1, True, 1, 1, 1, 1],
            [2.2, False, 2, 2, 2, 1],
            [3.3, True, 3, 1, 2, 1],
        ])

        # First, try playing it back on the original data.
        m1 = _playback(ctx, schema, training)
        self.assertEqual(m1, ctx.matrix.tolist())

        # Try a case where all the values are expected.
        m2 = _playback(ctx, schema, [
            [1.11, 30, 'bar', 'bar', False, 5.0],
            [2.22, 20, 'foo', 'foo', False, 5.0],
            [3.33, 10, 'bar', 'foo', True, 5.0],
        ])
        self.assertEqual(m2, [
            new_headers,
            [1.11, False, 3, 2, 1, 1],
            [2.22, False, 2, 1, 2, 1],
            [3.33, True, 1, 2, 2, 1],
        ])

        # Now try a case where some of the values are unexpected.
        m3 = _playback(ctx, schema, [
            [1.12, 33, 'baz', 'bar', False, 5.0],
            [2.23, 22, 'foo', 'boo', False, 5.0],
            [3.34, 11, 'baz', 'boo', True, 6.0],
        ])
        self.assertEqual(m3, [
            new_headers,
            [1.12, False, 0, 0, 1, 1],
            [2.23, False, 0, 1, 0, 1],
            [3.34, True, 0, 0, 0, 0],
        ])

        # Now try a case where one of the columns has the wrong type.
        m4 = _playback(ctx, schema, [
            [1.17, 20, 3, 'foo', False, 5.0],
            [2.26, 10, 4, 'bar', True, 5.0],
            [3.35, 30, 5, 'baz', True, 5.0],
        ])
        self.assertEqual(m4, [
            new_headers,
            [1.17, False, 2, 0, 2, 1],
            [2.26, True, 1, 0, 1, 1],
            [3.35, True, 3, 0, 0, 1],
        ])

        # Now try just encoding the string columns.
        new_headers = ['A', 'B', 'E', 'F', 'ENCODED C', 'ENCODED D']
        ctx = restore_ctx.copy()
        autom8.encode_categories(ctx, method='ordinal', only_strings=True)

        # Once again, try playing it back on the original data.
        m5 = _playback(ctx, schema, training)
        self.assertEqual(m5, ctx.matrix.tolist())

        # Try a case where all the values are expected.
        m6 = _playback(ctx, schema, [
            [1.11, 30, 'bar', 'bar', False, 5.0],
            [2.22, 20, 'foo', 'foo', False, 5.0],
            [3.33, 10, 'bar', 'foo', True, 5.0],
        ])
        self.assertEqual(m6, [
            new_headers,
            [1.11, 30, False, 5.0, 2, 1],
            [2.22, 20, False, 5.0, 1, 2],
            [3.33, 10, True, 5.0, 2, 2],
        ])

    def test_one_hot_encode_categories(self):
        training = [
            [1, 10, 'foo', 'bar', -1.0],
            [2, 20, 'bar', 'foo', -1.0],
            [3, 30, 'foo', 'foo', -1.0],
        ]
        schema = [
            {'name': 'A', 'role': 'numerical'},
            {'name': 'B', 'role': 'categorical'},
            {'name': 'C', 'role': 'categorical'},
            {'name': 'D', 'role': 'categorical'},
            {'name': 'E', 'role': 'categorical'},
        ]
        matrix = autom8.create_matrix({'rows': training, 'schema': schema})
        ctx = autom8.create_training_context(matrix, [], [], 'regression')

        restore_ctx = ctx.copy()
        autom8.encode_categories(ctx, method='one-hot', only_strings=False)

        new_headers = [
            'A',
            'B = 10', 'B = 20', 'B = 30', 'B = -1',
            'C = foo', 'C = bar', 'C = -1',
            'D = bar', 'D = foo', 'D = -1',
            'E = -1.0', 'E = -1',
        ]

        self.assertEqual(len(ctx.steps), 1)
        self.assertEqual(ctx.matrix.tolist(), [
            new_headers,
            [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            [2, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
            [3, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
        ])

        # Try playing it back on the original data.
        m1 = _playback(ctx, schema, training)
        self.assertEqual(m1, ctx.matrix.tolist())

        # Try a case where all the values are expected.
        m2 = _playback(ctx, schema, [
            [4, 30, 'bar', 'bar', -1.0],
            [5, 10, 'foo', 'foo', -1.0],
            [6, 20, 'bar', 'bar', -1.0],
        ])
        self.assertEqual(m2, [
            new_headers,
            [4, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
            [5, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
            [6, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
        ])

        # Now try a case where some of the values are unexpected.
        m3 = _playback(ctx, schema, [
            [4, 33, 'foo', 'bar', -1.0],
            [5, 20, 'baz', 'foo', -1.0],
            [6, 10, 'bar', 'zim', -1.1],
        ])
        self.assertEqual(m3, [
            new_headers,
            [4, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            [5, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
            [6, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        ])

        # Now try a case where some of the columns have the wrong type.
        m4 = _playback(ctx, schema, [
            [4, True, 'foo', 7, 'A'],
            [5, True, 'baz', 8, 'B'],
            [6, False, 'bar', 9, 'C'],
        ])
        self.assertEqual(m4, [
            new_headers,
            [4, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [6, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        ])

    def test_ordinal_encode_categories_when_something_goes_wrong(self):
        import autom8.categories

        training = [
            [1, 10, 'foo', 'bar', -1.0],
            [2, 20, 'bar', 'foo', -1.0],
            [3, -1, 'foo', 'foo', -1.0],
        ]
        schema = [
            {'name': 'A', 'role': 'numerical'},
            {'name': 'B', 'role': 'categorical'},
            {'name': 'C', 'role': 'categorical'},
            {'name': 'D', 'role': 'categorical'},
            {'name': 'E', 'role': 'categorical'},
        ]
        matrix = autom8.create_matrix({'rows': training, 'schema': schema})
        ctx = autom8.create_training_context(matrix, [], [], 'regression')

        autom8.encode_categories(ctx, method='ordinal', only_strings=False)
        encoder = ctx.steps[0].args[0]

        acc = autom8.Accumulator()
        pip = PipelineContext(matrix, observer=acc)

        # For now, just monkey-patch in a "steps" list.
        # (This is pretty terrible.)
        ctx.steps = []

        # Break the encoder so that our function will raise an exception.
        encoder.transform = None

        autom8.categories.encode(pip, encoder, [1, 2, 3, 4])
        self.assertEqual(
            [c.name for c in ctx.matrix.columns],
            [c.name for c in pip.matrix.columns],
        )
        self.assertEqual(pip.matrix.tolist(), [
            ['A', 'ENCODED B', 'ENCODED C', 'ENCODED D', 'ENCODED E'],
            [1, 0, 0, 0, 0],
            [2, 0, 0, 0, 0],
            [3, 0, 0, 0, 0],
        ])
        self.assertEqual(len(acc.warnings), 1)

    def test_one_hot_encode_categories_when_something_goes_wrong(self):
        import autom8.categories

        training = [
            [1, 10, 'foo', 'bar', -1.0],
            [2, 20, 'bar', 'foo', -1.0],
            [3, -1, 'foo', 'foo', -1.0],
        ]
        schema = [
            {'name': 'A', 'role': 'numerical'},
            {'name': 'B', 'role': 'categorical'},
            {'name': 'C', 'role': 'categorical'},
            {'name': 'D', 'role': 'categorical'},
            {'name': 'E', 'role': 'categorical'},
        ]
        matrix = autom8.create_matrix({'rows': training, 'schema': schema})
        ctx = autom8.create_training_context(matrix, [], [], 'regression')

        autom8.encode_categories(ctx, method='one-hot', only_strings=False)
        encoder = ctx.steps[0].args[0]

        acc = autom8.Accumulator()
        pip = PipelineContext(matrix, observer=acc)

        # As in the previous test, just monkey-patch in a "steps" list.
        # (Again, this is pretty terrible.)
        ctx.steps = []

        # Break the encoder so that our function will raise an exception.
        encoder.transform = None

        autom8.categories.encode(pip, encoder, [1, 2, 3, 4])
        self.assertEqual(
            [c.name for c in ctx.matrix.columns],
            [c.name for c in pip.matrix.columns],
        )
        self.assertEqual(pip.matrix.tolist(), [
            [
                'A',
                'B = 10', 'B = 20', 'B = -1', 'B = -1 [2]',
                'C = foo', 'C = bar', 'C = -1',
                'D = bar', 'D = foo', 'D = -1',
                'E = -1.0', 'E = -1',
            ],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])
        self.assertEqual(len(acc.warnings), 1)

    def test_encode_text(self):
        training = [
            ['foo and bar or baz', 'foo'],
            ['bar or foo and baz', 'bar'],
            ['baz or bar and foo', 'baz'],
            ['foo and baz or bar', 'foo bar'],
            ['foo or baz and bar', 'foo baz'],
        ]
        schema = [
            {'name': 'A', 'role': 'textual'},
            {'name': 'B', 'role': 'textual'},
        ]
        matrix = autom8.create_matrix({'rows': training, 'schema': schema})
        ctx = autom8.create_training_context(matrix, [], [], 'regression')
        autom8.encode_text(ctx)
        self.assertEqual(len(ctx.steps), 1)
        self.assertEqual(ctx.matrix.columns[0].name, 'ENCODED TEXT 1')
        self.assertTrue(all(i.name.startswith('ENCODED TEXT ') for i in ctx.matrix.columns))
        self.assertTrue(all(i.role == 'encoded' for i in ctx.matrix.columns))

        # First, try playing it back on the original data.
        m1 = _playback(ctx, schema, training)
        self.assertEqual(m1, ctx.matrix.tolist())

        def assert_expected_shape(matrix):
            self.assertEqual(matrix[0], ctx.matrix.tolist()[0])
            for row in matrix[1:]:
                self.assertTrue(all(isinstance(i, float) for i in row))

        # Try playing it back on some new data, but with the same set of words.
        assert_expected_shape(_playback(ctx, schema, [
            ['baz or foo or bar', 'foo'],
            ['bar or baz and foo', 'bar'],
        ]))

        # Try playing it back on some data that has old and new words.
        assert_expected_shape(_playback(ctx, schema, [
            ['foo bar baz zim zam', 'foo'],
            ['bar or baz and foo', 'bar'],
        ]))

        # Try playing it back on all new words.
        mx = _playback(ctx, schema, [
            ['zim zam', 'zug'],
            ['bim bam', 'blam'],
        ])
        self.assertEqual(mx, [
            ['ENCODED TEXT 1', 'ENCODED TEXT 2', 'ENCODED TEXT 3'],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])

        # Try playing it back when the first column doesn't contain text.
        my1 = _playback(ctx, schema, [
            [True, 'foo bar'],
            [False, 'foo baz']
        ])

        # Try playing it back when the second column doesn't contain text.
        my2 = _playback(ctx, schema, [
            ['foo bar', 10],
            ['foo baz', 20]
        ])

        assert_expected_shape(my1)
        assert_expected_shape(my2)
        self.assertEqual(my1, my2)

        # Try playing it back on columns that don't contain text.
        mz = _playback(ctx, schema, [[1.0, None], [2.0, None]])
        assert_expected_shape(mz)
        self.assertEqual(mz, [
            ['ENCODED TEXT 1', 'ENCODED TEXT 2', 'ENCODED TEXT 3'],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])

        # Try playing it back on empty strings.
        mz = _playback(ctx, schema, [['', ''], ['', '']])
        assert_expected_shape(mz)
        self.assertEqual(mz, [
            ['ENCODED TEXT 1', 'ENCODED TEXT 2', 'ENCODED TEXT 3'],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])

    def test_logarithm_columns(self):
        training = [[1, 2], [3, 4]]
        schema = [
            {'name': 'A', 'role': 'numerical'},
            {'name': 'B', 'role': 'numerical'},
        ]
        new_headers = ['A', 'B', 'LOG A', 'LOG B']
        matrix = autom8.create_matrix({'rows': training, 'schema': schema})

        ctx = autom8.create_training_context(matrix, [], [], 'regression')
        autom8.logarithm_columns(ctx)

        self.assertEqual(ctx.matrix.tolist(), [
            new_headers,
            [1, 2, math.log(1), math.log(2)],
            [3, 4, math.log(3), math.log(4)],
        ])

        # Try playing it back on the original data.
        m1 = _playback(ctx, schema, training)
        self.assertEqual(m1, ctx.matrix.tolist())

        # Try playing it back on some new data.
        m2 = _playback(ctx, schema, [[5, 6], [7, 8]])
        self.assertEqual(m2, [
            new_headers,
            [5, 6, math.log(5), math.log(6)],
            [7, 8, math.log(7), math.log(8)],
        ])

        # Try playing it back on some invalid data.
        m3 = _playback(ctx, schema, [[-55, 66], [77, 0]])
        self.assertEqual(m3, [
            new_headers,
            [-55, 66, 0, math.log(66)],
            [77, 0, math.log(77), 0],
        ])

    def test_multiply_colums(self):
        training = [
            [3, 10, 'foo', 'bar', True, 5.0],
            [2, 20, 'bar', 'foo', False, 6.0],
            [1, 30, 'foo', 'foo', True, 7.0],
        ]
        schema = [
            {'name': 'A', 'role': 'numerical'},
            {'name': 'B', 'role': 'numerical'},
            {'name': 'C', 'role': 'categorical'},
            {'name': 'D', 'role': 'categorical'},
            {'name': 'E', 'role': 'encoded'},
            {'name': 'F', 'role': 'numerical'},
        ]
        new_headers = [
            'A', 'B', 'C', 'D', 'E', 'F', 'A TIMES B', 'A TIMES F', 'B TIMES F',
        ]
        matrix = autom8.create_matrix({'rows': training, 'schema': schema})
        ctx = autom8.create_training_context(matrix, [], [], 'regression')
        autom8.multiply_columns(ctx)

        self.assertEqual(len(ctx.steps), 1)
        self.assertEqual(ctx.matrix.tolist(), [
            new_headers,
            [3, 10, 'foo', 'bar', True, 5.0, 3*10, 3*5.0, 10*5.0],
            [2, 20, 'bar', 'foo', False, 6.0, 2*20, 2*6.0, 20*6.0],
            [1, 30, 'foo', 'foo', True, 7.0, 1*30, 1*7.0, 30*7.0],
        ])

        # Try playing it back on the original data.
        m1 = _playback(ctx, schema, training)
        self.assertEqual(m1, ctx.matrix.tolist())

        # Try playing it back on some new data.
        m2 = _playback(ctx, schema, [
            [4, 9, 'foo', 'bar', True, 2.0],
            [3, 8, 'bar', 'foo', False, 3.0],
            [2, 7, 'foo', 'foo', True, 4.0],
        ])
        self.assertEqual(m2, [
            new_headers,
            [4, 9, 'foo', 'bar', True, 2.0, 36, 8.0, 18.0],
            [3, 8, 'bar', 'foo', False, 3.0, 24, 9.0, 24.0],
            [2, 7, 'foo', 'foo', True, 4.0, 14, 8.0, 28.0],
        ])

        # Try playing it back on some valid data and some invalid data.
        m3 = _playback(ctx, schema, [
            [4, '9', 'foo', 'bar', True, 2.0],
            [3, '8', 'bar', 'foo', False, 3.0],
            [2, '7', 'foo', 'foo', True, 4.0],
        ])
        self.assertEqual(m3, [
            new_headers,
            [4, '9', 'foo', 'bar', True, 2.0, 36.0, 8.0, 18.0],
            [3, '8', 'bar', 'foo', False, 3.0, 24.0, 9.0, 24.0],
            [2, '7', 'foo', 'foo', True, 4.0, 14.0, 8.0, 28.0],
        ])

    def test_scale_columns(self):
        training = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
        schema = [
            {'name': 'A', 'role': 'numerical'},
            {'name': 'B', 'role': 'numerical'},
        ]
        new_headers = ['SCALED (A)', 'SCALED (B)']
        matrix = autom8.create_matrix({'rows': training, 'schema': schema})
        ctx = autom8.create_training_context(matrix, [], [], 'regression')
        autom8.scale_columns(ctx)

        self.assertEqual(len(ctx.steps), 1)
        self.assertEqual(ctx.matrix.tolist(), [
            new_headers,
            [-5/9, -5/9],
            [-3/9, -3/9],
            [-1/9, -1/9],
            [1/9, 1/9],
            [3/9, 3/9],
            [5/9, 5/9],
        ])

        # Try playing it back on the original data.
        m1 = _playback(ctx, schema, training)
        self.assertEqual(m1, ctx.matrix.tolist())

        m2 = _playback(ctx, schema, [[-100, 200], [100, 200]])
        self.assertEqual(m2[0], new_headers)

    def test_square_columns(self):
        training = [
            [1, 2, 'x', -1, 3],
            [4, 5, 'y', -2, 6],
            [7, 8, 'z', -3, 9],
        ]
        schema = [
            {'name': 'A', 'role': 'numerical'},
            {'name': 'B', 'role': 'numerical'},
            {'name': 'C', 'role': 'categorical'},
            {'name': 'D', 'role': 'encoded'},
            {'name': 'E', 'role': 'numerical'},
        ]
        new_headers = [
            'A', 'B', 'C', 'D', 'E', 'A SQUARED', 'B SQUARED', 'E SQUARED',
        ]
        matrix = autom8.create_matrix({'rows': training, 'schema': schema})
        ctx = autom8.create_training_context(matrix, [], [], 'regression')
        autom8.square_columns(ctx)

        self.assertEqual(len(ctx.steps), 1)
        self.assertEqual(ctx.matrix.tolist(), [
            new_headers,
            [1, 2, 'x', -1, 3, 1*1, 2*2, 3*3],
            [4, 5, 'y', -2, 6, 4*4, 5*5, 6*6],
            [7, 8, 'z', -3, 9, 7*7, 8*8, 9*9],
        ])

        # Try playing it back on the original data.
        m1 = _playback(ctx, schema, training)
        self.assertEqual(m1, ctx.matrix.tolist())

    def test_sqrt_columns(self):
        training = [[1, 2], [3, 4]]
        schema = [
            {'name': 'A', 'role': 'numerical'},
            {'name': 'B', 'role': 'numerical'},
        ]
        new_headers = ['A', 'B', 'SQRT A', 'SQRT B']
        matrix = autom8.create_matrix({'rows': training, 'schema': schema})

        ctx = autom8.create_training_context(matrix, [], [], 'regression')
        autom8.sqrt_columns(ctx)

        self.assertEqual(ctx.matrix.tolist(), [
            new_headers,
            [1, 2, math.sqrt(1), math.sqrt(2)],
            [3, 4, math.sqrt(3), math.sqrt(4)],
        ])

        # Try playing it back on the original data.
        m1 = _playback(ctx, schema, training)
        self.assertEqual(m1, ctx.matrix.tolist())

        # Try playing it back on some new data.
        m2 = _playback(ctx, schema, [[5, 6], [7, 8]])
        self.assertEqual(m2, [
            new_headers,
            [5, 6, math.sqrt(5), math.sqrt(6)],
            [7, 8, math.sqrt(7), math.sqrt(8)],
        ])

        # Try playing it back on some invalid data.
        m3 = _playback(ctx, schema, [[-55, 66], [77, -88]])
        self.assertEqual(m3, [
            new_headers,
            [-55, 66, 0, math.sqrt(66)],
            [77, -88, math.sqrt(77), 0],
        ])


def _playback(training_context, schema, rows, observer=None):
    if observer is None:
        observer = autom8.Accumulator()
    matrix = autom8.create_matrix({'rows': rows, 'schema': schema})
    ctx = PipelineContext(matrix, observer=observer)
    playback(training_context.steps, ctx)
    return ctx.matrix.tolist()
