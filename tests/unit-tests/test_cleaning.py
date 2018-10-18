import unittest
import logging

import autom8
from autom8.pipeline import PipelineContext
from autom8.preprocessors import playback


class TestCleaning(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_matrix_with_unexpected_value(self):
        dataset = [
            ['A', 'B', 'C'],
            [1, 2, ()],
            [3, 4, {}],
            [5, 6, object()],
        ]
        acc = autom8.Accumulator()
        matrix = autom8.create_matrix(_add_labels(dataset), observer=acc)
        ctx = autom8.create_training_context(matrix, observer=acc)

        autom8.clean_dataset(ctx)
        self.assertEqual(len(acc.warnings), 1)
        self.assertTrue('Dropping column' in acc.warnings[0])
        self.assertTrue('contain booleans, numbers' in acc.warnings[0])
        self.assertEqual(ctx.matrix.tolist(), [
            ['A', 'B'], [1, 2], [3, 4], [5, 6]
        ])

        vectors = [['A', 'B', 'C'], [1, 2, 'foo'], [3, 4, 'bar']]
        matrix = autom8.create_matrix(vectors, observer=acc)
        out = PipelineContext(matrix, observer=acc)
        playback(ctx.steps, out)
        self.assertEqual(out.matrix.tolist(), [['A', 'B'], [1, 2], [3, 4]])

    def test_primitives_with_object_dtype(self):
        dataset = [
            ['A', 'B', 'C'],
            [True, 1.1, 2],
            [False, 3.1, 4],
            [True, 5.1, 6],
        ]

        acc = autom8.Accumulator()
        matrix = autom8.create_matrix(_add_labels(dataset), observer=acc)
        for col in matrix.columns:
            col.values = col.values.astype(object)

        ctx = autom8.create_training_context(matrix, observer=acc)
        autom8.clean_dataset(ctx)

        dtypes = [c.dtype for c in ctx.matrix.columns]
        self.assertEqual(dtypes[0], bool)
        self.assertEqual(dtypes[1], float)
        self.assertEqual(dtypes[2], int)

        vectors = [['A', 'B', 'C'], [1, 2, 3.0], [0, 4, 5.0], [1, False, 6.9]]
        matrix = autom8.create_matrix(vectors, observer=acc)
        out = PipelineContext(matrix, observer=acc)
        playback(ctx.steps, out)
        self.assertEqual(out.matrix.tolist(), [
            ['A', 'B', 'C'], [True, 2.0, 3], [False, 4.0, 5], [True, 0.0, 6]
        ])

        dtypes = [c.dtype for c in out.matrix.columns]
        self.assertEqual(dtypes[0], bool)
        self.assertEqual(dtypes[1], float)
        self.assertEqual(dtypes[2], int)

        vectors = [['A', 'B', 'C'], ['1', '2', None], ['', None, ()]]
        matrix = autom8.create_matrix(vectors, observer=acc)
        out = PipelineContext(matrix, observer=acc)
        playback(ctx.steps, out)

        # Just use repr to avoid having to fart around with nan.
        self.assertEqual(repr(out.matrix.tolist()),
            "[['A', 'B', 'C'], [True, 2.0, 0], [False, nan, 0]]"
        )

    def test_column_with_all_none(self):
        dataset = [
            ['A', 'B', 'C'],
            [True, None, 2],
            [False, None, 4],
            [True, None, 6],
        ]

        acc = autom8.Accumulator()
        matrix = autom8.create_matrix(_add_labels(dataset), observer=acc)
        ctx = autom8.create_training_context(matrix, observer=acc)

        autom8.clean_dataset(ctx)
        self.assertEqual(len(acc.warnings), 1)
        self.assertTrue('Dropping column' in acc.warnings[0])

        self.assertEqual(ctx.matrix.tolist(), [
            ['A', 'C'], [True, 2], [False, 4], [True, 6],
        ])

        vectors = [['A', 'B', 'C'], [1, 2, 'foo'], [3, 4, 'bar']]
        matrix = autom8.create_matrix(vectors, observer=acc)
        out = PipelineContext(matrix, observer=acc)
        playback(ctx.steps, out)
        self.assertEqual(out.matrix.tolist(), [['A', 'C'], [1, 'foo'], [3, 'bar']])

    def test_columns_with_numbers_as_strings(self):
        dataset = [
            ['A', 'B', 'C'],
            ['1.1', '$4', 7],
            ['2.2', '$5', 8],
            ['3.3', '6%', 9],
        ]

        acc = autom8.Accumulator()
        matrix = autom8.create_matrix(_add_labels(dataset), observer=acc)
        ctx = autom8.create_training_context(matrix, observer=acc)

        autom8.clean_dataset(ctx)
        self.assertEqual(len(acc.warnings), 0)
        self.assertEqual(len(ctx.steps), 2)

        self.assertEqual(ctx.matrix.tolist(), [
            ['A', 'B', 'C'], [1.1, 4, 7], [2.2, 5, 8], [3.3, 6, 9]
        ])

        vectors = [['A', 'B', 'C'], [1, '2%', 'foo'], ['3', 4.0, 'bar']]
        matrix = autom8.create_matrix(vectors, observer=acc)
        out = PipelineContext(matrix, observer=acc)
        playback(ctx.steps, out)
        self.assertEqual(out.matrix.tolist(), [
            ['A', 'B', 'C'], [1, 2, 'foo'], [3, 4, 'bar']
        ])
        self.assertEqual(out.matrix.columns[0].dtype, int)
        self.assertEqual(out.matrix.columns[1].dtype, float)

    def test_column_of_all_strings(self):
        dataset = [
            ['A', 'B'],
            ['1', 2],
            ['3', 4],
            ['n', 0],
        ]

        acc = autom8.Accumulator()
        matrix = autom8.create_matrix(_add_labels(dataset), observer=acc)
        ctx = autom8.create_training_context(matrix, observer=acc)

        autom8.clean_dataset(ctx)
        self.assertEqual(len(acc.warnings), 0)
        self.assertEqual(ctx.steps, [])
        self.assertEqual(ctx.matrix.tolist(), [
            ['A', 'B'], ['1', 2], ['3', 4], ['n', 0],
        ])

    def test_column_of_all_strings_and_none_values(self):
        dataset = [
            ['A', 'B'],
            ['1', 2],
            ['foo', 4],
            [None, 0],
        ]

        acc = autom8.Accumulator()
        matrix = autom8.create_matrix(_add_labels(dataset), observer=acc)
        ctx = autom8.create_training_context(matrix, observer=acc)

        autom8.clean_dataset(ctx)
        self.assertEqual(len(acc.warnings), 0)
        self.assertEqual(len(ctx.steps), 1)
        self.assertEqual(ctx.matrix.tolist(), [
            ['A', 'B'], ['1', 2], ['foo', 4], ['', 0],
        ])

        vectors = [['A', 'B'], [None, 'bar'], ['baz', None]]
        matrix = autom8.create_matrix(vectors, observer=acc)
        out = PipelineContext(matrix, observer=acc)
        playback(ctx.steps, out)
        self.assertEqual(out.matrix.tolist(), [
            ['A', 'B'], ['', 'bar'], ['baz', None]
        ])

    def test_column_of_ints_and_floats(self):
        dataset = [
            ['A', 'B'],
            [1, 3.3],
            [2.2, 4],
            [None, None],
        ]

        acc = autom8.Accumulator()
        matrix = autom8.create_matrix(_add_labels(dataset), observer=acc)
        ctx = autom8.create_training_context(matrix, observer=acc)

        autom8.clean_dataset(ctx)

        self.assertEqual(len(ctx.steps), 4)
        self.assertEqual(len(acc.warnings), 2)
        self.assertEqual(ctx.matrix.tolist(), [
            ['A', 'PRESENT (A)', 'B', 'PRESENT (B)'],
            [1.0, True, 3.3, True],
            [2.2, True, 4.0, True],
            [0.0, False, 0.0, False],
        ])

        vectors = [['A', 'B'], [None, 10], [20.0, None], [30, 40]]
        matrix = autom8.create_matrix(vectors, observer=acc)
        out = PipelineContext(matrix, observer=acc)
        playback(ctx.steps, out)
        self.assertEqual(out.matrix.tolist(), [
            ['A', 'PRESENT (A)', 'B', 'PRESENT (B)'],
            [0.0, False, 10.0, True],
            [20.0, True, 0.0, False],
            [30.0, True, 40.0, True],
        ])

        self.assertEqual(out.matrix.columns[0].dtype, float)
        self.assertEqual(out.matrix.columns[2].dtype, float)

    def test_columns_with_some_empty_strings(self):
        dataset = [
            ['A', 'B', 'C'],
            [True, 1.1, 20],
            ['', 2.2, 30],
            [False, '', 40],
            [False, 3.3, ''],
            ['', 4.4, ''],
        ]

        acc = autom8.Accumulator()
        matrix = autom8.create_matrix(_add_labels(dataset), observer=acc)
        ctx = autom8.create_training_context(matrix, observer=acc)

        autom8.clean_dataset(ctx)

        self.assertEqual(len(ctx.steps), 6)
        self.assertEqual(len(acc.warnings), 3)
        self.assertEqual(ctx.matrix.tolist(), [
            ['A', 'PRESENT (A)', 'B', 'PRESENT (B)', 'C', 'PRESENT (C)'],
            [True, True, 1.1, True, 20, True],
            [False, False, 2.2, True, 30, True],
            [False, True, 0.0, False, 40, True],
            [False, True, 3.3, True, 0, False],
            [False, False, 4.4, True, 0, False],
        ])

        vectors = [['A', 'B', 'C'], ['', 5.5, ''], [True, '', 50]]
        matrix = autom8.create_matrix(vectors, observer=acc)
        out = PipelineContext(matrix, observer=acc)
        playback(ctx.steps, out)
        self.assertEqual(out.matrix.tolist(), [
            ['A', 'PRESENT (A)', 'B', 'PRESENT (B)', 'C', 'PRESENT (C)'],
            [False, False, 5.5, True, 0, False],
            [True, True, 0.0, False, 50, True],
        ])

    def test_mixed_up_columns_with_strings_and_numbers(self):
        # bool, float, int, str
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
        matrix = autom8.create_matrix(_add_labels(dataset), observer=acc)
        ctx = autom8.create_training_context(matrix, observer=acc)

        autom8.clean_dataset(ctx)

        self.assertEqual(len(ctx.steps), 6)
        self.assertEqual(acc.warnings, [])
        self.assertEqual(ctx.matrix.tolist(), [
            ['NUMBERS (A)', 'STRINGS (A)', 'NUMBERS (B)', 'STRINGS (B)'],
            [1.0, '', 0.0, 'foo'],
            [1.1, '', 30.0, ''],
            [20.0, '', 4.4, ''],
            [0.0, 'bar', 0.0, ''],
            [0.0, '', 0.0, 'baz'],
            [50.0, '', 0.0, 'fiz'],
            [0.0, '', 1.0, ''],
        ])

        vectors = [['A', 'B'], [False, 'buz'], ['zim', 10], [2, None]]
        matrix = autom8.create_matrix(vectors, observer=acc)
        out = PipelineContext(matrix, observer=acc)
        playback(ctx.steps, out)
        self.assertEqual(out.matrix.tolist(), [
            ['NUMBERS (A)', 'STRINGS (A)', 'NUMBERS (B)', 'STRINGS (B)'],
            [0.0, '', 0.0, 'buz'],
            [0.0, 'zim', 10.0, ''],
            [2.0, '', 0.0, ''],
        ])


def _add_labels(dataset):
    return [i + ['<label>'] for i in dataset]
