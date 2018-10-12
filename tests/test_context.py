import unittest
import numpy as np
from context import autom8, Accumulator
from autom8.context import PredictingContext, create_training_context


class TestContext(unittest.TestCase):
    def test_is_training_property(self):
        matrix = autom8.create_matrix([[1]])
        c1 = autom8.create_training_context(matrix, [], [], 'regression')
        c2 = PredictingContext(matrix, Accumulator())
        self.assertTrue(c1.is_training)
        self.assertFalse(c2.is_training)

    def test_require_training_context(self):
        matrix = autom8.create_matrix([[1]])
        c1 = autom8.create_training_context(matrix, [], [], 'regression')
        c2 = PredictingContext(matrix, Accumulator())

        # This should not raise an exception.
        c1.require_training_context()

        # But this should raise one.
        with self.assertRaisesRegex(autom8.Autom8Exception, 'Expected.*TrainingContext'):
            c2.require_training_context()


class TestTrainingContext(unittest.TestCase):
    def test_training_and_testing_data(self):
        matrix = autom8.create_matrix([
            [1, 5, True, 9],
            [2, 6, False, 10],
            [3, 7, False, 11],
            [4, 8, True, 12],
        ])
        labels = np.array([10, 20, 30, 40])
        ctx = autom8.create_training_context(matrix, labels, [1, 3], 'regression')
        m1, a1 = ctx.testing_data()
        m2, a2 = ctx.training_data()
        self.assertTrue(np.array_equal(a1, [20, 40]))
        self.assertTrue(np.array_equal(a2, [10, 30]))
        self.assertTrue(np.array_equal(m1, [
            [2, 6, False, 10],
            [4, 8, True, 12],
        ]))
        self.assertTrue(np.array_equal(m2, [
            [1, 5, True, 9],
            [3, 7, False, 11],
        ]))

    def test_sandbox(self):
        matrix = autom8.create_matrix({
            'rows': [
                [1, 5, True, 9],
                [2, 6, False, 10],
                [3, 7, False, 11],
                [4, 8, True, 12],
            ],
            'schema': [
                {'name': 'A', 'role': 'numerical'},
                {'name': 'B', 'role': 'numerical'},
                {'name': 'C', 'role': 'encoded'},
                {'name': 'D', 'role': 'numerical'},
            ]
        })

        ctx = autom8.create_training_context(matrix, [], [], 'regression')
        autom8.add_column_of_ones(ctx)

        self.assertEqual(len(ctx.preprocessors), 1)
        self.assertEqual(len(ctx.matrix.columns), 4+1)
        self.assertEqual(ctx.matrix.tolist()[1:], [
            [1, 5, True, 9, 1],
            [2, 6, False, 10, 1],
            [3, 7, False, 11, 1],
            [4, 8, True, 12, 1],
        ])

        with ctx.sandbox():
            autom8.multiply_columns(ctx)
            self.assertEqual(len(ctx.preprocessors), 2)
            self.assertEqual(len(ctx.matrix.columns), 4+1+3)
            self.assertEqual(ctx.matrix.tolist()[1:], [
                [1, 5, True, 9, 1, 1*5, 1*9, 5*9],
                [2, 6, False, 10, 1, 2*6, 2*10, 6*10],
                [3, 7, False, 11, 1, 3*7, 3*11, 7*11],
                [4, 8, True, 12, 1, 4*8, 4*12, 8*12],
            ])

        # Now check that the context has been restored to its previous state.
        self.assertEqual(len(ctx.preprocessors), 1)
        self.assertEqual(len(ctx.matrix.columns), 4+1)
        self.assertEqual(ctx.matrix.tolist()[1:], [
            [1, 5, True, 9, 1],
            [2, 6, False, 10, 1],
            [3, 7, False, 11, 1],
            [4, 8, True, 12, 1],
        ])
