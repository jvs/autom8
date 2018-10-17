import unittest
import numpy as np

import autom8
from autom8.pipeline import PipelineContext


class TestContextInterface(unittest.TestCase):
    def test_is_training_property(self):
        matrix = autom8.create_matrix([[1, 2]])
        c1 = autom8.create_training_context(matrix)
        c2 = PipelineContext(matrix, autom8.Accumulator())
        self.assertTrue(c1.is_training)
        self.assertFalse(c2.is_training)
        self.assertTrue(hasattr(c1, 'observer'))
        self.assertTrue(hasattr(c2, 'observer'))

    def test_planner_decorator(self):
        matrix = autom8.create_matrix([[1, 1], [2, 2]])
        c1 = autom8.create_training_context(matrix)
        c2 = PipelineContext(matrix, autom8.Accumulator())

        # This should not raise an exception.
        autom8.drop_duplicate_columns(c1)

        # But this should raise one.
        with self.assertRaisesRegex(autom8.Autom8Exception, 'Expected.*TrainingContext'):
            autom8.drop_duplicate_columns(c2)


class TestTrainingContext(unittest.TestCase):
    def test_copy_method(self):
        # TODO: Make sure that the copied context actually works.
        c1 = autom8.create_training_context([[1, 2]])
        c2 = c1.copy()
        self.assertIsNot(c1, c2)
        self.assertIsNot(c1.matrix, c2.matrix)
        self.assertIsNot(c1.steps, c2.steps)

    def test_training_and_testing_data(self):
        dataset = autom8.create_matrix([
            [1, 5, True, 9, 10],
            [2, 6, False, 10, 20],
            [3, 7, False, 11, 30],
            [4, 8, True, 12, 40],
        ])
        ctx = autom8.create_training_context(dataset)

        # For now, just hack in the test_indices that we want.
        ctx.test_indices = [1, 3]

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
        dataset = {
            'rows': [
                [1, 5, True, 9, 10],
                [2, 6, False, 10, 20],
                [3, 7, False, 11, 30],
                [4, 8, True, 12, 40],
            ],
            'schema': [
                {'name': 'A', 'role': 'numerical'},
                {'name': 'B', 'role': 'numerical'},
                {'name': 'C', 'role': 'encoded'},
                {'name': 'D', 'role': 'numerical'},
                {'name': 'E', 'role': 'numerical'},
            ]
        }

        ctx = autom8.create_training_context(dataset)
        autom8.add_column_of_ones(ctx)

        self.assertEqual(len(ctx.steps), 1)
        self.assertEqual(len(ctx.matrix.columns), 4+1)
        self.assertEqual(ctx.matrix.tolist()[1:], [
            [1, 5, True, 9, 1],
            [2, 6, False, 10, 1],
            [3, 7, False, 11, 1],
            [4, 8, True, 12, 1],
        ])

        with ctx.sandbox():
            autom8.multiply_columns(ctx)
            self.assertEqual(len(ctx.steps), 2)
            self.assertEqual(len(ctx.matrix.columns), 4+1+3)
            self.assertEqual(ctx.matrix.tolist()[1:], [
                [1, 5, True, 9, 1, 1*5, 1*9, 5*9],
                [2, 6, False, 10, 1, 2*6, 2*10, 6*10],
                [3, 7, False, 11, 1, 3*7, 3*11, 7*11],
                [4, 8, True, 12, 1, 4*8, 4*12, 8*12],
            ])

        # Now check that the context has been restored to its previous state.
        self.assertEqual(len(ctx.steps), 1)
        self.assertEqual(len(ctx.matrix.columns), 4+1)
        self.assertEqual(ctx.matrix.tolist()[1:], [
            [1, 5, True, 9, 1],
            [2, 6, False, 10, 1],
            [3, 7, False, 11, 1],
            [4, 8, True, 12, 1],
        ])
