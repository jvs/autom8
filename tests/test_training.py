import unittest
import numpy as np
import sklearn.linear_model
from context import autom8


class TestTraining(unittest.TestCase):
    def test_train_function(self):
        report = autom8.create_matrix([
            [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16],
        ])
        labels = np.array([1+2, 3+4, 5+6, 7+8, 9+10, 11+12, 13+14, 15+16])
        test_indices = [2, 5]
        ctx = autom8.create_training_context(
            report.matrix, labels, test_indices, 'regression'
        )
        autom8.add_column_of_ones(ctx)
        result = autom8.train(ctx, sklearn.linear_model.LinearRegression())

        self.assertEqual(result.train['r2_score'], 1.0)
        self.assertEqual(result.test['r2_score'], 1.0)

        self.assertTrue(np.allclose(
            result.train['predictions'],
            np.array([1+2, 3+4, 7+8, 9+10, 13+14, 15+16]),
        ))

        self.assertTrue(np.allclose(
            result.test['predictions'],
            np.array([5+6, 11+12]),
        ))
