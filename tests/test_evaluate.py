import unittest
import numpy as np
import sklearn.linear_model
from context import autom8, Accumulator


class TestEvaluate(unittest.TestCase):
    def test_evaluate_pipeline(self):
        acc = Accumulator()
        matrix = autom8.create_matrix([
            [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16],
        ])
        labels = np.array([1+2, 3+4, 5+6, 7+8, 9+10, 11+12, 13+14, 15+16])
        test_indices = [2, 5]
        ctx = autom8.create_training_context(
            matrix, labels, test_indices, 'regression', observer=acc
        )
        autom8.add_column_of_ones(ctx)

        ctx << sklearn.linear_model.LinearRegression()
        self.assertEqual(len(acc.pipelines), 1)

        pipeline, report = acc.pipelines[0]
        self.assertEqual(report.train.metrics['r2_score'], 1.0)
        self.assertEqual(report.test.metrics['r2_score'], 1.0)

        self.assertTrue(np.allclose(
            report.train.predictions,
            np.array([1+2, 3+4, 7+8, 9+10, 13+14, 15+16]),
        ))

        self.assertTrue(np.allclose(
            report.test.predictions,
            np.array([5+6, 11+12]),
        ))

        # Try using the pipeline to make some predictions.
        result = pipeline.run([[17, 18], [19, 20], [21, 22]], observer=acc)

        self.assertTrue(np.allclose(
            result.predictions,
            np.array([17+18, 19+20, 21+22]),
        ))
        self.assertIsNone(result.probabilities)
        self.assertEqual(acc.warnings, [])
