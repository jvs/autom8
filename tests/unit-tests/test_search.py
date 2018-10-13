import unittest
import random
import warnings
import numpy as np

import autom8


class TestSearch(unittest.TestCase):
    def test_simple_search(self):
        X = np.arange(0.0, 1, 0.01).reshape(-1, 1)
        y = np.sin(2 * np.pi * X).ravel()

        acc = autom8.Accumulator()
        matrix = autom8.create_matrix(X, observer=acc)

        count = len(y)
        test_indices = random.sample(range(count), int(count * 0.2) or 1)

        ctx = autom8.create_training_context(
            matrix, y, test_indices, 'regression', observer=acc
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            autom8.search(ctx)

        # Assert that we at least got 10 pipelines.
        self.assertGreater(len(acc.pipelines), 10)

        # Make sure they all have an r2_score.
        for pipeline, report in acc.pipelines:
            s1 = report.train.metrics['r2_score']
            s2 = report.test.metrics['r2_score']
            self.assertIsInstance(s1, float)
            self.assertIsInstance(s2, float)

        # Make sure the best scores are at least 0.5.
        best_train = max(r.train.metrics['r2_score'] for _, r in acc.pipelines)
        best_test = max(r.test.metrics['r2_score'] for _, r in acc.pipelines)
        self.assertGreater(best_train, 0.5)
        self.assertGreater(best_test, 0.5)
