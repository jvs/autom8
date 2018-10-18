import unittest
import warnings
import autom8
from datasets import load_dataset


class TestBostonDataset(unittest.TestCase):
    def test_boston_dataset(self):
        ctx = load_dataset('boston.csv')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            autom8.search(ctx)

        # Grab our accumulator.
        acc = ctx.observer

        # Assert that we at least got 10 pipelines.
        self.assertGreater(len(acc.pipelines), 10)

        # Make sure each report has an r2_score.
        for _, report in acc.pipelines:
            s1 = report.train.metrics['r2_score']
            s2 = report.test.metrics['r2_score']
            self.assertIsInstance(s1, float)
            self.assertIsInstance(s2, float)
            self.assertTrue(s1 <= 1.0)
            self.assertTrue(s2 <= 1.0)

        # Assert that the best test score is better than 0.6.
        best = max(i.test.metrics['r2_score'] for _, i in acc.pipelines)
        self.assertGreater(best, 0.6)

        # Make sure each pipeline can make predictions.
        vectors = [
            [
                'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT',
            ],
            [0.007, 18, 2.3, 0, 0.5, 6.5, 65, 4, 1, 296, 15.4, 396.9, 4.98],
            [0.05, 0, 2.4, 0, 0.5, 7.8, 53, 3, 3, 193, 18, 392.63, 4.45],
        ]
        for pipeline, _ in acc.pipelines:
            tmp = autom8.Accumulator()
            pred = pipeline.run(vectors, observer=tmp)
            self.assertEqual(len(pred.predictions), 2)
            self.assertIsInstance(pred.predictions[0], float)
            self.assertIsInstance(pred.predictions[1], float)
            self.assertEqual(tmp.warnings, [])
