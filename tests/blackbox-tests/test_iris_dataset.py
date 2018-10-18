import unittest
import warnings
import autom8
from datasets import load_dataset


class TestIrisDataset(unittest.TestCase):
    def test_iris_dataset(self):
        ctx = load_dataset('iris.csv')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            autom8.search(ctx)

        # Grab our accumulator.
        acc = ctx.observer

        # Assert that we at least got 7 pipelines.
        self.assertGreater(len(acc.pipelines), 7)

        # Make sure each report has a valid f1_score.
        for _, report in acc.pipelines:
            s1 = report.train.metrics['f1_score']
            s2 = report.test.metrics['f1_score']
            self.assertIsInstance(s1, float)
            self.assertIsInstance(s2, float)
            self.assertTrue(0 <= s1 <= 1.0)
            self.assertTrue(0 <= s2 <= 1.0)

        # Make sure each report has valid predictions.
        for _, report in acc.pipelines:
            for section in [report.train, report.test]:
                for pred in section.predictions:
                    self.assertTrue(pred in {'setosa', 'versicolor', 'virginica'})

        # Assert that the best test score is better than 0.6.
        best = max(i.test.metrics['f1_score'] for _, i in acc.pipelines)
        self.assertGreater(best, 0.6)

        # Make sure each pipeline can make predictions.
        vectors = [
            [
                'sepal length (cm)', 'sepal width (cm)',
                'petal length (cm)', 'petal width (cm)',
            ],
            [6.21, 2.21, 4.41, 1.53],
            [5.78, 2.78, 5.09, 2.37],
            [5.12, 3.54, 1.39, 0.21],
        ]
        for pipeline, _ in acc.pipelines:
            tmp = autom8.Accumulator()
            pred = pipeline.run(vectors, observer=tmp)
            self.assertEqual(len(pred.predictions), 3)
            for name in pred.predictions:
                self.assertTrue(name in {'setosa', 'versicolor', 'virginica'})
            self.assertEqual(tmp.warnings, [])
