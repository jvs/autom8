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

        # Make sure they all have an r2_score.
        for pipeline, report in acc.pipelines:
            s1 = report.train.metrics['r2_score']
            s2 = report.test.metrics['r2_score']
            self.assertIsInstance(s1, float)
            self.assertIsInstance(s2, float)
