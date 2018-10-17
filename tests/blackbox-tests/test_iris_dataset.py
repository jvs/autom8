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
