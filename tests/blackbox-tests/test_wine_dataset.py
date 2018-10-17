import unittest
import warnings
import autom8
from datasets import load_dataset


class TestWineDataset(unittest.TestCase):
    def test_wine_dataset(self):
        ctx = load_dataset('wine.csv')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            autom8.search(ctx)

        # Grab our accumulator.
        acc = ctx.observer

        # Assert that we at least got 7 pipelines.
        self.assertGreater(len(acc.pipelines), 7)
