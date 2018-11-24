import random
import warnings
import numpy as np
import autom8


def test_fit():
    x = np.arange(0.0, 1, 0.01)
    y = np.sin(2 * np.pi * x)

    dataset = np.column_stack([x, y])
    acc = autom8.Accumulator()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        autom8.fit(dataset, receiver=acc)

    # Assert that we at least got 10 reports.
    assert len(acc.reports) >= 10

    # Make sure they all have an r2_score.
    for report in acc.reports:
        s1 = report.train.metrics['r2_score']
        s2 = report.test.metrics['r2_score']
        assert isinstance(s1.tolist(), float)
        assert isinstance(s2.tolist(), float)

    # Make sure the best scores are at least 0.5.
    best_train = max(r.train.metrics['r2_score'] for r in acc.reports)
    best_test = max(r.test.metrics['r2_score'] for r in acc.reports)
    assert best_train > 0.5
    assert best_test > 0.5
