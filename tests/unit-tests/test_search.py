import random
import warnings
import numpy as np
import autom8


def test_simple_search():
    x = np.arange(0.0, 1, 0.01)
    y = np.sin(2 * np.pi * x)
    dataset = np.column_stack([x, y])

    acc = autom8.Accumulator()
    ctx = autom8.create_training_context(dataset, receiver=acc)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        autom8.search(ctx)

    # Assert that we at least got 10 pipelines.
    assert len(acc.pipelines) >= 10

    # Make sure they all have an r2_score.
    for pipeline, report in acc.pipelines:
        s1 = report.train.metrics['r2_score']
        s2 = report.test.metrics['r2_score']
        assert isinstance(s1.tolist(), float)
        assert isinstance(s2.tolist(), float)

    # Make sure the best scores are at least 0.5.
    best_train = max(r.train.metrics['r2_score'] for _, r in acc.pipelines)
    best_test = max(r.test.metrics['r2_score'] for _, r in acc.pipelines)
    assert best_train > 0.5
    assert best_test > 0.5
