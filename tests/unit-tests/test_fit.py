import random
import warnings
import numpy as np
import autom8


def test_fit():
    x = np.arange(0.0, 1, 0.01)
    y = np.sin(2 * np.pi * x)

    dataset = np.column_stack([x, y])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pipeline = autom8.fit(dataset)

    assert isinstance(pipeline, autom8.Pipeline)

    result = pipeline.run([[0.5]])
    assert len(result.predictions) == 1
    assert isinstance(result.predictions[0], float)


def test_run():
    x = np.arange(0.0, 1, 0.01)
    y = np.sin(2 * np.pi * x)

    dataset = np.column_stack([x, y])
    acc = autom8.Accumulator()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        autom8.run(dataset, receiver=acc)

    # Assert that we at least got 10 candidates.
    assert len(acc.candidates) >= 10

    # Make sure they all have an r2_score.
    for candidate in acc.candidates:
        s1 = candidate.train.metrics['r2_score']
        s2 = candidate.test.metrics['r2_score']
        assert isinstance(s1, float)
        assert isinstance(s2, float)

    # Make sure the best scores are at least 0.5.
    best_train = max(r.train.metrics['r2_score'] for r in acc.candidates)
    best_test = max(r.test.metrics['r2_score'] for r in acc.candidates)
    assert best_train > 0.5
    assert best_test > 0.5
