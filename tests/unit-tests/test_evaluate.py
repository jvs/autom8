from pytest import approx
import numpy as np
import sklearn.linear_model
import autom8


def test_evaluate_pipeline():
    acc = autom8.Accumulator()
    inputs = [
        [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16],
    ]
    dataset = [i + [i[0] + i[1]] for i in inputs]
    ctx = autom8.create_context(dataset, receiver=acc)

    # For now, just hack in the test_indices that we want.
    ctx.test_indices = [2, 5]

    autom8.add_column_of_ones(ctx)
    ctx << sklearn.linear_model.LinearRegression()
    assert len(acc.pipelines) == 1

    pipeline, report = acc.pipelines[0]
    assert report.train.metrics['r2_score'] == 1.0
    assert report.test.metrics['r2_score'] == 1.0

    assert np.allclose(
        report.train.predictions,
        np.array([1+2, 3+4, 7+8, 9+10, 13+14, 15+16]),
    )

    assert np.allclose(
        report.test.predictions,
        np.array([5+6, 11+12]),
    )

    # Make sure that we can literalize the report.
    obj = autom8.literalize(report)
    assert isinstance(obj, dict)
    assert 'train' in obj and 'test' in obj
    assert obj['test']['metrics']['r2_score'] == 1.0
    assert obj['test']['metrics']['r2_score'] == 1.0

    # Try using the pipeline to make some predictions.
    result = pipeline.run([[17, 18], [19, 20], [21, 22]], receiver=acc)

    assert np.allclose(result.predictions, np.array([17+18, 19+20, 21+22]))
    assert result.probabilities is None
    assert not acc.warnings

    # Make sure that we can literalize the result.
    obj = autom8.literalize(result)
    assert isinstance(obj, dict)
    assert 'predictions' in obj and 'probabilities' in obj
    assert obj['predictions'] == approx([17+18, 19+20, 21+22])
    assert obj['probabilities'] is None
