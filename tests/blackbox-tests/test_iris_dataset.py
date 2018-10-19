import autom8
import datasets


def test_iris_dataset():
    acc = datasets.fit('iris.csv')

    # Assert that we at least got 7 pipelines.
    assert len(acc.pipelines) >= 7

    # Make sure each report has a valid f1_score.
    for _, report in acc.pipelines:
        s1 = report.train.metrics['f1_score']
        s2 = report.test.metrics['f1_score']
        assert isinstance(s1.tolist(), float)
        assert isinstance(s2.tolist(), float)
        assert 0 <= s1 <= 1.0
        assert 0 <= s2 <= 1.0

    # Make sure each report has valid predictions.
    for _, report in acc.pipelines:
        for section in [report.train, report.test]:
            for pred in section.predictions:
                assert pred in {'setosa', 'versicolor', 'virginica'}

    # Assert that the best test score is better than 0.6.
    best = max(i.test.metrics['f1_score'] for _, i in acc.pipelines)
    assert best > 0.6

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
        pred = pipeline.run(vectors, receiver=tmp)
        assert len(pred.predictions) == 3
        for name in pred.predictions:
            assert name in {'setosa', 'versicolor', 'virginica'}
        assert not tmp.warnings
