import json
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

    valid_labels = {'setosa', 'versicolor', 'virginica'}

    # Make sure each report has valid predictions.
    for _, report in acc.pipelines:
        for section in [report.train, report.test]:
            for label in section.predictions:
                assert label in valid_labels

            # Make sure we have the same number of predictions as probabilities.
            if section.probabilities is not None:
                assert len(section.predictions) == len(section.probabilities)

    # Make sure that we can literalize and encode each report.
    for _, report in acc.pipelines:
        json.dumps(autom8.literalize(report))

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
        assert not tmp.warnings

        # Make sure that we can literalize and encode the predictions.
        json.dumps(autom8.literalize(pred))

        assert len(pred.predictions) == 3
        for label in pred.predictions:
            assert label in valid_labels

        if pred.probabilities is None:
            continue

        assert len(pred.probabilities) == 3
        for probs in pred.probabilities:
            # Make sure we have be one and three pairs.
            assert 1 <= len(probs) <= 3

            # Make sure each pair is a valid label and a valid probability.
            for label, score in probs:
                assert label in valid_labels
                assert 0 < score <= 1

            # Make sure that the probabilities are sorted from highest to lowest.
            scores = [score for _, score in probs]
            assert sorted(scores, reverse=True) == scores

            # Make sure that they don't all add up to something greater than 1.
            0 < sum(score for _, score in probs) <= 1

        # Make sure that the prediction is the first label.
        for label, probs in zip(pred.predictions, pred.probabilities):
            assert label == probs[0][0]
