import json
import os
import warnings
import autom8


def load(name):
    testdir = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(testdir, 'datasets', name)
    return autom8.load_csv(path)


def fit(name):
    dataset = load(name)
    acc = autom8.Accumulator()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        autom8.fit(dataset, receiver=acc)

    return acc


def check_classifier_reports(acc, valid_labels):
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
            for label in section.predictions:
                assert label in valid_labels

            # Make sure we have the same number of predictions as probabilities.
            if section.probabilities is not None:
                assert len(section.predictions) == len(section.probabilities)

    # Make sure that we can literalize and encode each report.
    for _, report in acc.pipelines:
        json.dumps(autom8.literalize(report))


def check_classifier_predictions(acc, valid_labels, vectors):
    for pipeline, _ in acc.pipelines:
        tmp = autom8.Accumulator()
        pred = pipeline.run(vectors, receiver=tmp)
        assert not tmp.warnings

        # Make sure that we can literalize and encode the predictions.
        json.dumps(autom8.literalize(pred))

        assert len(pred.predictions) == len(vectors) - 1
        for label in pred.predictions:
            assert label in valid_labels

        if pred.probabilities is None:
            continue

        assert len(pred.probabilities) == len(pred.predictions)
        for probs in pred.probabilities:
            # Make sure we have between one and three pairs.
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
