import json
import os
import warnings
import autom8


def load(name):
    testdir = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(testdir, 'datasets', name)
    return autom8.read_csv(path)


def run(name):
    dataset = load(name)
    acc = autom8.Accumulator()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        autom8.run(dataset, receiver=acc)

    check_for_numpy_values(acc)
    return acc


def check_for_numpy_values(acc):
    # Just make sure that json.dumps() doesn't raise exceptions on this stuff.
    json.dumps(acc.test_indices)
    for candidate in acc.candidates:
        json.dumps(candidate.formulas)
        for section in [candidate.train, candidate.test]:
            json.dumps(section.predictions)
            json.dumps(section.probabilities)
            json.dumps(section.metrics)


def check_classifier_candidates(acc, valid_labels):
    # Make sure each candidate has a valid f1_score.
    for candidate in acc.candidates:
        s1 = candidate.train.metrics['f1_score']
        s2 = candidate.test.metrics['f1_score']
        assert isinstance(s1, float)
        assert isinstance(s2, float)
        assert 0 <= s1 <= 1.0
        assert 0 <= s2 <= 1.0

    # Make sure each candidate has valid predictions.
    for candidate in acc.candidates:
        for section in [candidate.train, candidate.test]:
            for label in section.predictions:
                assert label in valid_labels

            # Make sure we have the same number of predictions as probabilities.
            if section.probabilities is not None:
                assert len(section.predictions) == len(section.probabilities)

            # Make sure that we got the extended metrics.
            assert 'normalized_confusion_matrix' in section.metrics
            assert 'precision_recall_fscore_support' in section.metrics


def check_classifier_predictions(acc, valid_labels, vectors):
    for candidate in acc.candidates:
        tmp = autom8.Accumulator()
        pred = candidate.pipeline.run(vectors, receiver=tmp)
        assert not tmp.warnings

        # Just make sure that we can json-encode the predictions.
        json.dumps(pred)

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
