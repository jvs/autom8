import autom8
import datasets


def test_badges_dataset():
    acc = datasets.run('badges.csv')

    # Assert that we at least got 7 candidates.
    assert len(acc.candidates) >= 7

    valid_labels = {'+', '-'}
    datasets.check_classifier_candidates(acc, valid_labels)

    # Make sure each pipeline can make predictions.
    datasets.check_classifier_predictions(acc, valid_labels, [
        ['name'],
        ['Foo Barry Baz'],
        ['Zim Chimney Zoo'],
        ['Flimmy Flam Sam'],
    ])
