import datasets


def test_wine_dataset():
    acc = datasets.run('wine.csv')

    # Assert that we at least got 7 reports.
    assert len(acc.reports) >= 7

    valid_labels = {'class_0', 'class_1', 'class_2'}
    datasets.check_classifier_reports(acc, valid_labels)

    # Assert that the best test score is better than 0.6.
    best = max(i.test.metrics['f1_score'] for i in acc.reports)
    assert best > 0.6

    # Make sure each pipeline can make predictions.
    datasets.check_classifier_predictions(acc, valid_labels, [
        [
            'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
            'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
            'proanthocyanins', 'color_intensity', 'hue',
            'od280/od315_of_diluted_wines', 'proline',
        ],
        [
            13.5, 3.2, 2.7, 23.5, 97.1, 1.6, 0.5,
            0.5, 0.55, 4.4, 0.9, 2.0, 520.0,
        ],
        [
            13.6, 1.7, 2.4, 19.1, 106.0, 2.9, 3.2,
            0.22, 1.95, 6.9, 1.1, 2.9, 1515.0,
        ],
        [
            12.3, 1.7, 2.1, 19.0, 80.0, 1.7, 2.0,
            0.4, 1.63, 3.4, 1.0, 3.2, 510.0,
        ],
    ])
