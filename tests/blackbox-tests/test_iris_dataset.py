import datasets


def test_iris_dataset():
    acc = datasets.fit('iris.csv')

    # Assert that we at least got 7 pipelines.
    assert len(acc.pipelines) >= 7

    valid_labels = {'setosa', 'versicolor', 'virginica'}
    datasets.check_classifier_reports(acc, valid_labels)

    # Assert that the best test score is better than 0.6.
    best = max(i.test.metrics['f1_score'] for _, i in acc.pipelines)
    assert best > 0.6

    # Make sure each pipeline can make predictions.
    datasets.check_classifier_predictions(acc, valid_labels, [
        [
            'sepal length (cm)', 'sepal width (cm)',
            'petal length (cm)', 'petal width (cm)',
        ],
        [6.21, 2.21, 4.41, 1.53],
        [5.78, 2.78, 5.09, 2.37],
        [5.12, 3.54, 1.39, 0.21],
    ])
