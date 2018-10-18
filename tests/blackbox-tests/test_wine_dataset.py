import warnings
import autom8
from datasets import load_dataset

def test_wine_dataset():
    ctx = load_dataset('wine.csv')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        autom8.search(ctx)

    # Grab our accumulator.
    acc = ctx.observer

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
                assert pred in {'class_0', 'class_1', 'class_2'}

    # Assert that the best test score is better than 0.6.
    best = max(i.test.metrics['f1_score'] for _, i in acc.pipelines)
    assert best > 0.6

    # Make sure each pipeline can make predictions.
    vectors = [
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
    ]
    for pipeline, _ in acc.pipelines:
        tmp = autom8.Accumulator()
        pred = pipeline.run(vectors, observer=tmp)
        assert len(pred.predictions) == 3
        for name in pred.predictions:
            assert name in {'class_0', 'class_1', 'class_2'}
        assert not tmp.warnings
