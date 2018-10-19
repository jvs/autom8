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
