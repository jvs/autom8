import os
import autom8


def load_dataset(name):
    testdir = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(testdir, 'datasets', name)
    dataset = autom8.load_csv(path)
    acc = autom8.Accumulator()
    return autom8.create_training_context(dataset, observer=acc)
