import autom8
import datasets


def test_summarize_boston():
    dataset = datasets.load('boston.csv')
    summaries = autom8.summarize(dataset)
    for s in summaries:
        print()
        print(s)
