from .compare import better_candidate
from .docstrings import render_docstring
from .receiver import Receiver


@render_docstring
def create_selector(selector=None):
    """"Returns an `autom8.Receiver` that selects the best candidate pipeline.

    Parameters:
        $selector_parameters
    """

    if selector is None:
        return Comparator(lambda a, b: better_candidate(a, b))

    if isinstance(selector, str):
        return create_selector([selector])

    if isinstance(selector, (list, tuple)):
        if all(isinstance(i, str) for i in selector):
            compare = lambda r1, r2: _optimize_metrics(selector, r1, r2)
            return create_selector(compare)

    if callable(selector):
        return Comparator(selector)

    raise TypeError(
        'selector must be None, a string, a list of strings, or a function.'
    )


class Comparator(Receiver):
    def __init__(self, compare):
        self._compare = compare
        self.best = None

    def receive_candidate(self, candidate):
        if self.best is None:
            self.best = candidate
        else:
            self.best = self._compare(self.best, candidate)


def _optimize_metrics(metrics, candidate1, candidate2):
    for metric in metrics:
        x1 = _read_metric(candidate1, metric)
        x2 = _read_metric(candidate2, metric)

        # If it's not a tie, then return the candidate with the better score.
        if x1 != x2:
            is_better = x1 > x2 if metric.endswith('_score') else x1 < x2
            return candidate1 if is_better else candidate2

    # The reigning champion wins ties.
    return candidate1


def _read_metric(candidate, metric):
    if metric == 'column_count':
        return len(candidate.formulas)
    elif metric == 'step_count':
        return len(candidate.pipeline.steps)
    else:
        return candidate.test.metrics[metric]
