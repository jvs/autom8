from .receiver import Receiver


def create_selector(selector=None):
    if selector is None:
        return DefaultSelector()

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

    def receive_report(self, report):
        if self.best is None:
            self.best = report
        else:
            self.best = self._compare(self.best, report)


class DefaultSelector(Receiver):
    def __init__(self):
        self._selector = None

    @property
    def best(self):
        return None if self._selector is None else self._selector.best

    def receive_context(self, context):
        is_cls = context.problem_type == 'classification'
        metric = 'f1_score' if is_cls else 'r2_score'
        self._selector = create_selector([metric, 'step_count', 'column_count'])
        self._selector.receive_context(context)

    def receive_report(self, report):
        self._selector.receive_report(report)


def _optimize_metrics(metrics, report1, report2):
    for metric in metrics:
        x1 = _read_metric(report1, metric)
        x2 = _read_metric(report2, metric)

        # If it's not a tie, then return the report with the better score.
        if x1 != x2:
            is_better = x1 > x2 if metric.endswith('_score') else x1 < x2
            return report1 if is_better else report2

    # The reigning champion wins ties.
    return report1


def _read_metric(report, metric):
    if metric == 'column_count':
        return len(report.formulas)
    elif metric == 'step_count':
        return len(report.pipeline.steps)
    else:
        return report.test.metrics[metric]
