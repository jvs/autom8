def better_candidate(a, b):
    """Returns the candidate that autom8 thinks is better."""

    # If they're the same object, then exit early.
    if a is b:
        return a

    # If one is None, then return the other one.
    if a is None or b is None:
        return a or b

    # MAY: Take that class of estimator into consideration. (Maybe autom8
    # should prefer some estimator classes over other ones.)

    get_score = lambda m: m.get('r2_score', m.get('f1_score', 0))
    score_a = get_score(a.test.metrics)
    score_b = get_score(b.test.metrics)

    # If one score is at least 5% better than the other, then it wins.
    if score_a >= (score_b * 1.05): return a
    if score_b >= (score_a * 1.05): return b

    # If we have the "mean_absolute_error" metric, then see if there's a big
    # difference between the two.
    error_a = a.test.metrics.get('mean_absolute_error', 0)
    error_b = b.test.metrics.get('mean_absolute_error', 0)

    # If one is at least 10% smaller than the other, then it wins.
    if error_a <= (error_b * 0.9): return a
    if error_b <= (error_a * 0.9): return b

    # All right, see if we have the "accuracy_score" metric. If one score is
    # at least 10% better than the other, then it wins.
    acc_a = a.test.metrics.get('accuracy_score', 0)
    acc_b = b.test.metrics.get('accuracy_score', 0)
    if acc_a >= (acc_b * 1.10): return a
    if acc_b >= (acc_a * 1.10): return b

    # Well now we're in a gray area. See if one is simpler than the other.
    feat_a = _count_features(a)
    feat_b = _count_features(b)

    # If one is much simpler than the other, then it wins.
    if feat_a <= (feat_b - 3): return a
    if feat_b <= (feat_a - 3): return b

    # Well we tried! Just let the better score win.
    return a if score_a > score_b else b


def _count_features(candidate):
    num_cols = len(candidate.formulas)

    est = candidate.pipeline.estimator
    weights = getattr(est, 'coef_', getattr(est, 'feature_importances_', None))

    # If we don't have any weights, then just set them all to one.
    if weights is None:
        weights = [1 for _ in candidate.formulas]

    # If we don't have enough weights for some reason, then our estimator might
    # be messed up. In this case, just assume all the columns matter.
    if num_cols != len(weights):
        return num_cols

    for formula, weight in zip(candidate.formulas, weights):
        # Don't count any column whose importance is very close to zero.
        if abs(weight) <= 0.0001:
            num_cols -= 1

        # Only count "frequency" columns as one column.
        if isinstance(formula, list) and f[0].starswith('frequency['):
            num_cols -= 1
            saw_text = True

    # If we saw text columns, add one back to represent them all.
    if saw_text:
        num_cols += 1

    return num_cols
