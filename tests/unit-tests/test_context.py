import pytest
import numpy as np

import autom8
from autom8.pipeline import PlaybackContext


def test_is_recording_property():
    matrix = autom8.create_matrix([[1, 2]])
    c1 = autom8.create_context(matrix)
    c2 = PlaybackContext(matrix, autom8.Accumulator())
    assert c1.is_recording
    assert not c2.is_recording
    assert hasattr(c1, 'receiver')
    assert hasattr(c2, 'receiver')


def test_planner_decorator():
    matrix = autom8.create_matrix([[1, 1], [2, 2]])
    c1 = autom8.create_context(matrix)
    c2 = PlaybackContext(matrix, autom8.Accumulator())

    # This should not raise an exception.
    autom8.drop_duplicate_columns(c1)

    # But this should raise one.
    with pytest.raises(autom8.Autom8Exception) as excinfo:
        autom8.drop_duplicate_columns(c2)
    excinfo.match('Expected.*RecordingContext')


def test_invalid_contexts():
    with pytest.raises(autom8.Autom8Exception) as excinfo:
        autom8.create_context([])
    excinfo.match('Expected.*dataset')

    with pytest.raises(autom8.Autom8Exception) as excinfo:
        autom8.create_context([[1], [2], [3]])
    excinfo.match('Expected.*dataset')

    with pytest.raises(autom8.Autom8Exception) as excinfo:
        autom8.create_context(
            dataset=[['A', 'B'], [1, 2]],
            target_column='C',
        )
    excinfo.match('Expected.*target_column')

    with pytest.raises(autom8.Autom8Exception) as excinfo:
        autom8.create_context(
            dataset=[['A', 'B'], [1, 2]],
            target_column=object(),
        )
    excinfo.match('Expected.*target_column')

    with pytest.raises(autom8.Autom8Exception) as excinfo:
        autom8.create_context(
            dataset=[['A', 'B'], [1, 2]],
            target_column=10,
        )
    excinfo.match('Expected.*target_column')

    with pytest.raises(autom8.Autom8Exception) as excinfo:
        autom8.create_context(
            dataset=[['A', 'B'], [1, 2]],
            problem_type='classify',
        )
    excinfo.match('Expected.*problem_type')

    with pytest.raises(autom8.Autom8Exception) as excinfo:
        autom8.create_context(
            dataset=[['A', 'B'], [1, 2]],
            test_ratio=1.2,
        )
    excinfo.match('Expected.*test_ratio')


def test_training_and_testing_data():
    dataset = autom8.create_matrix([
        [1, 5, True, 9, 10],
        [2, 6, False, 10, 20],
        [3, 7, False, 11, 30],
        [4, 8, True, 12, 40],
    ])
    ctx = autom8.create_context(dataset)

    # For now, just hack in the test_indices that we want.
    ctx.test_indices = [1, 3]

    m1, a1 = ctx.testing_data()
    m2, a2 = ctx.training_data()
    assert np.array_equal(a1, [20, 40])
    assert np.array_equal(a2, [10, 30])
    assert np.array_equal(m1, [
        [2, 6, False, 10],
        [4, 8, True, 12],
    ])
    assert np.array_equal(m2, [
        [1, 5, True, 9],
        [3, 7, False, 11],
    ])


def test_sandbox():
    ctx = autom8.create_context(
        dataset=[
            [1, 5, True, 9, 10],
            [2, 6, False, 10, 20],
            [3, 7, False, 11, 30],
            [4, 8, True, 12, 40],
        ],
        column_names=['A', 'B', 'C', 'D', 'E'],
        column_roles=['numerical'] * 2 + ['encoded'] + ['numerical'] * 2,
    )
    autom8.add_column_of_ones(ctx)

    assert len(ctx.steps) == 1
    assert len(ctx.matrix.columns) == 4+1
    assert ctx.matrix.tolist() == [
        [1, 5, True, 9, 1],
        [2, 6, False, 10, 1],
        [3, 7, False, 11, 1],
        [4, 8, True, 12, 1],
    ]

    with ctx.sandbox():
        autom8.multiply_columns(ctx)
        assert len(ctx.steps) == 2
        assert len(ctx.matrix.columns) == 4+1+3
        assert ctx.matrix.tolist() == [
            [1, 5, True, 9, 1, 1*5, 1*9, 5*9],
            [2, 6, False, 10, 1, 2*6, 2*10, 6*10],
            [3, 7, False, 11, 1, 3*7, 3*11, 7*11],
            [4, 8, True, 12, 1, 4*8, 4*12, 8*12],
        ]

    # Now check that the context has been restored to its previous state.
    assert len(ctx.steps) == 1
    assert len(ctx.matrix.columns) == 4+1
    assert ctx.matrix.tolist() == [
        [1, 5, True, 9, 1],
        [2, 6, False, 10, 1],
        [3, 7, False, 11, 1],
        [4, 8, True, 12, 1],
    ]
