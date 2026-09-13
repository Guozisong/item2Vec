import numpy as np
import pandas as pd
import pytest

from item2vec import inference, training


@pytest.mark.parametrize('weight,expected', [(1.0, 'B'), (0.0, 'C'), (0.7, 'B')])
def test_fusion_changes_ranking_with_weight(weight, expected):
    text = np.array([[1., 0.], [.8, .6], [0., 1.]])
    behavior = np.array([[1., 0., 0.], [0., 1., 0.], [1., 0., 0.]])
    result = inference.rank_items(
        text, {'0': 'A', '1': 'B', '2': 'C'}, [0], 2,
        behavior_vectors=behavior, behavior_order_counts=[50, 50, 50], text_weight=weight,
    )
    assert result.iloc[0].slave_prod_id == expected
    scores = result.set_index('slave_prod_id').similarity
    assert scores['B'] == pytest.approx(weight * .8)
    assert scores['C'] == pytest.approx(1 - weight)


def test_fusion_falls_back_for_either_missing_behavior_and_is_block_invariant():
    text = np.array([[1., 0.], [.8, .6], [0., 1.]])
    behavior = np.array([[1., 0.], [0., 1.], [0., 0.]])
    mapping = {'0': 'A', '1': 'B', '2': 'C'}
    small = inference.rank_items(text, mapping, range(3), 2, block_size=1,
                                 behavior_vectors=behavior, behavior_order_counts=[50, 50, 50], text_weight=.5)
    large = inference.rank_items(text, mapping, range(3), 2, block_size=2,
                                 behavior_vectors=behavior, behavior_order_counts=[50, 50, 50], text_weight=.5)
    pd.testing.assert_frame_equal(small, large)
    scores = small.set_index(['master_prod_id', 'slave_prod_id']).similarity
    assert scores['A', 'B'] == pytest.approx(.4)
    assert scores['B', 'C'] == pytest.approx(.6)
    assert scores['C', 'B'] == pytest.approx(.6)


@pytest.mark.parametrize('weight', [-.1, 1.1, float('nan'), float('inf')])
def test_fusion_rejects_invalid_weight(weight):
    with pytest.raises(ValueError, match='text_weight'):
        inference.rank_items(np.eye(2), {'0': 'A', '1': 'B'}, [0], 1,
                             behavior_vectors=np.eye(2), text_weight=weight)


def test_no_eligible_behavior_stops_training():
    with pytest.raises(RuntimeError, match='行为共现数据不足'):
        training.train_item2vec(
            {'A': 0, 'B': 1}, [['0', '1']], np.array([1, 1]), vector_size=4,
        )


@pytest.mark.parametrize('mode,expected', [('similar', .85), ('complement', .20), ('hybrid', .60)])
def test_resolve_text_weight_presets_and_override(mode, expected):
    assert inference.resolve_text_weight(mode, None) == expected
    assert inference.resolve_text_weight(mode, .37) == .37


@pytest.mark.parametrize('weight', [None, .4])
def test_resolve_text_weight_rejects_unknown_mode(weight):
    with pytest.raises(ValueError, match='recall_mode'):
        inference.resolve_text_weight('unknown', weight)


@pytest.mark.parametrize('weight', [-.1, 1.1, np.nan, np.inf, -np.inf, '0.5', True])
def test_resolve_text_weight_rejects_invalid_override(weight):
    with pytest.raises(ValueError, match='text_weight'):
        inference.resolve_text_weight('hybrid', weight)


@pytest.mark.parametrize('counts,expected', [([50, 5], .92), ([5, 50], .92), ([500, 500], .2), ([0, 50], 1.)])
def test_fusion_uses_minimum_capped_order_confidence(counts, expected):
    result = inference.rank_items(
        np.array([[2., 0.], [3., 0.]]), {'0': 'A', '1': 'B'}, range(2), 1,
        behavior_vectors=np.array([[0., 4.], [2., 0.]]),
        behavior_order_counts=counts, full_confidence_orders=50, text_weight=.2,
    )
    np.testing.assert_allclose(result.similarity, [expected, expected])


@pytest.mark.parametrize('behavior', [None, [[1., 0.], [0., 0.]], [[0., 0.], [1., 0.]]])
def test_fusion_missing_behavior_uses_full_text_despite_high_counts(behavior):
    result = inference.rank_items(
        np.array([[1., 0.], [.8, .6]]), {'0': 'A', '1': 'B'}, range(2), 1,
        behavior_vectors=behavior, behavior_order_counts=[500, 500], text_weight=.2,
    )
    np.testing.assert_allclose(result.similarity, [.8, .8])


@pytest.mark.parametrize('threshold', [0, -1, 50., True, np.bool_(True), '50', None])
def test_fusion_rejects_invalid_full_confidence_orders(threshold):
    with pytest.raises(ValueError, match='full_confidence_orders'):
        inference.rank_items(np.eye(2), {'0': 'A', '1': 'B'}, [0], 1,
                             full_confidence_orders=threshold)


@pytest.mark.parametrize('counts', [None, [50], [[50], [50]], [-1, 50], [1.5, 50], [np.nan, 50], [np.inf, 50], [True, False], ['5', '50']])
def test_fusion_rejects_invalid_behavior_order_counts(counts):
    with pytest.raises(ValueError, match='behavior_order_counts'):
        inference.rank_items(np.eye(2), {'0': 'A', '1': 'B'}, [0], 1,
                             behavior_vectors=np.eye(2), behavior_order_counts=counts)


def test_confidence_fusion_is_stable_self_excluding_and_block_invariant():
    text = np.array([[1., 0.], [.8, .6], [.8, .6], [0., 1.]])
    behavior = np.array([[1., 0.], [0., 1.], [0., 1.], [1., 0.]])
    mapping = {str(i): str(i) for i in range(4)}
    kwargs = dict(behavior_vectors=behavior, behavior_order_counts=[50, 5, 5, 20],
                  full_confidence_orders=np.int64(50), text_weight=.2)
    small = inference.rank_items(text, mapping, range(4), 3, block_size=1, **kwargs)
    large = inference.rank_items(text, mapping, range(4), 3, block_size=3, **kwargs)
    pd.testing.assert_frame_equal(small, large)
    assert not (small.master_prod_id == small.slave_prod_id).any()
    assert small[small.master_prod_id == '0'].slave_prod_id.tolist() == ['1', '2', '3']
