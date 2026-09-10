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
        behavior_vectors=behavior, text_weight=weight,
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
                                 behavior_vectors=behavior, text_weight=.5)
    large = inference.rank_items(text, mapping, range(3), 2, block_size=2,
                                 behavior_vectors=behavior, text_weight=.5)
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
