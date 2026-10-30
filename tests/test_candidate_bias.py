import json

import numpy as np
import pandas as pd
import pytest

from item2vec.inference import rank_items
from item2vec.scoring import compute_candidate_bias
from item2vec.training import write_behavior_embedding


def test_bias_matches_dense_top_ten_and_excludes_cold_items():
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(14, 4)).astype(np.float32)
    contexts = rng.normal(size=(14, 4)).astype(np.float32)
    vectors[-1] = contexts[-1] = 0
    scores = vectors[:-1] @ contexts[:-1].T
    expected = .4 * np.sort(scores, axis=0)[-10:].mean(axis=0)
    bias = compute_candidate_bias(vectors, contexts)
    np.testing.assert_allclose(bias[:-1], expected, rtol=1e-6, atol=1e-6)
    assert bias[-1] == 0


def test_bias_changes_hub_ranking_and_is_block_invariant():
    inputs = np.array([[1., 0.], [0., 1.], [0., 1.]])
    contexts = np.array([[0., 1.], [1., 5.], [.9, 0.]])
    bias = compute_candidate_bias(inputs, contexts)
    mapping = {'0': 'A', '1': 'hub', '2': 'specific'}
    kwargs = dict(behavior_vectors=inputs, behavior_context_vectors=contexts,
                  behavior_order_counts=[5, 5, 5], recall_mode='complement')
    raw = rank_items(np.eye(3), mapping, [0], 2, **kwargs)
    corrected = rank_items(np.eye(3), mapping, [0], 2, candidate_bias=bias, **kwargs)
    assert raw.iloc[0].slave_prod_id == 'hub'
    assert corrected.iloc[0].slave_prod_id == 'specific'
    small = rank_items(np.eye(3), mapping, range(3), 2, block_size=1,
                       candidate_bias=bias, **kwargs)
    large = rank_items(np.eye(3), mapping, range(3), 2, block_size=3,
                       candidate_bias=bias, **kwargs)
    pd.testing.assert_frame_equal(small, large)
    with pytest.raises(ValueError, match='candidate_bias'):
        rank_items(np.eye(3), mapping, [0], 2, candidate_bias=[np.nan]*3, **kwargs)


def test_behavior_artifact_persists_precomputed_bias(tmp_path):
    vectors = np.array([[1., 0.], [0., 1.]])
    contexts = np.array([[2., 1.], [1., 3.]])
    path = write_behavior_embedding(vectors, {'0': 'A', '1': 'B'}, [5, 5], {},
                                    tmp_path, context_vectors=contexts)
    with np.load(path, allow_pickle=False) as artifact:
        np.testing.assert_array_equal(artifact['candidate_bias'],
                                      compute_candidate_bias(vectors, contexts))


@pytest.mark.parametrize('weight', [0., .6, .8, 1.2])
def test_candidate_bias_weight_scales_saved_bias_and_metadata(tmp_path, weight):
    inputs = np.array([[1., 0.], [0., 1.]])
    contexts = np.array([[2., 1.], [1., 3.]])
    baseline = compute_candidate_bias(inputs, contexts)
    bias = compute_candidate_bias(inputs, contexts, weight=weight)
    np.testing.assert_allclose(bias, baseline * (weight / .4))
    original_metadata = {'epochs': 10}
    path = write_behavior_embedding(inputs, {'0': 'A', '1': 'B'}, [5, 5], original_metadata,
                                    tmp_path, context_vectors=contexts,
                                    candidate_bias_weight=weight)
    with np.load(path, allow_pickle=False) as artifact:
        np.testing.assert_array_equal(artifact['candidate_bias'], bias)
        assert json.loads(artifact['metadata'].item())['candidate_bias_weight'] == weight
    assert original_metadata == {'epochs': 10}


@pytest.mark.parametrize('weight', [-.1, float('nan'), float('inf'), True, '0.6'])
def test_candidate_bias_rejects_invalid_weight(weight):
    with pytest.raises(ValueError, match='candidate_bias_weight'):
        compute_candidate_bias(np.eye(2), np.eye(2), weight=weight)
