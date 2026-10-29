import numpy as np
import pandas as pd

from item2vec.cold_start import transfer_behavior
from item2vec.inference import rank_items


def test_cold_source_transfers_from_same_category_warm_anchors():
    text = np.array([[1., 0.], [1., 0.], [1., 0.], [0., 1.]])
    inputs = np.array([[0., 0.], [1., 0.], [1., 0.], [0., 1.]])
    contexts = np.array([[0., 0.], [0., 1.], [0., 1.], [2., 0.]])
    mapping = {str(i): str(i) for i in range(4)}
    kwargs = dict(behavior_vectors=inputs, behavior_context_vectors=contexts,
                  behavior_order_counts=[0, 5, 5, 5], recall_mode='complement',
                  item_categories={'0': 'beef', '1': 'beef', '2': 'beef', '3': 'vegetable'})
    one = rank_items(text, mapping, range(4), 2, block_size=1, **kwargs)
    many = rank_items(text, mapping, range(4), 2, block_size=4, **kwargs)
    pd.testing.assert_frame_equal(one, many)
    cold = one[one.master_prod_id == '0']
    assert cold.iloc[0].slave_prod_id == '3'
    assert set(cold.recall_source) == {'behavior_transfer'}
    assert '0' not in set(cold.slave_prod_id)
    assert set(one[one.master_prod_id == '1'].recall_source) == {'behavior'}


def test_insufficient_or_wrong_category_anchors_fall_back_to_text():
    text = np.eye(3)
    behavior = np.array([[0., 0.], [1., 0.], [0., 1.]])
    mapping = {str(i): str(i) for i in range(3)}
    result = rank_items(text, mapping, [0], 2, behavior_vectors=behavior,
                        behavior_order_counts=[0, 5, 5], recall_mode='complement',
                        item_categories={'0': 'beef', '1': 'milk', '2': 'beef'})
    assert set(result.recall_source) == {'text_fallback'}
    vector, anchors = transfer_behavior(0, text, behavior, mapping,
                                       np.array([False, True, True]), {})
    assert vector is None and anchors == []
