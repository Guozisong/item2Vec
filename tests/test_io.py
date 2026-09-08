import numpy as np

from item2vec.io import get_cosine_similarity, load_item_index, write_item_indexes


def test_write_item_indexes_round_trips_duplicate_ids(tmp_path):
    indexes = write_item_indexes(["A", "B", "A"], tmp_path)
    assert indexes == {"A": 0, "B": 1}
    assert load_item_index(tmp_path) == {"A": 0, "B": 1}


def test_get_cosine_similarity_excludes_source_item():
    result = get_cosine_similarity(
        np.array([[1, 0], [0, 1]], dtype=np.float32),
        {"0": "A", "1": "B"},
        topk=1,
    )
    assert result[["master_prod_id", "slave_prod_id"]].values.tolist() == [
        ["A", "B"],
        ["B", "A"],
    ]
