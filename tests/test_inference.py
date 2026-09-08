import numpy as np
import pandas as pd
import pytest

from item2vec.inference import COLUMNS, rank_items, validate_artifacts, validate_top_k


def test_rank_items_excludes_self_and_uses_stable_descending_order():
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    index2item = {"0": "A", "1": "B", "2": "C", "3": "D"}

    result = rank_items(vectors, index2item, source_indexes=range(len(vectors)), top_k=2)

    assert COLUMNS == ["master_prod_id", "slave_prod_id", "similarity"]
    assert list(result.columns) == COLUMNS
    assert result[["master_prod_id", "slave_prod_id"]].values.tolist() == [
        ["A", "B"],
        ["A", "C"],
        ["B", "A"],
        ["B", "C"],
        ["C", "B"],
        ["C", "A"],
        ["D", "A"],
        ["D", "B"],
    ]
    np.testing.assert_allclose(
        result["similarity"].to_numpy(),
        [0.8, 0.0, 0.8, 0.6, 0.6, 0.0, 0.0, 0.0],
        atol=1e-7,
    )


def test_rank_items_processes_only_requested_source_indexes():
    vectors = np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])
    index2item = {"0": "A", "1": "B", "2": "C"}

    result = rank_items(vectors, index2item, source_indexes=[2], top_k=2)

    assert result[["master_prod_id", "slave_prod_id"]].values.tolist() == [
        ["C", "B"],
        ["C", "A"],
    ]


def test_validate_artifacts_rejects_count_mismatch():
    with pytest.raises(ValueError, match="row count"):
        validate_artifacts(np.ones((2, 3)), {"0": "A"})


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf])
def test_validate_artifacts_rejects_non_finite_vectors(invalid_value):
    vectors = np.array([[1.0, invalid_value], [0.0, 1.0]])

    with pytest.raises(ValueError, match="finite"):
        validate_artifacts(vectors, {"0": "A", "1": "B"})


@pytest.mark.parametrize(
    ("vectors", "index2item", "message"),
    [
        (np.ones(3), {"0": "A"}, "2D"),
        (np.ones((2, 3)), {"0": "A", "2": "B"}, "contiguous"),
    ],
)
def test_validate_artifacts_rejects_invalid_shape_or_keys(vectors, index2item, message):
    with pytest.raises(ValueError, match=message):
        validate_artifacts(vectors, index2item)


def test_rank_items_is_equivalent_across_block_sizes():
    vectors = np.array(
        [[1.0, 2.0], [2.0, 1.0], [-1.0, 1.0], [0.0, 0.0]],
        dtype=np.float64,
    )
    index2item = {str(index): f"item-{index}" for index in range(len(vectors))}

    source_indexes = range(len(vectors))
    expected = rank_items(vectors, index2item, source_indexes, top_k=2, block_size=1)
    actual = rank_items(vectors, index2item, source_indexes, top_k=2, block_size=3)

    pd.testing.assert_frame_equal(actual, expected)


def test_normalization_preserves_float32_dtype():
    from item2vec.inference import _normalize_vectors

    vectors = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)

    normalized = _normalize_vectors(vectors)

    assert normalized.dtype == np.float32


@pytest.mark.parametrize("top_k", [1.0, "1", True])
def test_validate_top_k_rejects_non_integer_values(top_k):
    with pytest.raises(ValueError, match="integer"):
        validate_top_k(top_k, item_count=3)


def test_validate_top_k_accepts_numpy_integer():
    validate_top_k(np.int64(1), item_count=3)


@pytest.mark.parametrize("top_k", [0, 3])
def test_validate_top_k_rejects_zero_and_item_count(top_k):
    with pytest.raises(ValueError, match="1 <= top_k < item_count"):
        validate_top_k(top_k, item_count=3)
