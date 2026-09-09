import json

import numpy as np
import pandas as pd
import pytest

from item2vec import inference
from item2vec.inference import COLUMNS, rank_items, validate_artifacts, validate_top_k


def _write_artifacts(path, item_ids, vectors=None):
    if vectors is None:
        vectors = np.zeros((len(item_ids), 768), dtype=np.float32)
    np.asarray(vectors, dtype=np.float32).tofile(path / "trained_item.featCLS")
    (path / "index2item.json").write_text(
        json.dumps({str(index): item_id for index, item_id in enumerate(item_ids)}),
        encoding="utf-8",
    )


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


def test_load_trained_artifacts_reads_vectors_and_mapping(tmp_path):
    vectors = np.zeros((2, 768), dtype=np.float32)
    vectors[0, 0] = 1.0
    vectors[1, 1] = 1.0
    _write_artifacts(tmp_path, ["A", "B"], vectors)

    actual_vectors, actual_mapping = inference.load_trained_artifacts(tmp_path)

    np.testing.assert_array_equal(actual_vectors, vectors)
    assert actual_mapping == {"0": "A", "1": "B"}


@pytest.mark.parametrize("missing_name", ["trained_item.featCLS", "index2item.json"])
def test_load_trained_artifacts_reports_missing_files(tmp_path, missing_name):
    _write_artifacts(tmp_path, ["A", "B"])
    (tmp_path / missing_name).unlink()

    with pytest.raises(FileNotFoundError, match=missing_name):
        inference.load_trained_artifacts(tmp_path)


@pytest.mark.parametrize(
    ("item_id", "expected"),
    [
        ("A/../B", "A_.._B"),
        (" product:42 ", "_product_42_"),
        ("", "item"),
        ("..", "item"),
    ],
)
def test_safe_item_filename_replaces_unsafe_characters(item_id, expected):
    assert inference.safe_item_filename(item_id) == expected


def test_query_item_resolves_string_value_and_writes_exact_csv(tmp_path):
    vectors = np.zeros((3, 768), dtype=np.float32)
    vectors[0, :2] = [1.0, 0.0]
    vectors[1, :2] = [0.8, 0.6]
    vectors[2, :2] = [0.0, 1.0]
    _write_artifacts(tmp_path, [101, "B", "C"], vectors)

    output_path = inference.query_item(tmp_path, 101, top_k=2)

    assert output_path == tmp_path / "query_101.csv"
    result = pd.read_csv(output_path)
    assert list(result.columns) == COLUMNS
    assert result[["master_prod_id", "slave_prod_id"]].values.tolist() == [
        [101, "B"],
        [101, "C"],
    ]
    np.testing.assert_allclose(result["similarity"], [0.8, 0.0], atol=1e-7)


def test_query_item_rejects_unknown_item_id(tmp_path):
    _write_artifacts(tmp_path, ["A", "B"])

    with pytest.raises(ValueError, match="Unknown item ID: missing"):
        inference.query_item(tmp_path, "missing", top_k=1)


def test_export_all_writes_all_sources_with_exact_schema(tmp_path):
    vectors = np.zeros((3, 768), dtype=np.float32)
    vectors[0, :2] = [1.0, 0.0]
    vectors[1, :2] = [0.8, 0.6]
    vectors[2, :2] = [0.0, 1.0]
    _write_artifacts(tmp_path, ["A", "B", "C"], vectors)

    output_path = inference.export_all(tmp_path, top_k=1, block_size=1)

    assert output_path == tmp_path / "item_cosine_similarity.csv"
    result = pd.read_csv(output_path)
    assert list(result.columns) == COLUMNS
    assert result[["master_prod_id", "slave_prod_id"]].values.tolist() == [
        ["A", "B"],
        ["B", "A"],
        ["C", "B"],
    ]


@pytest.mark.parametrize(
    ("argv", "expected_top_k", "expected_block_size"),
    [
        (["export", "dataset/downstream"], 10, 512),
        (["export", "dataset/downstream", "--top-k", "6", "--block-size", "128"], 6, 128),
    ],
)
def test_main_forwards_export_block_size(
    monkeypatch, argv, expected_top_k, expected_block_size
):
    captured = {}

    def fake_export_all(downstream_dir, top_k, block_size):
        captured.update(
            downstream_dir=downstream_dir,
            top_k=top_k,
            block_size=block_size,
        )

    monkeypatch.setattr(inference, "export_all", fake_export_all)

    inference.main(argv)

    assert captured == {
        "downstream_dir": "dataset/downstream",
        "top_k": expected_top_k,
        "block_size": expected_block_size,
    }
