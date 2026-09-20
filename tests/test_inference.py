import json

import numpy as np
import pandas as pd
import pytest

from item2vec import inference
from item2vec.inference import COLUMNS, rank_items, validate_artifacts, validate_top_k


def _write_artifacts(path, item_ids, vectors=None):
    if vectors is None:
        vectors = np.zeros((len(item_ids), 768), dtype=np.float32)
    np.asarray(vectors, dtype=np.float32).tofile(path / "item.feat1CLS")
    np.savez(
        path / "behavior_item.npz",
        vectors=np.zeros((len(item_ids), 3), dtype=np.float32),
        item_ids=np.asarray([str(item_id) for item_id in item_ids]),
        order_counts=np.arange(len(item_ids), dtype=np.int64),
    )
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

    actual_vectors, actual_mapping, behavior_vectors, order_counts = inference.load_trained_artifacts(tmp_path)

    np.testing.assert_array_equal(actual_vectors, vectors)
    assert actual_mapping == {"0": "A", "1": "B"}
    assert behavior_vectors.shape == (2, 3)
    np.testing.assert_array_equal(order_counts, [0, 1])


@pytest.mark.parametrize("missing_name", ["item.feat1CLS", "index2item.json", "behavior_item.npz"])
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


def test_query_item_prints_chinese_status_messages(tmp_path, capsys):
    vectors = np.zeros((3, 768), dtype=np.float32)
    vectors[0, :2] = [1.0, 0.0]
    vectors[1, :2] = [0.8, 0.6]
    vectors[2, :2] = [0.0, 1.0]
    _write_artifacts(tmp_path, ["A", "B", "C"], vectors)

    output_path = inference.query_item(tmp_path, "A", top_k=2)

    assert capsys.readouterr().out == (
        "召回模式：hybrid，文本权重：0.6，行为权重：0.4，满置信订单数：50\n"
        "正在加载训练向量与索引…\n"
        "正在查询商品 A 的 Top-2 相似商品…\n"
        f"查询完成，共写入 2 条结果：{output_path}\n"
    )


def test_export_all_writes_all_sources_with_exact_schema(tmp_path):
    vectors = np.zeros((3, 768), dtype=np.float32)
    vectors[0, :2] = [1.0, 0.0]
    vectors[1, :2] = [0.8, 0.6]
    vectors[2, :2] = [0.0, 1.0]
    _write_artifacts(tmp_path, ["A", "B", "C"], vectors)

    output_path = inference.export_all(tmp_path, top_k=1, block_size=1)

    assert output_path == tmp_path / "item_similarity_hybrid.csv"
    result = pd.read_csv(output_path)
    assert list(result.columns) == COLUMNS
    assert result[["master_prod_id", "slave_prod_id"]].values.tolist() == [
        ["A", "B"],
        ["B", "A"],
        ["C", "B"],
    ]


def test_export_all_writes_complement_output_and_forwards_scoring_options(monkeypatch, tmp_path):
    _write_artifacts(tmp_path, ["A", "B"])
    captured = {}

    def fake_rank_items(*args, **kwargs):
        captured.update(kwargs)
        return pd.DataFrame([("A", "B", 0.2)], columns=COLUMNS)

    monkeypatch.setattr(inference, "rank_items", fake_rank_items)

    output_path = inference.export_all(tmp_path, top_k=1, recall_mode="complement")

    assert output_path == tmp_path / "item_similarity_complement.csv"
    assert captured["text_weight"] == 0.20
    assert captured["full_confidence_orders"] == 50
    assert list(pd.read_csv(output_path).columns) == COLUMNS


def test_query_item_keeps_legacy_filename_and_forwards_recall_mode(monkeypatch, tmp_path):
    _write_artifacts(tmp_path, ["A", "B"])
    captured = {}

    def fake_rank_items(*args, **kwargs):
        captured.update(kwargs)
        return pd.DataFrame([("A", "B", 0.2)], columns=COLUMNS)

    monkeypatch.setattr(inference, "rank_items", fake_rank_items)

    output_path = inference.query_item(tmp_path, "A", top_k=1, recall_mode="complement")

    assert output_path == tmp_path / "query_A.csv"
    assert captured["text_weight"] == 0.20
    assert captured["full_confidence_orders"] == 50


def test_write_csv_atomic_keeps_existing_file_when_write_fails(monkeypatch, tmp_path):
    output_path = tmp_path / "output.csv"
    output_path.write_text("previous\n", encoding="utf-8")

    def fail_to_csv(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_to_csv)

    with pytest.raises(OSError, match="disk full"):
        inference.write_csv_atomic(pd.DataFrame([("A", "B", 0.2)], columns=COLUMNS), output_path)

    assert output_path.read_text(encoding="utf-8") == "previous\n"
    assert list(tmp_path.glob("*.tmp")) == []


def test_write_csv_atomic_keeps_existing_file_when_csv_is_empty(monkeypatch, tmp_path):
    output_path = tmp_path / "output.csv"
    output_path.write_text("previous\n", encoding="utf-8")

    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda *args, **kwargs: None)

    with pytest.raises(OSError, match="non-empty CSV"):
        inference.write_csv_atomic(pd.DataFrame([("A", "B", 0.2)], columns=COLUMNS), output_path)

    assert output_path.read_text(encoding="utf-8") == "previous\n"
    assert list(tmp_path.glob("*.tmp")) == []


def test_write_csv_atomic_cleans_temporary_file_on_keyboard_interrupt(monkeypatch, tmp_path):
    output_path = tmp_path / "output.csv"
    output_path.write_text("previous\n", encoding="utf-8")

    def interrupt_to_csv(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(pd.DataFrame, "to_csv", interrupt_to_csv)

    with pytest.raises(KeyboardInterrupt):
        inference.write_csv_atomic(pd.DataFrame([("A", "B", 0.2)], columns=COLUMNS), output_path)

    assert output_path.read_text(encoding="utf-8") == "previous\n"
    assert list(tmp_path.glob("*.tmp")) == []


def test_write_csv_atomic_replaces_output_and_returns_path(tmp_path):
    output_path = tmp_path / "output.csv"

    actual_path = inference.write_csv_atomic(
        pd.DataFrame([("A", "B", 0.2)], columns=COLUMNS), output_path
    )

    assert actual_path == output_path
    assert output_path.stat().st_size > 0
    assert list(pd.read_csv(output_path).columns) == COLUMNS


def test_export_all_shows_block_progress_and_chinese_status_messages(
    monkeypatch, tmp_path, capsys
):
    vectors = np.zeros((3, 768), dtype=np.float32)
    vectors[0, :2] = [1.0, 0.0]
    vectors[1, :2] = [0.8, 0.6]
    vectors[2, :2] = [0.0, 1.0]
    _write_artifacts(tmp_path, ["A", "B", "C"], vectors)
    progress_calls = []

    def fake_tqdm(iterable, **kwargs):
        progress_calls.append(kwargs)
        return iterable

    monkeypatch.setattr(inference, "tqdm", fake_tqdm)

    output_path = inference.export_all(tmp_path, top_k=1, block_size=1)

    assert progress_calls == [
        {"desc": "计算商品相似度", "unit": "块", "total": 3}
    ]
    assert capsys.readouterr().out == (
        "召回模式：hybrid，文本权重：0.6，行为权重：0.4，满置信订单数：50\n"
        "正在加载训练向量与索引…\n"
        "正在计算全量商品相似度…\n"
        f"导出完成，共写入 3 条结果：{output_path}\n"
    )


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

    def fake_export_all(downstream_dir, top_k, block_size, text_weight, recall_mode, full_confidence_orders, output_dir):
        captured.update(
            downstream_dir=downstream_dir,
            top_k=top_k,
            block_size=block_size,
            text_weight=text_weight,
            recall_mode=recall_mode,
            full_confidence_orders=full_confidence_orders,
            output_dir=output_dir,
        )

    monkeypatch.setattr(inference, "export_all", fake_export_all)

    inference.main(argv)

    assert captured == {
        "downstream_dir": "dataset/downstream",
        "top_k": expected_top_k,
        "block_size": expected_block_size,
        "text_weight": None,
        "recall_mode": "hybrid",
        "full_confidence_orders": 50,
        "output_dir": None,
    }


@pytest.mark.parametrize('counts', [None, [1], [-1, 1], [1.5, 2], [np.nan, 1], [np.inf, 1], [[1], [2]], [True, False], ['1', '2']])
def test_load_trained_artifacts_rejects_missing_or_invalid_order_counts(tmp_path, counts):
    _write_artifacts(tmp_path, ['A', 'B'])
    fields = dict(vectors=np.eye(2), item_ids=np.array(['A', 'B']))
    if counts is not None:
        fields['order_counts'] = np.asarray(counts)
    np.savez(tmp_path / 'behavior_item.npz', **fields)
    with pytest.raises(ValueError, match='order_counts.*train.sh'):
        inference.load_trained_artifacts(tmp_path)


@pytest.mark.parametrize('command', ['query', 'export'])
@pytest.mark.parametrize('options,expected', [
    ([], {'recall_mode': 'hybrid', 'text_weight': None, 'full_confidence_orders': 50}),
    (['--recall-mode', 'complement', '--text-weight', '.3', '--full-confidence-orders', '100'],
     {'recall_mode': 'complement', 'text_weight': .3, 'full_confidence_orders': 100}),
])
def test_main_forwards_recall_options(monkeypatch, command, options, expected):
    captured = {}

    def capture(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(inference, 'query_item' if command == 'query' else 'export_all', capture)
    argv = [command, 'dataset/downstream'] + (['A'] if command == 'query' else []) + options
    inference.main(argv)
    assert {key: captured[key] for key in expected} == expected


@pytest.mark.parametrize('command', ['query', 'export'])
@pytest.mark.parametrize('mode,weight,expected', [('similar', None, .985), ('complement', None, .92), ('hybrid', None, .96), ('similar', .3, .93)])
def test_query_and_export_apply_recall_mode_and_confidence(tmp_path, command, mode, weight, expected):
    text = np.zeros((2, 768), dtype=np.float32)
    text[:, 0] = 1.
    _write_artifacts(tmp_path, ['A', 'B'], text)
    np.savez(tmp_path / 'behavior_item.npz', vectors=np.eye(2),
             item_ids=np.array(['A', 'B']), order_counts=np.array([100, 10]))
    kwargs = dict(top_k=1, recall_mode=mode, text_weight=weight, full_confidence_orders=100)
    if command == 'query':
        path = inference.query_item(tmp_path, 'A', **kwargs)
    else:
        path = inference.export_all(tmp_path, **kwargs)
    np.testing.assert_allclose(pd.read_csv(path).similarity, expected)


@pytest.mark.parametrize('command', ['query', 'export'])
@pytest.mark.parametrize('weight,expected_weights', [(None, '文本权重：0.2，行为权重：0.8'), (.35, '文本权重：0.35，行为权重：0.65')])
def test_query_and_export_log_selected_mode_and_resolved_weights(tmp_path, capsys, command, weight, expected_weights):
    _write_artifacts(tmp_path, ['A', 'B'])
    kwargs = dict(top_k=1, recall_mode='complement', text_weight=weight, full_confidence_orders=100)
    if command == 'query':
        inference.query_item(tmp_path, 'A', **kwargs)
    else:
        inference.export_all(tmp_path, **kwargs)
    assert capsys.readouterr().out.splitlines()[0] == (
        f'召回模式：complement，{expected_weights}，满置信订单数：100'
    )


def test_query_item_writes_to_separate_output_directory(tmp_path):
    artifact_dir = tmp_path / "embeddings"
    artifact_dir.mkdir()
    output_dir = tmp_path / "results"
    _write_artifacts(artifact_dir, ["A", "B"])

    output_path = inference.query_item(
        artifact_dir, "A", top_k=1, output_dir=output_dir
    )

    assert output_path == output_dir / "query_A.csv"
    assert output_path.is_file()


def test_export_all_writes_to_separate_output_directory(tmp_path):
    artifact_dir = tmp_path / "embeddings"
    artifact_dir.mkdir()
    output_dir = tmp_path / "results"
    _write_artifacts(artifact_dir, ["A", "B"])

    output_path = inference.export_all(
        artifact_dir, top_k=1, output_dir=output_dir
    )

    assert output_path == output_dir / "item_similarity_hybrid.csv"
    assert output_path.is_file()
