import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

from item2vec import training
from item2vec.training import write_behavior_embedding


def test_prepare_order_baskets_groups_orders_deduplicates_and_canonicalizes_indexes():
    item2index = {"A": 2, "B": 0, "C": 1}
    dataframe = pd.DataFrame(
        [
            ("order-2", "C", "20260910"),
            ("order-2", "missing", "20260910"),
            ("order-2", "A", "20260910"),
            ("order-2", "C", "20260910"),
            ("order-1", "A", "20260909"),
            ("order-1", "B", "20260909"),
        ],
        columns=["order_id", "prod_id", "dt"],
    )
    reordered = dataframe.iloc[[2, 0, 3, 1, 5, 4]]

    basket_indexes, order_counts, stats = training.prepare_order_baskets(dataframe, item2index)
    reordered_baskets, reordered_counts, reordered_stats = training.prepare_order_baskets(
        reordered, item2index
    )

    assert basket_indexes == [["1", "2"], ["0", "2"]]
    np.testing.assert_array_equal(order_counts, [1, 1, 2])
    assert stats == {"orders": 2, "valid_baskets": 2, "large_baskets": 0}
    assert reordered_baskets == basket_indexes
    np.testing.assert_array_equal(reordered_counts, order_counts)
    assert reordered_stats == stats


def test_prepare_order_baskets_counts_large_and_single_item_orders_without_retaining_them():
    item2index = {f"P{index}": index for index in range(31)}
    dataframe = pd.DataFrame(
        [("large", product, "20260910") for product in item2index]
        + [("single", "P0", "20260910")],
        columns=["order_id", "prod_id", "dt"],
    )

    basket_indexes, order_counts, stats = training.prepare_order_baskets(dataframe, item2index)

    assert basket_indexes == []
    np.testing.assert_array_equal(order_counts, [2] + [1] * 30)
    assert stats == {"orders": 2, "valid_baskets": 0, "large_baskets": 1}


def test_prepare_order_baskets_retains_exactly_thirty_distinct_known_products():
    item2index = {f"P{index}": index for index in range(30)}
    dataframe = pd.DataFrame(
        [("boundary", product, "20260910") for product in reversed(item2index)]
        + [("boundary", "P0", "20260910"), ("boundary", "unknown", "20260910")],
        columns=["order_id", "prod_id", "dt"],
    )

    baskets, counts, stats = training.prepare_order_baskets(dataframe, item2index)

    assert baskets == [[str(index) for index in range(30)]]
    np.testing.assert_array_equal(counts, np.ones(30, dtype=np.int64))
    assert stats == {"orders": 1, "valid_baskets": 1, "large_baskets": 0}


def test_prepare_order_baskets_requires_order_level_columns():
    dataframe = pd.DataFrame({"prod_id": ["A"]})

    with pytest.raises(ValueError, match="order_id.*dt"):
        training.prepare_order_baskets(dataframe, {"A": 0})


def test_prepare_order_baskets_rejects_empty_and_null_required_fields():
    with pytest.raises(ValueError, match="must not be empty"):
        training.prepare_order_baskets(
            pd.DataFrame(columns=["order_id", "prod_id", "dt"]), {"A": 0}
        )
    with pytest.raises(ValueError, match="must not contain nulls"):
        training.prepare_order_baskets(
            pd.DataFrame({"order_id": ["one"], "prod_id": [None], "dt": ["20260910"]}),
            {"A": 0},
        )


def test_prepare_order_baskets_requires_a_two_item_minimum_basket_size():
    dataframe = pd.DataFrame({"order_id": ["one"], "prod_id": ["A"], "dt": ["20260910"]})

    with pytest.raises(ValueError, match="max_basket_size must be at least 2"):
        training.prepare_order_baskets(dataframe, {"A": 0}, max_basket_size=1)


def test_prepare_order_baskets_shows_chinese_progress(monkeypatch):
    captured = {}

    def fake_tqdm(iterable, **kwargs):
        captured["iterable"] = iterable
        captured["kwargs"] = kwargs
        return iterable

    monkeypatch.setattr(training, "tqdm", fake_tqdm)
    dataframe = pd.DataFrame(
        {"order_id": ["one", "one", "two", "two"], "prod_id": ["A", "B", "A", "B"],
         "dt": ["20260910"] * 4}
    )

    training.prepare_order_baskets(dataframe, {"A": 0, "B": 1})

    assert captured["kwargs"] == {"desc": "构建训练购物篮", "unit": "单", "total": 2}


def test_write_behavior_embedding_stores_counts_metadata_and_replaces_atomically(monkeypatch, tmp_path):
    embedding = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    metadata = {"vector_size": 2, "max_basket_size": 30, "min_order_count": 5}
    existing = tmp_path / "behavior_item.npz"
    existing.write_bytes(b"old artifact")
    replace = training.os.replace
    replacements = []

    def checked_replace(source, target):
        assert existing.read_bytes() == b"old artifact"
        assert source == tmp_path / "behavior_item.npz.tmp"
        assert target == existing
        with np.load(source, allow_pickle=False) as artifact:
            assert artifact["vectors"].shape == (2, 2)
            assert artifact["order_counts"].tolist() == [10, 20]
        replacements.append((source, target))
        replace(source, target)

    monkeypatch.setattr(training.os, "replace", checked_replace)
    output_path = write_behavior_embedding(
        embedding, {"0": 101, "1": "B"}, np.array([10, 20], dtype=np.int32), metadata, tmp_path
    )

    assert output_path == tmp_path / "behavior_item.npz"
    assert len(replacements) == 1
    assert not (tmp_path / "behavior_item.npz.tmp").exists()
    assert not (tmp_path / "behavior_item.npz.tmp.npz").exists()
    with np.load(output_path, allow_pickle=False) as artifact:
        assert artifact["vectors"].dtype == np.float32
        np.testing.assert_array_equal(artifact["vectors"], embedding.astype(np.float32))
        assert artifact["item_ids"].tolist() == ["101", "B"]
        assert artifact["order_counts"].dtype == np.int64
        np.testing.assert_array_equal(artifact["order_counts"], [10, 20])
        assert artifact["metadata"].item() == json.dumps(metadata, sort_keys=True)


@pytest.mark.parametrize("failure", ["write", "validation", "replace"])
def test_write_behavior_embedding_preserves_existing_artifact_on_failure(monkeypatch, tmp_path, failure):
    existing = tmp_path / "behavior_item.npz"
    existing.write_bytes(b"old artifact")
    savez = np.savez

    def failing_savez(stream, **arrays):
        if failure == "write":
            stream.write(b"partial artifact")
            raise OSError("simulated write failure")
        arrays["vectors"][0, 0] = np.nan
        savez(stream, **arrays)

    def failing_replace(source, target):
        raise OSError("simulated replace failure")

    if failure == "replace":
        monkeypatch.setattr(training.os, "replace", failing_replace)
    else:
        monkeypatch.setattr(training.np, "savez", failing_savez)

    with pytest.raises((ValueError, OSError)):
        write_behavior_embedding(np.eye(2), {"0": "A", "1": "B"}, [5, 5], {}, tmp_path)

    assert existing.read_bytes() == b"old artifact"
    assert not (tmp_path / "behavior_item.npz.tmp").exists()
    assert not (tmp_path / "behavior_item.npz.tmp.npz").exists()


@pytest.mark.parametrize(
    "change,message",
    [
        ({"vectors": np.ones(2)}, "2D"),
        ({"vectors": np.ones((1, 2))}, "row count"),
        ({"vectors": np.ones((2, 0))}, "dimension"),
        ({"vectors": np.zeros((2, 2))}, "coverage"),
        ({"vectors": np.array([[np.inf, 0], [0, 1]])}, "finite"),
        ({"order_counts": np.array([5])}, "order_counts"),
        ({"order_counts": np.array([-1, 5])}, "order_counts"),
        ({"item_ids": np.array(["B", "A"])}, "item_ids"),
        ({"metadata": np.array("not JSON")}, "metadata"),
        ({"order_counts": None}, "required"),
    ],
)
def test_write_behavior_embedding_validates_saved_arrays_before_replacing(monkeypatch, tmp_path, change, message):
    existing = tmp_path / "behavior_item.npz"
    existing.write_bytes(b"old artifact")
    savez = np.savez

    def corrupt_savez(stream, **arrays):
        for key, value in change.items():
            if value is None:
                arrays.pop(key)
            else:
                arrays[key] = value
        savez(stream, **arrays)

    monkeypatch.setattr(training.np, "savez", corrupt_savez)

    with pytest.raises(ValueError, match=message):
        write_behavior_embedding(np.eye(2), {"0": "A", "1": "B"}, [5, 5], {}, tmp_path)

    assert existing.read_bytes() == b"old artifact"
    assert not (tmp_path / "behavior_item.npz.tmp").exists()


@pytest.fixture
def captured_word2vec(monkeypatch):
    captured = {}

    class FakeVectors:
        index_to_key = []

        def __getitem__(self, key):
            return np.full(captured["init"]["vector_size"], int(key) + 1, dtype=np.float32)

    class FakeWord2Vec:
        def __init__(self, **kwargs):
            captured["init"] = kwargs
            self.wv = FakeVectors()

        def build_vocab(self, sentences):
            captured["vocab"] = sentences
            self.wv.index_to_key = sorted({item for basket in sentences for item in basket})

        def train(self, sentences, **kwargs):
            captured["train"] = {"sentences": sentences, **kwargs}

    monkeypatch.setitem(sys.modules, "gensim", types.ModuleType("gensim"))
    monkeypatch.setitem(sys.modules, "gensim.models", types.SimpleNamespace(Word2Vec=FakeWord2Vec))
    return captured


@pytest.mark.parametrize("config,expected_window", [({}, 30), ({"max_basket_size": 8}, 8)])
def test_train_uses_fixed_full_basket_window_and_configured_parameters(captured_word2vec, config, expected_window):
    captured = captured_word2vec
    baskets = [[str(index) for index in range(expected_window)]]
    result, _model = training.train_item2vec(
        {f"P{index}": index for index in range(expected_window)},
        baskets,
        np.full(expected_window, 5, dtype=np.int64),
        vector_size=4,
        negative=4,
        epochs=6,
        workers=1,
        **config,
    )

    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, [[index + 1] * 4 for index in range(expected_window)])
    assert "sentences" not in captured["init"]
    assert captured["init"]["vector_size"] == 4
    assert captured["init"]["window"] == expected_window
    assert captured["init"]["shrink_windows"] is False
    assert captured["init"]["min_count"] == 1
    assert captured["init"]["sg"] == 1
    assert captured["init"]["negative"] == 4
    assert captured["init"]["epochs"] == 6
    assert captured["init"]["workers"] == 1
    assert captured["init"]["sample"] == 1e-4
    assert captured["vocab"] == baskets
    assert captured["train"] == {
        "sentences": baskets, "total_examples": 1, "epochs": 6,
        "start_alpha": 0.002, "end_alpha": 0.0005,
    }


def test_train_filters_by_order_counts_and_only_populates_supported_items(captured_word2vec):
    baskets = [["0", "1", "2"], ["2", "3"], ["3"]] + [["2", "4"]] * 8
    vectors, model = training.train_item2vec(
        {"A": 0, "B": 1, "rare": 2, "solo": 3, "other_rare": 4, "absent": 5},
        baskets, np.array([10, 12, 9, 30, 9, 0]), vector_size=4, min_order_count=10,
    )

    assert model is not None
    assert captured_word2vec["vocab"] == [["0", "1"]]
    assert captured_word2vec["train"]["sentences"] == [["0", "1"]]
    assert vectors.dtype == np.float32
    np.testing.assert_array_equal(vectors[:2], [[1] * 4, [2] * 4])
    np.testing.assert_array_equal(vectors[2:], np.zeros((4, 4), dtype=np.float32))


@pytest.mark.parametrize("baskets,counts", [([], [5, 5]), ([["0", "1"]], [4, 5]), ([["0"], ["1"]], [5, 5])])
def test_train_raises_when_no_behavior_pair_survives(baskets, counts):
    with pytest.raises(RuntimeError, match="行为共现数据不足"):
        training.train_item2vec({"A": 0, "B": 1}, baskets, np.array(counts), vector_size=4)


@pytest.mark.parametrize("name", ["vector_size", "max_basket_size", "negative", "epochs", "min_order_count", "workers"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_train_rejects_invalid_integer_configuration(name, value):
    with pytest.raises(ValueError, match=name):
        training.train_item2vec({"A": 0, "B": 1}, [["0", "1"]], np.array([5, 5]), **{name: value})


@pytest.mark.parametrize("counts", [[5], [[5, 5]], [-1, 5], [np.nan, 5], [np.inf, 5], [1.5, 5]])
def test_train_rejects_invalid_order_counts(counts):
    with pytest.raises(ValueError, match="order_counts"):
        training.train_item2vec({"A": 0, "B": 1}, [["0", "1"]], np.array(counts))


def test_main_trains_behavior_only_and_preserves_string_product_ids(monkeypatch, tmp_path, capsys):
    captured = {}
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "order_item.csv").write_text(
        "order_id,user_id,dt,prod_id\no1,u1,d1,001\no1,u1,d1,B\no2,u1,d1,001\n",
        encoding="utf-8",
    )
    (tmp_path / "item2index.json").write_text(json.dumps({"001": 0, "B": 1}))
    (tmp_path / "index2item.json").write_text(json.dumps({"0": "001", "1": "B"}))
    monkeypatch.setattr(training, "tqdm", lambda iterable, **_kwargs: iterable)
    monkeypatch.setattr(
        training,
        "train_item2vec",
        lambda item2index, baskets, order_counts, **kwargs: (
            captured.update(item2index=item2index, baskets=baskets, order_counts=order_counts, **kwargs)
            or (np.ones((2, kwargs["vector_size"]), dtype=np.float32), None)
        )
    )

    training.main([str(raw_dir), str(tmp_path), "--vector-size", "4", "--max-basket-size", "9",
                   "--negative", "3", "--epochs", "7", "--min-order-count", "2"])

    assert captured["item2index"] == {"001": 0, "B": 1}
    assert captured["baskets"] == [["0", "1"]]
    np.testing.assert_array_equal(captured["order_counts"], [2, 1])
    assert (captured["max_basket_size"], captured["negative"], captured["epochs"], captured["min_order_count"]) == (9, 3, 7, 2)
    with np.load(tmp_path / "behavior_item.npz", allow_pickle=False) as artifact:
        assert artifact["vectors"].shape == (2, 4)
        assert artifact["item_ids"].tolist() == ["001", "B"]
        assert artifact["order_counts"].tolist() == [2, 1]
        assert json.loads(artifact["metadata"].item()) == {
            "vector_size": 4, "max_basket_size": 9, "negative": 3, "epochs": 7, "min_order_count": 2,
        }
    output = capsys.readouterr().out
    assert "已读取 3 条行为，包含 2 个订单。" in output
    assert "已构建 1 个有效购物篮" in output
    assert "排除 0 个超大购物篮" in output
    assert "商品目录共 2 个商品" in output
    assert "行为向量覆盖 2 个商品" in output
    assert str(tmp_path / "behavior_item.npz") in output
    assert not (tmp_path / "trained_item.featCLS").exists()


@pytest.mark.parametrize(
    "csv,exception,message",
    [
        ("order_id,dt,prod_id\n", ValueError, "empty"),
        ("dt,prod_id\nd1,A\nd1,B\n", ValueError, "order_id"),
        ("order_id,dt,prod_id\no1,d1,A\no1,d1,B\n", RuntimeError, "行为共现数据不足"),
    ],
)
def test_main_preserves_existing_artifact_when_production_data_is_insufficient(tmp_path, csv, exception, message):
    (tmp_path / "order_item.csv").write_text(csv)
    (tmp_path / "item2index.json").write_text(json.dumps({"A": 0, "B": 1}))
    (tmp_path / "index2item.json").write_text(json.dumps({"0": "A", "1": "B"}))
    existing = tmp_path / "behavior_item.npz"
    existing.write_bytes(b"old artifact")

    with pytest.raises(exception, match=message):
        training.main([str(tmp_path), str(tmp_path)])

    assert existing.read_bytes() == b"old artifact"
