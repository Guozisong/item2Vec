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


def test_write_behavior_embedding_writes_float32_vectors_and_item_ids(tmp_path):
    embedding = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)

    output_path = write_behavior_embedding(embedding, {"0": 101, "1": "B"}, tmp_path)

    assert output_path == tmp_path / "behavior_item.npz"
    with np.load(output_path, allow_pickle=False) as artifact:
        assert artifact["vectors"].dtype == np.float32
        np.testing.assert_array_equal(artifact["vectors"], embedding.astype(np.float32))
        assert artifact["item_ids"].tolist() == ["101", "B"]


def test_train_uses_configured_word2vec_parameters_without_text_initialization(monkeypatch):
    captured = {}

    class FakeVectors:
        index_to_key = ["0", "1"]

        def __getitem__(self, key):
            return np.full(4, int(key) + 1, dtype=np.float32)

    class FakeWord2Vec:
        def __init__(self, **kwargs):
            captured["init"] = kwargs
            self.wv = FakeVectors()

        def build_vocab(self, sentences):
            captured["vocab"] = sentences

        def train(self, sentences, **kwargs):
            captured["train"] = {"sentences": sentences, **kwargs}

    monkeypatch.setitem(sys.modules, "gensim", types.ModuleType("gensim"))
    monkeypatch.setitem(sys.modules, "gensim.models", types.SimpleNamespace(Word2Vec=FakeWord2Vec))

    baskets = [["A", "B"]] * 5
    result, _model = training.train_item2vec(
        {"A": 0, "B": 1},
        baskets,
        vector_size=4,
        window=8,
        negative=4,
        epochs=6,
    )

    np.testing.assert_array_equal(result, [[1] * 4, [2] * 4])
    assert "sentences" not in captured["init"]
    assert captured["init"]["vector_size"] == 4
    assert captured["init"]["window"] == 8
    assert captured["init"]["negative"] == 4
    assert captured["vocab"] == [["0", "1"]] * 5
    assert captured["train"]["epochs"] == 6


def test_main_trains_behavior_only_and_preserves_string_product_ids(monkeypatch, tmp_path, capsys):
    captured = {}
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "order_item.csv").write_text(
        "user_id,dt,prod_id\nu1,d1,001\nu1,d1,B\nu2,d2,001\n",
        encoding="utf-8",
    )
    (tmp_path / "item2index.json").write_text(json.dumps({"001": 0, "B": 1}))
    (tmp_path / "index2item.json").write_text(json.dumps({"0": "001", "1": "B"}))
    monkeypatch.setattr(training, "tqdm", lambda iterable, **_kwargs: iterable)
    monkeypatch.setattr(
        training,
        "train_item2vec",
        lambda item2index, baskets, **kwargs: (
            captured.update(item2index=item2index, baskets=baskets, **kwargs)
            or (np.zeros((2, kwargs["vector_size"]), dtype=np.float32), None)
        )
    )

    training.main([str(raw_dir), str(tmp_path), "--vector-size", "4", "--window", "9",
                   "--negative", "3", "--epochs", "7"])

    assert captured["item2index"] == {"001": 0, "B": 1}
    assert captured["basket_index"] == [["0", "1"]]
    assert (captured["window"], captured["negative"], captured["epochs"]) == (9, 3, 7)
    with np.load(tmp_path / "behavior_item.npz", allow_pickle=False) as artifact:
        assert artifact["vectors"].shape == (2, 4)
        assert artifact["item_ids"].tolist() == ["001", "B"]
    output = capsys.readouterr().out
    assert "已读取 3 条行为，包含 2 个用户。" in output
    assert "已构建 1 个有效购物篮" in output
    assert "行为向量覆盖 0 个商品" in output
    assert not (tmp_path / "trained_item.featCLS").exists()


def test_no_eligible_behavior_produces_explicit_zero_vectors():
    vectors, model = training.train_item2vec(
        {"A": 0, "B": 1}, [["A", "B"]], vector_size=4,
    )
    assert model is None
    np.testing.assert_array_equal(vectors, np.zeros((2, 4), dtype=np.float32))
