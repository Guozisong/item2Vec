import sys
import types

import numpy as np

from item2vec import training
from item2vec.training import build_basket_indexes, write_trained_embedding


def test_build_basket_indexes_drops_unknown_and_invalid_lengths():
    baskets = [["A", "missing", "B"], ["A"], list("ABCDEFGHIJKLMNOPQRSTU")]
    item2index = {code: index for index, code in enumerate("ABCDEFGHIJKLMNOPQRSTU")}
    assert build_basket_indexes(baskets, item2index) == [["0", "1"]]


def test_write_trained_embedding_writes_float32_artifact_without_similarity_csv(tmp_path):
    embedding = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)

    output_path = write_trained_embedding(embedding, tmp_path)

    assert output_path == tmp_path / "trained_item.featCLS"
    assert np.array_equal(np.fromfile(output_path, dtype=np.float32), embedding.astype(np.float32).ravel())
    assert not (tmp_path / "item_cosine_similarity.csv").exists()


def test_train_uses_configured_word2vec_parameters(monkeypatch):
    captured = {}

    class FakeVectors:
        index_to_key = []
        vectors = np.empty((0, 2), dtype=np.float32)

        def __getitem__(self, key):
            raise AssertionError("no learned vectors expected")

    class FakeWord2Vec:
        def __init__(self, **kwargs):
            captured["init"] = kwargs
            self.wv = FakeVectors()

        def train(self, sentences, **kwargs):
            captured["train"] = {"sentences": sentences, **kwargs}

    monkeypatch.setitem(sys.modules, "gensim", types.ModuleType("gensim"))
    monkeypatch.setitem(sys.modules, "gensim.models", types.SimpleNamespace(Word2Vec=FakeWord2Vec))

    embedding = np.ones((2, 2), dtype=np.float32)
    result, _model = training.train_item2vec_with_bert_init(
        embedding,
        {"A": 0, "B": 1},
        [["A", "B"]],
        lambda_bert=0.25,
        window=8,
        negative=4,
        epochs=6,
    )

    assert np.array_equal(result, embedding)
    assert captured["init"]["window"] == 8
    assert captured["init"]["negative"] == 4
    assert captured["train"]["epochs"] == 6


def test_main_forwards_named_training_options(monkeypatch, tmp_path):
    captured = {}

    class FakeGroupedBaskets:
        def __getitem__(self, _column):
            return self

        def apply(self, function):
            function(["A", "B"])
            return self

        def tolist(self):
            return [["A", "B"]]

    class FakeDataframe:
        def groupby(self, _columns):
            return FakeGroupedBaskets()

    monkeypatch.setitem(
        sys.modules,
        "pandas",
        types.SimpleNamespace(read_csv=lambda _path: FakeDataframe()),
    )
    monkeypatch.setattr(training, "load_plm_embedding", lambda _path: np.ones((2, 2), dtype=np.float32))
    monkeypatch.setattr(training, "load_item_index", lambda _path: {"A": 0, "B": 1})
    monkeypatch.setattr(
        training,
        "train_item2vec_with_bert_init",
        lambda embedding, item2index, baskets, **kwargs: (
            captured.update({"embedding": embedding, "item2index": item2index, "baskets": baskets, **kwargs})
            or (embedding, object())
        ),
    )
    monkeypatch.setattr(training, "write_trained_embedding", lambda embedding, path: captured.update(output=(embedding, path)))

    training.main([
        str(tmp_path / "raw"), str(tmp_path / "downstream"),
        "--bert-weight", "0.4", "--window", "9", "--negative", "3", "--epochs", "7",
    ])

    assert captured["lambda_bert"] == 0.4
    assert captured["window"] == 9
    assert captured["negative"] == 3
    assert captured["epochs"] == 7
