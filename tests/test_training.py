import json

import numpy as np
import pytest

from item2vec import training
from item2vec.training import write_behavior_embedding


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


def test_write_behavior_embedding_works_with_nonseekable_output(monkeypatch, tmp_path):
    real_open = training.Path.open
    temporary_path = tmp_path / "behavior_item.npz.tmp"

    class NonseekableWriter:
        def __init__(self, stream):
            self.stream = stream
            self.seeks = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def seek(self, *args):
            self.seeks += 1
            if self.seeks > 1:
                raise OSError(22, "Invalid argument")
            return self.stream.seek(*args)

    def open_output(path, mode='r', *args, **kwargs):
        stream = real_open(path, mode, *args, **kwargs)
        if path == temporary_path and mode == 'wb':
            return NonseekableWriter(stream)
        return stream

    monkeypatch.setattr(training.Path, 'open', open_output)
    output_path = write_behavior_embedding(np.eye(2), {"0": "A", "1": "B"}, [5, 5], {}, tmp_path)

    with np.load(output_path, allow_pickle=False) as artifact:
        np.testing.assert_array_equal(artifact['vectors'], np.eye(2))


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


def test_main_preserves_distinct_string_order_ids(monkeypatch, tmp_path, capsys):
    (tmp_path / "order_item.csv").write_text(
        "order_id,dt,prod_id\n001,20261001,A\n1,20261001,B\n", encoding="utf-8"
    )
    (tmp_path / "item2index.json").write_text(json.dumps({"A": 0, "B": 1}))
    (tmp_path / "index2item.json").write_text(json.dumps({"0": "A", "1": "B"}))
    from item2vec.preprocessing import preprocess_baskets
    with pytest.raises(ValueError, match="no qualifying baskets"):
        preprocess_baskets(tmp_path, tmp_path, tmp_path / "processed")
    with pytest.raises(FileNotFoundError):
        training.main([str(tmp_path), str(tmp_path)])
    assert not (tmp_path / "behavior_item.npz").exists()


@pytest.mark.parametrize(
    "csv,exception,message",
    [
        ("order_id,dt,prod_id\n", ValueError, "empty"),
        ("dt,prod_id\nd1,A\nd1,B\n", ValueError, "order_id"),
        ("order_id,dt,prod_id\no1,20261001,A\no1,20261001,B\n", ValueError, "no qualifying baskets"),
    ],
)
def test_main_preserves_existing_artifact_when_production_data_is_insufficient(tmp_path, csv, exception, message):
    (tmp_path / "order_item.csv").write_text(csv)
    (tmp_path / "item2index.json").write_text(json.dumps({"A": 0, "B": 1}))
    (tmp_path / "index2item.json").write_text(json.dumps({"0": "A", "1": "B"}))
    existing = tmp_path / "behavior_item.npz"
    existing.write_bytes(b"old artifact")

    from item2vec.preprocessing import preprocess_baskets
    with pytest.raises(exception, match=message):
        preprocess_baskets(tmp_path, tmp_path, tmp_path / "processed")

    assert existing.read_bytes() == b"old artifact"
